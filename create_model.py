import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, GRU, Embedding, Concatenate, Add, AveragePooling2D, \
    BatchNormalization, Dropout


class Image_encoder(Layer):
    def __init__(self, name="image_encoder_block"):
        super().__init__(name=name)
        self.chexnet = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet')
        self.chexnet.trainable = False
        self.avgpool = AveragePooling2D()

    def call(self, data):
        op = self.chexnet(data)
        op = self.avgpool(op)
        shape = tf.shape(op)
        op = tf.reshape(op, shape=(-1, shape[1] * shape[2], shape[3]))
        return op


class global_attention(Layer):
    def __init__(self, dense_dim=512):
        super().__init__()
        self.W1 = Dense(units=dense_dim)
        self.W2 = Dense(units=dense_dim)
        self.V = Dense(units=1)

    def call(self, encoder_output, decoder_h):
        decoder_h = tf.expand_dims(decoder_h, axis=1)
        tanh_input = self.W1(encoder_output) + self.W2(decoder_h)
        tanh_output = tf.nn.tanh(tanh_input)
        attention_weights = tf.nn.softmax(self.V(tanh_output), axis=1)
        op = attention_weights * encoder_output
        context_vector = tf.reduce_sum(op, axis=1)
        return context_vector, attention_weights


class One_Step_Decoder(Layer):
    def __init__(self, vocab_size, embedding_dim, dense_dim, name="onestepdecoder"):
        super().__init__(name=name)
        self.dense_dim = dense_dim
        self.embedding = Embedding(input_dim=vocab_size + 1,
                                   output_dim=embedding_dim,
                                   mask_zero=True)
        self.LSTM = GRU(units=self.dense_dim,
                        return_state=True,
                        name='onestepdecoder_LSTM')
        self.attention = global_attention(dense_dim=dense_dim)
        self.concat = Concatenate(axis=-1)
        self.dense = Dense(dense_dim, activation='relu')
        self.final = Dense(vocab_size + 1, activation='softmax')
        self.add = Add()

    def call(self, input_to_decoder, encoder_output, decoder_h):
        embedding_op = self.embedding(input_to_decoder)
        context_vector, attention_weights = self.attention(encoder_output, decoder_h)
        context_vector_time_axis = tf.expand_dims(context_vector, axis=1)
        concat_input = self.concat([context_vector_time_axis, embedding_op])
        output, decoder_h = self.LSTM(concat_input, initial_state=decoder_h)
        output = self.final(output)
        return output, decoder_h, attention_weights


class Decoder(Layer):
    def __init__(self, max_pad, embedding_dim, dense_dim, vocab_size, name="decoder"):
        super().__init__(name=name)
        self.onestepdecoder = One_Step_Decoder(vocab_size, embedding_dim, dense_dim)
        self.max_pad = max_pad

    def call(self, decoder_input, encoder_output):
        decoder_h = tf.zeros_like(encoder_output[:, 0])
        output, decoder_h, attention_weights = self.onestepdecoder(
            decoder_input, encoder_output, decoder_h)
        return output, attention_weights


def encoder(image1, image2, dense_dim=512, dropout_rate=0.2):
    im_encoder = Image_encoder()
    bkfeat1 = im_encoder(image1)
    bk_dense = Dense(dense_dim, name='bkdense', activation='relu')
    bkfeat1 = bk_dense(bkfeat1)
    bkfeat2 = im_encoder(image2)
    bkfeat2 = bk_dense(bkfeat2)
    concat = Concatenate(axis=1)([bkfeat1, bkfeat2])
    bn = BatchNormalization(name="encoder_batch_norm")(concat)
    dropout = Dropout(dropout_rate, name="encoder_dropout")(bn)
    return dropout
