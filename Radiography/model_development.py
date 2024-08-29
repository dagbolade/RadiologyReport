from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Concatenate, GlobalAveragePooling2D, LSTM, Embedding, GlobalAveragePooling1D, \
    Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications import DenseNet121, InceptionV3, VGG16, ResNet50
import numpy as np
import tensorflow as tf


def split_data(preprocessed_images, preprocessed_reports, test_size=0.2, random_state=42):
    train_images, test_images, train_reports, test_reports = train_test_split(
        preprocessed_images, preprocessed_reports, test_size=test_size, random_state=random_state)
    return train_images, test_images, train_reports, test_reports


def create_cnn_model(input_shape):
    #base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    # using inceptionv3 model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Adding dropout
    x = Dense(128, activation='relu')(x)
    cnn_model = Model(inputs=base_model.input, outputs=x)
    return cnn_model


def create_rnn_model(max_length, vocab_size):
    input_ids = Input(shape=(max_length,), dtype='int32', name='input_ids')
    x = Embedding(input_dim=vocab_size, output_dim=128)(input_ids)
    x = LSTM(128, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)  # Adding dropout
    x = Dense(128, activation='relu')(x)
    rnn_model = Model(inputs=input_ids, outputs=x)
    return rnn_model


def create_model(cnn_model, rnn_model, vocab_size, max_length):
    image_input = Input(shape=cnn_model.input_shape[1:])
    image_features = cnn_model(image_input)

    input_ids = Input(shape=(max_length,), dtype='int32', name='input_ids')
    report_features = rnn_model(input_ids)

    combined_features = Concatenate()([image_features, report_features])
    dense_layer = Dense(256, activation='relu')(combined_features)

    # Adjust the output layer to produce the correct shape directly
    output_layer = Dense(max_length * vocab_size, activation='softmax')(dense_layer)
    output_layer = Reshape((max_length, vocab_size))(output_layer)

    findings_output = Dense(max_length * vocab_size, activation='softmax')(dense_layer)
    findings_output = Reshape((max_length, vocab_size), name='findings_output')(findings_output)

    impression_output = Dense(max_length * vocab_size, activation='softmax')(dense_layer)
    impression_output = Reshape((max_length, vocab_size), name='impression_output')(impression_output)

    model = Model(inputs=[image_input, input_ids], outputs=[findings_output, impression_output])
    return model


def train_model(model, train_images, train_report_ids, epochs, batch_size, train_findings_labels, train_impression_labels):
    train_images = np.array(train_images)
    train_report_ids = np.array(train_report_ids)
    train_findings_labels = np.array(train_findings_labels)
    train_impression_labels = np.array(train_impression_labels)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=[['accuracy'], ['accuracy']])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('v3.keras', save_best_only=True)

    print(f"Shape of train_images: {train_images.shape}")
    print(f"Shape of train_report_ids: {train_report_ids.shape}")
    print(f"Shape of train_findings_labels: {train_findings_labels.shape}")
    print(f"Shape of train_impression_labels: {train_impression_labels.shape}")

    model.fit([train_images, train_report_ids], [train_findings_labels, train_impression_labels],
              epochs=int(epochs), batch_size=batch_size, validation_split=0.1,
              callbacks=[early_stopping, model_checkpoint])

    return model

def evaluate_model(model, test_images, test_report_ids, test_findings_labels, test_impression_labels):
    loss, findings_loss, impression_loss, findings_acc, impression_acc = model.evaluate([test_images, test_report_ids],
                                                                                        [test_findings_labels,
                                                                                         test_impression_labels])
    print("Test Loss:", loss)
    print("Findings Loss:", findings_loss)
    print("Impression Loss:", impression_loss)
    print("Findings Accuracy:", findings_acc)
    print("Impression Accuracy:", impression_acc)
