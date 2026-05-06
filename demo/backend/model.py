"""
Model loading and inference for the Radiology Report Generator.
Based on the proven Streamlit inference pipeline.

Architecture:
- Image Encoder: DenseNet-121 (CheXNet pre-trained) → AvgPool → Dense(512)
- Two images encoded with shared weights, concatenated → (batch, 18, 512)
- Decoder: GRU with Bahdanau (additive) global attention
- Greedy search decoding, max 29 tokens
"""

import os
import pickle
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, GRU, Embedding, Input, Concatenate, Add,
    BatchNormalization, Dropout, AveragePooling2D,
    GlobalAveragePooling2D
)
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = os.environ.get("MODEL_BASE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
CHEXNET_WEIGHTS = os.path.join(BASE_DIR, "Radiography", "Copy of Copy of brucechou1983_CheXNet_Keras_0.3.0_weights.h5")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "Radiography", "Attention_with_cheXnet.weights.h5")
MODEL_H5 = os.path.join(BASE_DIR, "Radiography", "Attention_with_cheXNet_full_model1.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "Radiography", "tokenizer", "tokenizer.pickle")

INPUT_SIZE = (224, 224)
DENSE_DIM = 512
EMBEDDING_DIM = 300
MAX_PAD = 28
BATCH_SIZE = 32
DROPOUT_RATE = 0.2


# ─── CheXNet (DenseNet-121 pretrained) ──────────────────────────────────────

def create_chexnet(chexnet_weights=CHEXNET_WEIGHTS, input_size=INPUT_SIZE):
    """
    Create CheXNet: DenseNet-121 pretrained on ChestX-ray14.
    Returns feature maps before global average pooling (7x7x1024).
    """
    model = tf.keras.applications.DenseNet121(
        include_top=False,
        input_shape=input_size + (3,)
    )
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(14, activation="sigmoid", name="chexnet_output")(x)

    chexnet = tf.keras.Model(inputs=model.input, outputs=x)
    chexnet.load_weights(chexnet_weights)

    # Take layer before global avg pooling for spatial features
    chexnet = tf.keras.Model(
        inputs=model.input,
        outputs=chexnet.layers[-3].output
    )
    return chexnet


# ─── Custom Layers ───────────────────────────────────────────────────────────

class Image_encoder(tf.keras.layers.Layer):
    """Encodes X-ray image using CheXNet backbone → (batch, 9, 1024)"""

    def __init__(self, name="image_encoder_block"):
        # NOTE: name is NOT passed to super().__init__() to match original training code.
        # Keras auto-generates name as 'image_encoder' from class name.
        super().__init__()
        self.chexnet = create_chexnet(input_size=INPUT_SIZE)
        self.chexnet.trainable = False
        self.avgpool = AveragePooling2D()

    def call(self, data):
        op = self.chexnet(data)       # (batch, 7, 7, 1024)
        op = self.avgpool(op)         # (batch, 3, 3, 1024)
        op = tf.reshape(op, shape=(-1, op.shape[1] * op.shape[2], op.shape[3]))  # (batch, 9, 1024)
        return op


class global_attention(tf.keras.layers.Layer):
    """Bahdanau (additive) global attention"""

    def __init__(self, dense_dim=DENSE_DIM):
        super().__init__()
        self.dense_dim = dense_dim
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


class One_Step_Decoder(tf.keras.layers.Layer):
    """Decodes a single token using GRU + attention"""

    def __init__(self, vocab_size, embedding_dim, dense_dim, name="onestepdecoder"):
        # NOTE: name NOT passed to super() to match original training code
        super().__init__()
        self.dense_dim = dense_dim
        self.embedding = Embedding(
            input_dim=vocab_size + 1,
            output_dim=embedding_dim,
            mask_zero=True,
            name='onestepdecoder_embedding'
        )
        self.LSTM = GRU(
            units=dense_dim,
            return_state=True,
            name='onestepdecoder_LSTM'
        )
        self.attention = global_attention(dense_dim=dense_dim)
        self.concat = Concatenate(axis=-1)
        self.dense = Dense(dense_dim, name='onestepdecoder_embedding_dense', activation='relu')
        self.final = Dense(vocab_size + 1, activation='softmax')
        self.concat = Concatenate(axis=-1)  # re-assigned in original code
        self.add = Add()    # present in original but unused in call()

    @tf.function
    def call(self, input_to_decoder, encoder_output, decoder_h):
        embedding_op = self.embedding(input_to_decoder)
        context_vector, attention_weights = self.attention(encoder_output, decoder_h)
        context_vector_time_axis = tf.expand_dims(context_vector, axis=1)
        concat_input = self.concat([context_vector_time_axis, embedding_op])
        output, decoder_h = self.LSTM(concat_input, initial_state=decoder_h)
        output = self.final(output)
        return output, decoder_h, attention_weights


class decoder(tf.keras.Model):
    """Full decoder: loops One_Step_Decoder over max_pad steps"""

    def __init__(self, max_pad, embedding_dim, dense_dim, batch_size, vocab_size):
        super().__init__()
        self.onestepdecoder = One_Step_Decoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            dense_dim=dense_dim
        )
        self.output_array = tf.TensorArray(tf.float32, size=max_pad)
        self.max_pad = max_pad
        self.batch_size = batch_size
        self.dense_dim = dense_dim

    @tf.function
    def call(self, encoder_output, caption):
        decoder_h, decoder_c = tf.zeros_like(encoder_output[:, 0]), tf.zeros_like(encoder_output[:, 0])
        output_array = tf.TensorArray(tf.float32, size=self.max_pad)
        for timestep in range(self.max_pad):
            output, decoder_h, attention_weights = self.onestepdecoder(
                caption[:, timestep:timestep + 1], encoder_output, decoder_h
            )
            output_array = output_array.write(timestep, output)
        self.output_array = tf.transpose(output_array.stack(), [1, 0, 2])
        return self.output_array


# ─── Encoder function ────────────────────────────────────────────────────────

def encoder(image1, image2, dense_dim=DENSE_DIM, dropout_rate=DROPOUT_RATE):
    """Encode two X-ray images into a combined feature representation"""
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


# ─── Model Loading ───────────────────────────────────────────────────────────

class RadiologyModel:
    """Wraps the trained model and tokenizer for inference"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.vocab_size = None
        self._loaded = False

    def load(self):
        """Load model weights and tokenizer"""
        if self._loaded:
            return

        logger.info("Loading tokenizer...")
        with open(TOKENIZER_PATH, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.vocab_size = len(self.tokenizer.word_index)
        logger.info(f"Tokenizer loaded. Vocabulary size: {self.vocab_size}")

        logger.info("Building model architecture...")
        tf.keras.backend.clear_session()

        image1 = Input(shape=(INPUT_SIZE + (3,)))
        image2 = Input(shape=(INPUT_SIZE + (3,)))
        caption = Input(shape=(MAX_PAD,))

        encoder_output = encoder(image1, image2, DENSE_DIM, DROPOUT_RATE)
        output = decoder(MAX_PAD, EMBEDDING_DIM, DENSE_DIM, BATCH_SIZE, self.vocab_size)(encoder_output, caption)

        self.model = tf.keras.Model(inputs=[image1, image2, caption], outputs=output)

        logger.info(f"Loading weights from: {MODEL_H5}")
        self.model.load_weights(MODEL_H5)
        logger.info("Model loaded successfully!")

        self._loaded = True

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess uploaded image bytes for model input"""
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize(INPUT_SIZE, Image.LANCZOS)
        image = np.array(image)
        # Convert grayscale to 3-channel
        image = np.stack([image] * 3, axis=-1)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def generate_report(self, image1_bytes: bytes, image2_bytes: bytes = None) -> dict:
        """
        Generate radiology report from X-ray image(s).
        Returns dict with report text, attention weights, and metadata.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Preprocess images
        img1 = self.preprocess_image(image1_bytes)
        img2 = self.preprocess_image(image2_bytes) if image2_bytes else img1

        # Encode images through the model layers
        enc_op1 = self.model.get_layer('image_encoder')(img1)
        enc_op2 = self.model.get_layer('image_encoder')(img2)
        feat1 = self.model.get_layer('bkdense')(enc_op1)
        feat2 = self.model.get_layer('bkdense')(enc_op2)
        concat = self.model.get_layer('concatenate')([feat1, feat2])
        enc_op = self.model.get_layer('encoder_batch_norm')(concat)
        enc_op = self.model.get_layer('encoder_dropout')(enc_op)

        # Greedy search decoding
        decoder_h = tf.zeros_like(enc_op[:, 0])
        predicted_ids = []
        attention_weights_list = []
        max_repeat = 3
        repeat_count = 0
        last_predicted_id = None

        for i in range(MAX_PAD + 1):
            if i == 0:
                caption = np.array(self.tokenizer.texts_to_sequences(['<cls>']))
            else:
                caption = np.array([[predicted_id]])

            output, decoder_h, attention_weights = self.model.get_layer('decoder').onestepdecoder(
                caption, enc_op, decoder_h
            )
            attention_weights_list.append(attention_weights.numpy())

            predicted_id = tf.argmax(output, axis=-1).numpy().flatten()[0]

            # Repetition guard
            if predicted_id == last_predicted_id:
                repeat_count += 1
            else:
                repeat_count = 0
            last_predicted_id = predicted_id

            if repeat_count >= max_repeat:
                break

            if predicted_id == self.tokenizer.word_index.get('<end>', 1):
                break

            predicted_ids.append(predicted_id)

        # Decode tokens to text
        predicted_text = self.tokenizer.sequences_to_texts([predicted_ids])[0]

        # Clean up the text
        predicted_text = predicted_text.replace('<cls>', '').replace('<end>', '').strip()
        predicted_text = ' '.join(predicted_text.split())  # normalize whitespace

        return {
            "report": predicted_text,
            "attention_weights": attention_weights_list,
            "num_tokens": len(predicted_ids),
        }

    def generate_attention_map(self, image_bytes: bytes, attention_weights_list: list) -> str:
        """
        Generate attention heatmap overlay on the original image.
        Returns base64-encoded PNG.
        """
        # Load original image
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image_array = np.array(image)

        # Average attention weights across all decoding steps
        if len(attention_weights_list) > 0:
            combined = np.mean(attention_weights_list, axis=0)
            attention = np.squeeze(combined)

            # Reshape to 2D grid
            att_len = attention.shape[0]
            grid_size = int(np.ceil(np.sqrt(att_len)))
            padded = np.pad(attention, (0, grid_size**2 - att_len))
            attention_map = padded.reshape(grid_size, grid_size)

            # Resize to image dimensions
            attention_map = tf.image.resize(
                tf.expand_dims(tf.expand_dims(attention_map, axis=0), axis=-1),
                (image_array.shape[0], image_array.shape[1])
            )
            attention_map = np.squeeze(attention_map.numpy())
        else:
            attention_map = np.zeros_like(image_array, dtype=np.float32)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#0a0a0a')

        ax1.imshow(image_array, cmap='gray')
        ax1.set_title("Original X-ray", color='white', fontsize=14, fontweight='bold')
        ax1.axis('off')

        ax2.imshow(image_array, cmap='gray')
        im = ax2.imshow(attention_map, cmap='hot', alpha=0.5)
        ax2.set_title("Attention Heatmap", color='white', fontsize=14, fontweight='bold')
        ax2.axis('off')

        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='#0a0a0a', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')


# ─── Severity Scoring ────────────────────────────────────────────────────────

SEVERITY_TERMS = {
    "pneumothorax": 8.5, "tuberculosis": 7.5,
    "pneumonia": 6.0, "cardiomegaly": 5.5, "effusion": 5.5, "edema": 5.0,
    "consolidation": 4.5, "emphysema": 4.0, "nodule": 4.0, "fibrosis": 4.0,
    "atelectasis": 3.5,
    "opacity": 2.5, "calcification": 2.0, "granuloma": 2.0,
}

MEDICAL_TERMS = {
    "cardiopulmonary": "Relating to the heart and lungs",
    "pneumothorax": "Collapsed lung",
    "effusion": "Fluid accumulation",
    "consolidation": "Lung tissue filled with liquid",
    "atelectasis": "Collapsed or closed air sacs in the lungs",
    "cardiomegaly": "Enlarged heart",
    "opacity": "Area of increased density on X-ray",
    "edema": "Swelling caused by excess fluid",
    "emphysema": "Lung condition causing shortness of breath",
    "nodule": "Small, round growth or lump",
    "tuberculosis": "Infectious disease primarily affecting the lungs",
    "pleural": "Relating to the membrane covering the lungs",
    "infiltrate": "Substance abnormally accumulating in tissue",
    "bronchial": "Relating to the airways in the lungs",
    "pulmonary": "Relating to the lungs",
    "vascular": "Relating to blood vessels",
    "mediastinal": "Relating to the area between the lungs",
    "hilar": "Relating to where vessels and airways enter the lungs",
    "pneumonia": "Infection that inflames air sacs in the lungs",
    "fibrosis": "Thickening and scarring of connective tissue",
    "calcification": "Accumulation of calcium in soft tissue",
    "granuloma": "Small area of inflammation in tissue",
    "emphysematous": "Relating to emphysema",
    "thoracic": "Relating to the chest",
    "bilateral": "Affecting both sides",
    "interstitial": "Relating to tissue around the air sacs",
    "disease": "A disorder of structure or function",
}


def compute_severity(report_text: str) -> dict:
    """Compute severity score and recommendations from report text"""
    words = report_text.lower().split()
    max_score = 0.0

    for word in words:
        clean = word.strip('.,;:')
        if clean in SEVERITY_TERMS:
            max_score = max(max_score, SEVERITY_TERMS[clean])

    score = min(10.0, max(0.0, max_score))

    if score >= 7.0:
        urgency = "IMMEDIATE ATTENTION REQUIRED"
        followup = "Emergency consultation recommended"
    elif score >= 5.0:
        urgency = "URGENT ATTENTION NEEDED"
        followup = "Follow-up within 24-48 hours"
    elif score >= 3.0:
        urgency = "PROMPT FOLLOW-UP"
        followup = "Follow-up within 1 week"
    else:
        urgency = "ROUTINE"
        followup = "Routine follow-up (3-6 months)"

    # Extract key findings
    findings = []
    for word in words:
        clean = word.strip('.,;:')
        if clean in MEDICAL_TERMS and clean not in findings:
            findings.append(clean)

    return {
        "score": round(score, 1),
        "urgency": urgency,
        "followup": followup,
        "key_findings": findings,
        "findings_explained": {f: MEDICAL_TERMS[f] for f in findings},
    }
