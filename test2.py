import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import joblib
import cv2
import matplotlib.pyplot as plt
from create_model import Image_encoder, global_attention, One_Step_Decoder, Decoder, encoder

# Constants
max_pad = 28
input_size = (224, 224)
dense_dim = 512
embedding_dim = 300


# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_path = 'Radiography/Attention_with_cheXNet_full_model'
    tokenizer_path = 'Radiography/tokenizer/tokenizer.pickle'

    custom_objects = {
        'Image_encoder': Image_encoder,
        'global_attention': global_attention,
        'One_Step_Decoder': One_Step_Decoder,
        'Decoder': Decoder,
        'encoder': encoder
    }

    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

    try:
        tokenizer = joblib.load(tokenizer_path)
        st.success("Tokenizer loaded successfully!")
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        return model, None

    return model, tokenizer


# Image processing
def process_image(image):
    image = cv2.resize(image, input_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# Prediction function
def predict(image1, image2, model, tokenizer):
    image1 = process_image(image1)
    image2 = process_image(image2)

    # Get the encoder layers
    image_encoder = model.get_layer('image_encoder')
    bkdense = model.get_layer('bkdense')
    concatenate = model.get_layer('concatenate')
    encoder_batch_norm = model.get_layer('encoder_batch_norm')
    encoder_dropout = model.get_layer('encoder_dropout')

    # Perform encoding
    enc_op1 = image_encoder(image1)
    enc_op2 = image_encoder(image2)
    bkfeat1 = bkdense(enc_op1)
    bkfeat2 = bkdense(enc_op2)
    concat = concatenate([bkfeat1, bkfeat2])
    enc_op = encoder_batch_norm(concat)
    encoder_output = encoder_dropout(enc_op)

    decoder = model.get_layer('decoder')

    # Initialize with <cls> token and pad to max_pad length
    initial_input = [tokenizer.word_index['<cls>']] + [0] * (max_pad - 1)
    decoder_input = tf.expand_dims(initial_input, 0)

    result = []
    attention_weights_list = []

    for i in range(max_pad):
        predictions = decoder(encoder_output, decoder_input, training=False)
        predicted_id = tf.argmax(predictions[0, i], axis=-1).numpy()

        if predicted_id == tokenizer.word_index.get('<end>', 1) or i == max_pad - 1:
            break

        result.append(predicted_id)

        # Update decoder_input for the next iteration
        decoder_input = tf.concat([decoder_input[:, :i + 1],
                                   tf.expand_dims([predicted_id], 0),
                                   decoder_input[:, i + 2:]], axis=1)

    predicted_sequence = tokenizer.sequences_to_texts([result])[0]

    # For attention weights, we'll need to modify the model to output them
    # For now, we'll return a placeholder
    attention_weights = tf.ones((1, 18, 512))  # placeholder

    return predicted_sequence, attention_weights

def visualize_attention(image, attention_weights):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    ax1.imshow(image, cmap='gray')
    ax1.set_title("Original X-ray")
    ax1.axis('off')

    attention_weights = tf.reduce_mean(attention_weights, axis=1)
    attention_weights = tf.reshape(attention_weights, (attention_weights.shape[1], 1))
    attention_map = tf.keras.layers.UpSampling2D(size=(8, 8))(attention_weights[tf.newaxis, ..., tf.newaxis])
    attention_map = tf.squeeze(attention_map)

    ax2.imshow(image, cmap='gray')
    ax2.imshow(attention_map, cmap='hot', alpha=0.5)
    ax2.set_title("Attention Heatmap")
    ax2.axis('off')

    plt.tight_layout()
    return fig


# Streamlit UI
st.title("Chest X-ray Report Generator")

model, tokenizer = load_model_and_tokenizer()

if model is not None and tokenizer is not None:
    st.write("Upload two chest X-ray images:")
    col1, col2 = st.columns(2)
    with col1:
        image_file1 = st.file_uploader("Upload first X-ray image", type=['png', 'jpg', 'jpeg'])
    with col2:
        image_file2 = st.file_uploader("Upload second X-ray image", type=['png', 'jpg', 'jpeg'])

    if image_file1 is not None and image_file2 is not None:
        image1 = Image.open(image_file1).convert("RGB")
        image2 = Image.open(image_file2).convert("RGB")

        st.image([image1, image2], caption=['Image 1', 'Image 2'], width=300)

        if st.button("Generate Report"):
            with st.spinner("Analyzing X-rays and generating report..."):
                predicted_text, attention_weights = predict(np.array(image1), np.array(image2), model, tokenizer)

                st.subheader("Generated Report:")
                st.write(predicted_text)

                st.subheader("Attention Visualization:")
                fig = visualize_attention(np.array(image1), attention_weights.numpy())
                st.pyplot(fig)
else:
    st.error("Failed to load model or tokenizer. Please check your file paths and model compatibility.")

st.markdown("---")
st.markdown("Developed by [Your Name]")