import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import joblib
import os
import time

# Constants
max_pad = 28
input_size = (224, 224)
dense_dim = 512
embedding_dim = 300


# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    st.write("Loading model and tokenizer...")
    model_path = 'Radiography/Attention_with_cheXNet_full_model'
    tokenizer_path = 'Radiography/tokenizer/tokenizer.pickle'

    try:
        model = tf.keras.models.load_model(model_path, compile=True)
        st.success("Model loaded successfully!")
        #st.write("Model summary:")
        #model.summary(print_fn=lambda x: st.text(x))
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


# Image preprocessing
def preprocess_image(image):
    image = cv2.resize(image, input_size, interpolation=cv2.INTER_NEAREST)
    image = image.astype(np.float32) / 255.0
    return image


# Greedy search prediction
def greedy_search_predict(image1, image2, model, tokenizer):
    image1 = tf.expand_dims(cv2.resize(image1, input_size, interpolation=cv2.INTER_NEAREST), axis=0)
    image2 = tf.expand_dims(cv2.resize(image2, input_size, interpolation=cv2.INTER_NEAREST), axis=0)
    image1 = model.get_layer('image_encoder')(image1)
    image2 = model.get_layer('image_encoder')(image2)
    image1 = model.get_layer('bkdense')(image1)
    image2 = model.get_layer('bkdense')(image2)

    concat = model.get_layer('concatenate')([image1, image2])
    enc_op = model.get_layer('encoder_batch_norm')(concat)
    enc_op = model.get_layer('encoder_dropout')(enc_op)

    decoder_h = tf.zeros_like(enc_op[:, 0])
    a = []
    for i in range(max_pad):
        if i == 0:
            caption = np.array(tokenizer.texts_to_sequences(['<cls>']))

        # Cast caption to float32
        caption = tf.cast(caption, dtype=tf.float32)

        output, decoder_h, _ = model.get_layer('decoder').onestepdecoder(caption, enc_op, decoder_h)

        max_prob = tf.argmax(output, axis=-1)
        caption = np.array([max_prob])

        # Cast max_prob to float32
        max_prob = tf.cast(max_prob, dtype=tf.float32)

        if max_prob == np.squeeze(tokenizer.texts_to_sequences(['<end>'])):
            break
        else:
            a.append(tf.squeeze(max_prob).numpy())

    return tokenizer.sequences_to_texts([a])[0]


# Predict based on images
def predict(image1, image2=None, model_tokenizer=None):
    if image2 is None:
        image2 = image1
    model, tokenizer = model_tokenizer
    predicted_caption = greedy_search_predict(image1, image2, model, tokenizer)
    return predicted_caption


# Function to predict on uploaded images
def predict_on_upload(image_1, image_2, model_tokenizer):
    if image_1 is not None:
        image_1 = Image.open(image_1).convert("RGB")
        image_1 = np.array(image_1) / 255
        if image_2 is None:
            image_2 = image_1
        else:
            image_2 = Image.open(image_2).convert("RGB")
            image_2 = np.array(image_2) / 255
        st.image([image_1, image_2], width=300)
        caption = predict(image_1, image_2, model_tokenizer)
        st.markdown("### **Impression:**")
        st.write(caption)


# Function to predict on sample data


# Streamlit app setup
st.title("Chest X-ray Report Generator")
st.markdown("<small>by David</small>", unsafe_allow_html=True)

st.markdown(
    "This app will generate the impression part of an X-ray report. You can upload 2 X-rays that are front view and side view of the chest of the same individual. The 2nd X-ray is optional.")

col1, col2 = st.columns(2)
image_1 = col1.file_uploader("X-ray 1", type=['png', 'jpg', 'jpeg'])
image_2 = col2.file_uploader("X-ray 2 (optional)", type=['png', 'jpg', 'jpeg'])

predict_button = st.button('Predict on uploaded files')

model_tokenizer = load_model_and_tokenizer()

if predict_button:
    predict_on_upload(image_1, image_2, model_tokenizer)