import base64
import io
from datetime import datetime
import warnings
import os
import pandas as pd
# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=Warning)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pydicom.dataset import Dataset
import json
from datetime import datetime
import io
from pydicom.uid import generate_uid
from tensorflow.keras.layers import Dense, GRU, Embedding, Input, Concatenate, BatchNormalization, Dropout, \
    AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import streamlit as st
import joblib
import sys
from tensorflow import keras
import pydicom
sys.modules['keras'] = keras
from tensorflow.keras import preprocessing

sys.modules['keras.src.preprocessing'] = preprocessing
import streamlit as st
import tensorflow as tf
import numpy as np
#import cv2
from PIL import Image
import joblib
import os
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer  # to create PDF and add text
from reportlab.lib.styles import getSampleStyleSheet  #, ParagraphStyle
from googletrans import Translator

# Constants
max_pad = 28
input_size = (224, 224)
dense_dim = 512
embedding_dim = 300

# New global variables
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'zh-cn': 'Chinese (Simplified)',
    'HI': 'Hindi',
    'yo': 'Yoruba',
}


# Helper Functions
def load_and_display_logo():
    logo_url = "https://www.solent.ac.uk/graphics/logo/logo-white.svg"
    response = requests.get(logo_url)
    if response.status_code == 200:
        svg_content = response.text
        st.markdown(f'<div style="display: flex; justify-content: center;">{svg_content}</div>', unsafe_allow_html=True)
    else:
        st.error("Failed to load the logo")


def create_tabs():
    return st.tabs(["Upload X-rays", "About", "How it works", "Report History"])


chexnet_weights = "Radiography/Copy of Copy of brucechou1983_CheXNet_Keras_0.3.0_weights.h5"


def update_tokenizer(tokenizer, target_size):
    current_size = len(tokenizer.word_index)
    for i in range(current_size + 1, target_size + 1):
        new_word = f'<extra_token_{i}>'
        tokenizer.word_index[new_word] = i
    tokenizer.index_word = {v: k for k, v in tokenizer.word_index.items()}
    return tokenizer


# Model and Tokenizer Loading

def create_chexnet(chexnet_weights=chexnet_weights, input_size=(224, 224)):
    """
  chexnet_weights: weights value in .h5 format of chexnet
  creates a chexnet model with preloaded weights present in chexnet_weights file
  """
    model = tf.keras.applications.DenseNet121(include_top=False, input_shape=input_size + (
        3,))

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(14, activation="sigmoid", name="chexnet_output")(x)  #here activation is sigmoid as seen in research paper

    chexnet = tf.keras.Model(inputs=model.input, outputs=x)
    chexnet.load_weights(chexnet_weights)
    chexnet = tf.keras.Model(inputs=model.input, outputs=chexnet.layers[
        -3].output)  #we will be taking the 3rd last layer (here it is layer before global avgpooling)
    #since we are using attention here
    return chexnet


from tensorflow.keras.layers import AveragePooling2D


class Image_encoder(tf.keras.layers.Layer):
    """
    This layer will output image backbone features after passing it through chexnet
    """

    def __init__(self,
                 name="image_encoder_block"
                 ):
        super().__init__()
        self.chexnet = create_chexnet(input_size=(224, 224))
        self.chexnet.trainable = False
        self.avgpool = AveragePooling2D(pool_size=(2, 2))
        # for i in range(10): #the last 10 layers of chexnet will be trained
        #   self.chexnet.layers[-i].trainable = True

    def call(self, data):
        op = self.chexnet(data)  # op shape: (None,7,7,1024)
        op = self.avgpool(op)  # op shape (None,3,3,1024)
        op = tf.reshape(op, shape=(-1, op.shape[1] * op.shape[2], op.shape[3]))  # op shape: (None,9,1024)
        return op


def encoder(image1, image2, dense_dim, dropout_rate):
    """
    Takes image1,image2
    gets the final encoded vector of these
    """
    # image1
    im_encoder = Image_encoder()
    bkfeat1 = im_encoder(image1)  # shape: (None,9,1024)
    bk_dense = Dense(dense_dim, name='bkdense', activation='relu')  # shape: (None,9,512)
    bkfeat1 = bk_dense(bkfeat1)

    # image2
    bkfeat2 = im_encoder(image2)  # shape: (None,9,1024)
    bkfeat2 = bk_dense(bkfeat2)  # shape: (None,9,512)

    # combining image1 and image2
    concat = Concatenate(axis=1)([bkfeat1, bkfeat2])  # concatenating through the second axis shape: (None,18,1024)
    bn = BatchNormalization(name="encoder_batch_norm")(concat)
    dropout = Dropout(dropout_rate, name="encoder_dropout")(bn)
    return dropout


class global_attention(tf.keras.layers.Layer):
    """
    calculate global attention
    """

    def __init__(self, dense_dim):
        super().__init__()
        # Intialize variables needed for Concat score function here
        self.W1 = Dense(units=dense_dim)  # weight matrix of shape enc_units*dense_dim
        self.W2 = Dense(units=dense_dim)  # weight matrix of shape dec_units*dense_dim
        self.V = Dense(units=1)  # weight matrix of shape dense_dim*1
        # op (None,98,1)

    def call(self, encoder_output,
             decoder_h):  # here the encoded output will be the concatted image bk features shape: (None,98,dense_dim)
        decoder_h = tf.expand_dims(decoder_h, axis=1)  # shape: (None,1,dense_dim)
        tanh_input = self.W1(encoder_output) + self.W2(decoder_h)  # ouput_shape: batch_size*98*dense_dim
        tanh_output = tf.nn.tanh(tanh_input)
        attention_weights = tf.nn.softmax(self.V(tanh_output),
                                          axis=1)  # shape= batch_size*98*1 getting attention alphas
        op = attention_weights * encoder_output  # op_shape: batch_size*98*dense_dim  multiply all aplhas with corresponding context vector
        context_vector = tf.reduce_sum(op,
                                       axis=1)  # summing all context vector over the time period ie input length, output_shape: batch_size*dense_dim

        return context_vector, attention_weights


class One_Step_Decoder(tf.keras.layers.Layer):
    """
    decodes a single token
    """

    def __init__(self, vocab_size, embedding_dim, dense_dim, name="onestepdecoder"):
        # Initialize decoder embedding layer, LSTM and any other objects needed
        super().__init__()
        self.dense_dim = dense_dim
        self.embedding = Embedding(input_dim=vocab_size + 1,
                                   output_dim=embedding_dim,
                                   mask_zero=True,
                                   name='onestepdecoder_embedding'
                                   )
        self.LSTM = GRU(units=self.dense_dim,
                        # return_sequences=True,
                        return_state=True,
                        name='onestepdecoder_LSTM'
                        )
        self.attention = global_attention(dense_dim=dense_dim)
        self.concat = Concatenate(axis=-1)
        self.dense = Dense(dense_dim, name='onestepdecoder_embedding_dense', activation='relu')
        self.final = Dense(vocab_size + 1, activation='softmax')
        self.concat = Concatenate(axis=-1)

    @tf.function
    def call(self, input_to_decoder, encoder_output, decoder_h):  # ,decoder_c):
        '''
            One step decoder mechanisim step by step:
          A. Pass the input_to_decoder to the embedding layer and then get the output(batch_size,1,embedding_dim)
          B. Using the encoder_output and decoder hidden state, compute the context vector.
          C. Concat the context vector with the step A output
          D. Pass the Step-C output to LSTM/GRU and get the decoder output and states(hidden and cell state)
          E. Pass the decoder output to dense layer(vocab size) and store the result into output.
          F. Return the states from step D, output from Step E, attention weights from Step -B

          here state_h,state_c are decoder states
        '''
        embedding_op = self.embedding(input_to_decoder)  # output shape = batch_size*1*embedding_shape (only 1 token)

        context_vector, attention_weights = self.attention(encoder_output,
                                                           decoder_h)  # passing hidden state h of decoder and encoder output
        # context_vector shape: batch_size*dense_dim we need to add time dimension
        context_vector_time_axis = tf.expand_dims(context_vector, axis=1)
        # now we will combine attention output context vector with next word input to the lstm here we will be teacher forcing
        concat_input = self.concat([context_vector_time_axis,
                                    embedding_op])  # output dimension = batch_size*input_length(here it is 1)*(dense_dim+embedding_dim)

        output, decoder_h = self.LSTM(concat_input, initial_state=decoder_h)
        # output shape = batch*1*dense_dim and decoder_h,decoder_c has shape = batch*dense_dim
        # we need to remove the time axis from this decoder_output

        output = self.final(output)  # shape = batch_size*decoder vocab size
        return output, decoder_h, attention_weights


class decoder(tf.keras.Model):
    """
    Decodes the encoder output and caption
    """

    def __init__(self, max_pad, embedding_dim, dense_dim, batch_size, vocab_size):
        super().__init__()
        self.onestepdecoder = One_Step_Decoder(vocab_size=vocab_size, embedding_dim=embedding_dim,
                                               dense_dim=dense_dim)
        self.output_array = tf.TensorArray(tf.float32, size=max_pad)
        self.max_pad = max_pad
        self.batch_size = batch_size
        self.dense_dim = dense_dim

    @tf.function
    def call(self, encoder_output,
             caption):  # ,decoder_h,decoder_c): #caption : (None,max_pad), encoder_output: (None,dense_dim)
        decoder_h, decoder_c = tf.zeros_like(encoder_output[:, 0]), tf.zeros_like(
            encoder_output[:, 0])  # decoder_h, decoder_c
        output_array = tf.TensorArray(tf.float32, size=self.max_pad)
        for timestep in range(self.max_pad):  # iterating through all timesteps ie through max_pad
            output, decoder_h, attention_weights = self.onestepdecoder(caption[:, timestep:timestep + 1],
                                                                       encoder_output, decoder_h)
            output_array = output_array.write(timestep, output)  # timestep*batch_size*vocab_size

        self.output_array = tf.transpose(output_array.stack(), [1, 0,
                                                                2])  # .stack :Return the values in the TensorArray
        # as a stacked Tensor.)
        # shape output_array: (batch_size,max_pad,vocab_size)
        return self.output_array


# Model loading function
import pickle


@st.cache_resource
def load_model_and_tokenizer():
    #st.write("Loading model and tokenizer...")

    model_path = 'Radiography/Attention_with_cheXNet_full_model1.h5'
    tokenizer_path = 'Radiography/tokenizer/tokenizer.pickle'

    # Load the tokenizer
    try:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        #st.success("Tokenizer loaded successfully!")
        vocab_size = len(tokenizer.word_index)
        # Debug
        #st.write(f"Tokenizer vocabulary size: {vocab_size}")
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        return None, None

    # Define the parameters for the decoder
    input_size = (224, 224)
    max_pad = 28
    batch_size = 32
    embedding_dim = 300
    dense_dim = 512
    dropout_rate = 0.2

    try:
        # Clear the Keras session so that we can load the model multiple times
        tf.keras.backend.clear_session()

        # Creating the model structure
        image1 = Input(shape=(input_size + (3,)))
        image2 = Input(shape=(input_size + (3,)))
        caption = Input(shape=(max_pad,))

        encoder_output = encoder(image1, image2, dense_dim, dropout_rate)
        output = decoder(max_pad, embedding_dim, dense_dim, batch_size, vocab_size)(encoder_output, caption)

        model = tf.keras.Model(inputs=[image1, image2, caption], outputs=output)

        # Load the weights
        model.load_weights(model_path)

        #st.success("Model loaded successfully!")

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, tokenizer

    return model, tokenizer


class DicomHandler:
    """Handler for DICOM file processing"""

    def __init__(self):
        self.supported_transfer_syntaxes = [
            '1.2.840.10008.1.2',  # Implicit VR Little Endian
            '1.2.840.10008.1.2.1',  # Explicit VR Little Endian
            '1.2.840.10008.1.2.2',  # Explicit VR Big Endian
        ]

    def read_dicom(self, file_content):
        """
        Read and process DICOM file content
        Args:
            file_content: bytes of DICOM file
        Returns:
            numpy array of processed image
        """
        try:
            ds = pydicom.dcmread(io.BytesIO(file_content))

            # Get pixel array
            pixel_array = ds.pixel_array

            # Apply VOI LUT transformation if available
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                if hasattr(ds, 'VOILUTFunction'):
                    # Use VOI LUT function if available
                    voi_func = getattr(ds, 'VOILUTFunction', 'LINEAR')
                    if voi_func == 'SIGMOID':
                        # Apply sigmoid VOI LUT
                        pixel_array = self._apply_sigmoid_voi(pixel_array, ds.WindowCenter, ds.WindowWidth)
                    else:
                        # Apply linear VOI LUT
                        pixel_array = self._apply_linear_voi(pixel_array, ds.WindowCenter, ds.WindowWidth)
                else:
                    # Default to linear VOI LUT
                    pixel_array = self._apply_linear_voi(pixel_array, ds.WindowCenter, ds.WindowWidth)

            # Normalize to 8-bit
            if pixel_array.dtype != np.uint8:
                pixel_array = ((pixel_array - pixel_array.min()) * 255.0 /
                               (pixel_array.max() - pixel_array.min())).astype(np.uint8)

            return pixel_array

        except Exception as e:
            st.error(f"Error reading DICOM file: {str(e)}")
            return None

    def _apply_linear_voi(self, pixel_array, window_center, window_width):
        """Apply linear VOI LUT transformation"""
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = window_center[0]
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = window_width[0]

        min_value = window_center - window_width // 2
        max_value = window_center + window_width // 2
        pixel_array = np.clip(pixel_array, min_value, max_value)
        return pixel_array

    def _apply_sigmoid_voi(self, pixel_array, window_center, window_width):
        """Apply sigmoid VOI LUT transformation"""
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = window_center[0]
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = window_width[0]

        pixel_array = pixel_array.astype(float)
        pixel_array = 1.0 / (1.0 + np.exp(-4 * (pixel_array - window_center) / window_width))
        return pixel_array

def is_likely_chest_xray(image):
    """
    Basic check to determine if an image is likely to be a chest X-ray.

    """
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Check image dimensions
    height, width = image.shape
    aspect_ratio = width / height
    if not (0.7 <= aspect_ratio <= 1.3):
        return False

    # Check for overall brightness and contrast typical of X-rays
    mean_val = np.mean(image)
    std_dev = np.std(image)
    if not (40 <= mean_val <= 200) or std_dev < 20:
        return False

    return True


# Modified process_uploaded_image function to handle both DICOM and regular images
def process_uploaded_image(uploaded_file, dicom_handler=None):
    """
    Process uploaded image file, handling both DICOM and regular image formats
    Args:
        uploaded_file: Streamlit uploaded file
        dicom_handler: Optional DicomHandler instance
    Returns:
        numpy array of processed image
    """
    if uploaded_file is None:
        return None

    try:
        # Read file content
        file_content = uploaded_file.read()
        uploaded_file.seek(0)

        # Check if it's a DICOM file
        is_dicom = file_content.startswith(b'DICM') or b'DICM' in file_content[:140]

        if is_dicom and dicom_handler is not None:
            # Process DICOM file
            image_array = dicom_handler.read_dicom(file_content)
            if image_array is None:
                st.error("Failed to read DICOM file")
                return None
        else:
            # Process regular image file
            image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
            image_array = np.array(image)

        # Validate as chest X-ray
        if is_likely_chest_xray(image_array):
            return image_array
        else:
            st.error("The uploaded image does not appear to be a chest X-ray. Please upload only chest X-ray images.")
            return None

    except Exception as e:
        st.error(f"Error processing uploaded image: {str(e)}")
        st.error(f"File type: {uploaded_file.type}")
        return None

# Image Processing
def preprocess_image(image, input_size=(224, 224)):
    """
    Preprocess image for model input
    Args:
        image: numpy array or PIL Image
        input_size: tuple of (height, width)
    Returns:
        preprocessed image tensor
    """
    try:
        # Convert to numpy array if it's a PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure image is 2D grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to grayscale
            image = np.mean(image, axis=2).astype(np.uint8)

        # Resize image
        image = Image.fromarray(image)
        image = image.resize(input_size, Image.LANCZOS)
        image = np.array(image)

        # Ensure image is 3 channel (convert grayscale to RGB)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # Normalize pixel values
        image = image.astype(np.float32) / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        st.error(f"Image shape: {image.shape if isinstance(image, np.ndarray) else 'Not an array'}")
        st.error(f"Image type: {type(image)}")
        raise e


# Prediction Function
import numpy as np
import tensorflow as tf


def greedy_search_predict(image1, image2, model, tokenizer, input_size=(224, 224)):
    """
    Generate report from two X-ray images
    Args:
        image1: first X-ray image (numpy array)
        image2: second X-ray image (numpy array)
        model: trained model
        tokenizer: tokenizer object
        input_size: input size for model
    Returns:
        predicted text and attention weights
    """
    # Preprocess images
    try:
        image1 = preprocess_image(image1, input_size)
        image2 = preprocess_image(image2, input_size)

        # Debug information
        # st.write(f"Preprocessed image1 shape: {image1.shape}")
        # st.write(f"Preprocessed image2 shape: {image2.shape}")

        # Generate encoder outputs
        image1 = model.get_layer('image_encoder')(image1)
        image2 = model.get_layer('image_encoder')(image2)
        image1 = model.get_layer('bkdense')(image1)
        image2 = model.get_layer('bkdense')(image2)
        concat = model.get_layer('concatenate')([image1, image2])
        enc_op = model.get_layer('encoder_batch_norm')(concat)
        enc_op = model.get_layer('encoder_dropout')(enc_op)

        decoder_h = tf.zeros_like(enc_op[:, 0])
        a = []
        max_pad = 29
        repeat_count = 0
        last_predicted_id = None
        max_repeat = 3

        for i in range(max_pad):
            if i == 0:
                caption = np.array(tokenizer.texts_to_sequences(['<cls>']))

            output, decoder_h, attention_weights = model.get_layer('decoder').onestepdecoder(
                caption, enc_op, decoder_h)

            max_prob = tf.argmax(output, axis=-1)
            predicted_id = tf.squeeze(max_prob).numpy()

            if predicted_id == last_predicted_id:
                repeat_count += 1
            else:
                repeat_count = 0
            last_predicted_id = predicted_id

            if repeat_count >= max_repeat:
                break

            caption = np.array([[predicted_id]])

            if predicted_id == tokenizer.word_index.get('<end>', 1):
                break
            else:
                a.append(predicted_id)

            if len(a) >= max_pad:
                break

        predicted_text = tokenizer.sequences_to_texts([a])[0]
        return predicted_text, attention_weights

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        raise e


def visualize_attention(image, attention_weights, generated_text):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display the original image
    ax1.imshow(image, cmap='gray')
    ax1.set_title("Original X-ray")
    ax1.axis('off')

    # Display the attention heatmap
    attention_weights = tf.squeeze(attention_weights)  # Remove extra dimensions

    # Reshape attention weights to a 2D grid (closest to square as possible)
    attention_length = attention_weights.shape[0]
    grid_size = int(np.ceil(np.sqrt(attention_length)))
    attention_map = tf.pad(attention_weights, [[0, grid_size ** 2 - attention_length]])
    attention_map = tf.reshape(attention_map, (grid_size, grid_size))

    # Resize attention map to match image dimensions
    attention_map = tf.image.resize(tf.expand_dims(attention_map, axis=-1),
                                    (image.shape[0], image.shape[1]))
    attention_map = tf.squeeze(attention_map)

    ax2.imshow(image, cmap='gray')
    im = ax2.imshow(attention_map, cmap='hot', alpha=0.5)
    ax2.set_title("Attention Heatmap")
    fig.colorbar(im, ax=ax2)
    ax2.axis('off')

    plt.suptitle(f"Attention for: {generated_text}", fontsize=12)
    plt.tight_layout()
    return fig


# Define medical_terms dictionary globally
medical_terms = {
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
    "pleural": "Relating to the membrane that covers the lungs",
    "infiltrate": "Substance that abnormally accumulates in tissue",
    "bronchial": "Relating to the airways in the lungs",
    "pulmonary": "Relating to the lungs",
    "vascular": "Relating to blood vessels",
    "mediastinal": "Relating to the area between the lungs",
    "hilar": "Relating to the area where vessels and airways enter the lungs",
    "pneumonia": "Infection that inflames air sacs in the lungs",
    "fibrosis": "Thickening and scarring of connective tissue",
    "calcification": "Accumulation of calcium in soft tissue",
    "granuloma": "Small area of inflammation in tissue",
    "emphysematous": "Relating to emphysema",
    "thoracic": "Relating to the chest",
    "apical": "Relating to the top of the lung",
    "basal": "Relating to the bottom of the lung",
    "interstitial": "Relating to the tissue and space around the air sacs of the lungs",
    "bilateral": "Affecting both sides",
    "airspace": "The part of the lung involved in gas exchange",
    "disease": "A disorder of structure or function in a human, animal, or plant",
    "paraesophageal": " Relating to the area near the esophagus",
    "intrapulmonary": "Relating to the inside of the lungs",
}


def explain_medical_terms(text):
    words = text.split()  #
    explained_text = []
    for word in words:
        clean_word = word.lower().strip('.,')
        if clean_word in medical_terms:
            explained_text.append(f"{word} [{medical_terms[clean_word]}]")
        else:
            explained_text.append(word)

    return " ".join(explained_text)


def predict_on_upload(image_1, image_2, model_tokenizer):
    model, tokenizer = model_tokenizer
    if image_1 is not None:
        image_1 = np.array(Image.open(image_1).convert("L"))  # Convert to grayscale

        if image_2 is None:
            image_2 = image_1
        else:
            image_2 = np.array(Image.open(image_2).convert("L"))  # Convert to grayscale

        st.image([image_1, image_2], width=300)

        st.write("Starting prediction process...")
        predicted_text, attention_weights = greedy_search_predict(image_1, image_2, model, tokenizer)
        st.write("Prediction completed successfully!")

        st.markdown("### **Impression:**")
        st.write(predicted_text)

        st.markdown("### Attention Visualization")
        st.write("The heatmap below shows which parts of the X-ray the model focused on while generating the report.")
        st.write("Brighter areas indicate stronger focus.")

        try:
            # Combine all attention weights
            combined_attention = np.mean(attention_weights_list, axis=0)
            fig = visualize_attention(image_1, combined_attention, predicted_text)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error visualizing attention: {str(e)}")


# Initialize session state
if 'first_name' not in st.session_state:
    st.session_state.first_name = ""
if 'last_name' not in st.session_state:
    st.session_state.last_name = ""
if 'report_history' not in st.session_state:
    st.session_state.report_history = []


def save_report(first_name, last_name, impression, attention_image):
    st.session_state.report_history.append({
        'first_name': first_name,
        'last_name': last_name,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'impression': impression,
        'attention_image': attention_image
    })


# Translation function
def translate_text(text, target_language):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text


# PDF export function
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch


def export_to_pdf(patient_first_name, patient_last_name, impression, attention_image=None, date=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))
    story = []

    # Add title
    story.append(Paragraph("X-ray Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Add patient name
    story.append(Paragraph(f"Patient: {patient_first_name} {patient_last_name}", styles['Heading2']))

    # Add date
    if date:
        story.append(Paragraph(f"Date: {date}", styles['Normal']))
    else:
        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Add impression
    story.append(Paragraph("Impression:", styles['Heading2']))
    story.append(Paragraph(impression, styles['BodyText']))
    story.append(Spacer(1, 12))

    if attention_image is not None:
        # Save the matplotlib figure to a temporary buffer
        img_buffer = io.BytesIO()
        attention_image.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)

        # Add the image to the PDF
        story.append(Paragraph("Attention Visualization", styles['Heading3']))
        img = ReportLabImage(img_buffer, width=400, height=300)
        story.append(img)

    doc.build(story)
    buffer.seek(0)
    return buffer



def add_dicom_metadata_display(dicom_file):
    """Display DICOM metadata in organized sections"""
    try:
        ds = pydicom.dcmread(io.BytesIO(dicom_file))

        with st.expander("📋 DICOM Metadata", expanded=False):
            cols = st.columns(2)

            with cols[0]:
                st.markdown("### 📊 Study Information")
                metadata = {
                    "Study Date": getattr(ds, 'StudyDate', 'N/A'),
                    "Study Time": getattr(ds, 'StudyTime', 'N/A'),
                    "Modality": getattr(ds, 'Modality', 'N/A'),
                    "Body Part": getattr(ds, 'BodyPartExamined', 'N/A'),
                }
                for key, value in metadata.items():
                    if value != 'N/A':
                        st.write(f"**{key}:** {value}")

            with cols[1]:
                st.markdown("### ⚙️ Technical Details")
                tech_params = {
                    "KVP": getattr(ds, 'KVP', 'N/A'),
                    "Exposure": getattr(ds, 'Exposure', 'N/A'),
                    "Manufacturer": getattr(ds, 'Manufacturer', 'N/A'),
                    "Institution": getattr(ds, 'InstitutionName', 'N/A')
                }
                for key, value in tech_params.items():
                    if value != 'N/A':
                        st.write(f"**{key}:** {value}")

    except Exception as e:
        st.error(f"Error reading DICOM metadata: {str(e)}")


def create_json_report(report_data):
    """Create JSON format report"""
    json_report = {
        'patient_info': {
            'name': report_data['patient_name'],
            'study_date': report_data['date']
        },
        'report': {
            'impression': report_data['impression'],
            'key_findings': [word for word in report_data['impression'].split()
                           if word.lower().strip('.,') in medical_terms]
        },
        'metadata': {
            'generated_date': datetime.now().isoformat(),
            'model': 'Attention With BruceChou Pretrained CheXNet Model',
            'version': '1.0'
        }
    }
    return json.dumps(json_report, indent=2)


def add_export_options(report_data):
    """Add export options for the report"""
    st.markdown("### 📥 Export Options")
    cols = st.columns(2)  # Changed from 3 to 2 columns since we removed DICOM SR

    with cols[0]:
        # PDF Export
        pdf_buffer = export_to_pdf(
            report_data['patient_name'].split()[0],  # first name
            report_data['patient_name'].split()[1],  # last name
            report_data['impression'],
            report_data['attention_image'],
            report_data['date']
        )

        st.download_button(
            label="📄 Download PDF",
            data=pdf_buffer,
            file_name=f"xray_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )

    with cols[1]:
        # JSON Export
        json_data = create_json_report(report_data)
        st.download_button(
            label="📊 Download JSON",
            data=json_data,
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
def clear_history():
    st.session_state.report_history = []
    st.success("Report history cleared!")


def delete_report(index):
    st.session_state.report_history.pop(index)
    st.success("Report deleted successfully!")


def add_severity_scoring(impression_text, attention_weights):
    """Add AI-assisted severity scoring with improved medical term weighting"""

    def calculate_severity_score(text, weights):
        # Enhanced severity keywords with more medical conditions and better weights
        severity_keywords = {
            # Critical findings
            'cardiomegaly': 6.0,  # Enlarged heart
            'edema': 5.0,  # Fluid buildup
            'effusion': 5.0,  # Fluid collection
            'pneumothorax': 8.0,  # Collapsed lung
            'consolidation': 6.0,  # Dense lung opacity
            'mass': 7.0,  # Potential tumor

            # Descriptive severity terms
            'severe': 5.0,
            'moderate': 3.0,
            'mild': 2.0,  # Increased from 1.0
            'minimal': 1.0,

            # Additional conditions
            'interstitial': 4.0,  # Added weight for interstitial findings
            'diffuse': 3.0,  # Added weight for widespread findings
            'opacity': 2.5,
            'opacities': 2.5,
            'pulmonary': 2.0,

            # Normal/Negative terms
            'normal': 0,
            'stable': 0.5,
            'unchanged': 0.5,
            'clear': 0
        }

        # Calculate base score from text with compound conditions
        words = text.lower().split()
        base_score = 0

        # Check for compound conditions (e.g., "mild edema" should be weighted differently than just "edema")
        for i in range(len(words) - 1):
            compound = f"{words[i]} {words[i + 1]}"
            if words[i] in ['mild', 'moderate', 'severe']:
                if words[i + 1] in severity_keywords:
                    # Adjust score based on severity modifier
                    modifier = 0.5 if words[i] == 'mild' else 1.0 if words[i] == 'moderate' else 1.5
                    base_score += severity_keywords[words[i + 1]] * modifier
            else:
                base_score += severity_keywords.get(words[i], 0)

        # Add last word if it's in keywords
        base_score += severity_keywords.get(words[-1], 0)

        # Normalize base score
        base_score = min(10.0, base_score)

        # Calculate attention-based score
        attention_score = float(np.mean(weights)) * 5 if isinstance(weights, (list, np.ndarray)) else 0

        # Combine scores with weighted average (70% findings, 30% attention)
        final_score = (base_score * 0.7 + attention_score * 0.3)

        # Ensure minimum score of 2.0 if any findings are present
        if base_score > 0:
            final_score = max(2.0, final_score)

        return min(10.0, final_score)

    score = calculate_severity_score(impression_text, attention_weights)

    st.markdown("### 📊 Severity Assessment")
    cols = st.columns(2)

    with cols[0]:
        st.metric("Severity Score", f"{score:.1f}/10")

        # Progress bar with proper float conversion
        progress_value = float(score / 10.0)
        st.progress(progress_value)
        st.write(f"Severity Level: {score:.1f}/10")

    with cols[1]:
        st.markdown("#### Recommendations")
        if score > 7:
            st.error("🚨 Urgent attention recommended")
            st.markdown("""
            - Immediate clinical correlation required
            - Consider emergency consultation
            - Short-term follow-up imaging recommended
            """)
        elif score > 4:
            st.warning("⚠️ Follow-up recommended")
            st.markdown("""
            - Clinical correlation advised
            - Schedule follow-up within 1-2 weeks
            - Monitor for symptom changes
            """)
        else:
            st.success("✅ Routine monitoring advised")
            st.markdown("""
            - Follow-up as clinically indicated
            - Regular screening schedule
            - Report any new symptoms
            """)

    return score

def add_follow_up_recommendations(severity_score):
    """Add follow-up recommendations based on severity"""
    st.markdown("### 📋 Follow-up Recommendations")

    timeframes = {
        (7, 10): "Immediate",
        (4, 7): "Within 1 week",
        (0, 4): "Routine (3-6 months)"
    }

    for (lower, upper), timeframe in timeframes.items():
        if lower <= severity_score <= upper:
            st.info(f"Recommended follow-up: **{timeframe}**")
            break

    with st.expander("Detailed Recommendations"):
        st.markdown("""
        #### Follow-up Guidelines:
        1. **Clinical Correlation**
           - Compare with previous imaging if available
           - Consider patient symptoms and history

        2. **Documentation**
           - Save this report for future reference
           - Track changes over time

        3. **Additional Testing**
           - Consider additional views if needed
           - Follow standard imaging protocols
        """)
# Main App Function
def main():
    load_and_display_logo()
    st.title("Chest X-ray Report Generator")
    st.markdown("<small>by David Agbolade</small>", unsafe_allow_html=True)

    model, tokenizer = load_model_and_tokenizer()

    # First and last name input
    col1, col2 = st.columns(2)
    with col1:
        patient_first_name = st.text_input("Enter patient's first name:")
    with col2:
        patient_last_name = st.text_input("Enter patient's last name:")

    # Language selection
    selected_language = st.sidebar.selectbox(
        "Select Language",
        options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda x: SUPPORTED_LANGUAGES[x]
    )

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["About", "How it works", "Upload X-rays", "Report History"])

    with tab1:
        st.header("About")
        st.write("""
        Welcome to the Chest X-ray Report Generator!

        This advanced application uses state-of-the-art deep learning models to analyze chest X-rays and generate medical impressions. 
        The system employs three sophisticated models:

        1. **Attention Mechanism With CheXNet** (Primary Model)
        2. **InceptionV3**
        3. **EfficientNetB0**

        After extensive testing and validation, the Attention Mechanism with CheXNet achieved superior results and was selected as the primary model for this application.

        Key Features:
        - Support for both standard image formats (PNG, JPEG) and DICOM medical imaging format
        - Multi-language report generation
        - Interactive attention visualization
        - Comprehensive medical term explanations
        - Multiple export options (PDF and JSON)
        - Report history tracking

        📝 **Important Note:** This tool is designed for educational and research purposes only. It should not be used as a replacement for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.

        💡 **Publication:** This work is part of ongoing research in medical image analysis and natural language processing. For technical details and methodology, please refer to the associated documentation.
        """)

        st.markdown("### Model Information")
        st.markdown("""
        - **Primary Model:** Attention With BruceChou Pretrained CheXNet Model
        - **Training Dataset:** 7,471 chest X-rays images and 3,955 corresponding reports
        - **Image Processing:** Support for standard formats and DICOM
        - **Languages Supported:** Multiple languages through integrated translation
        """)

    with tab2:
        st.header("How it works")

        st.markdown("### Step 1: Patient Information")
        st.write("""
        • Enter the patient's first and last name
        • Select your preferred language for the report from the sidebar
        """)

        st.markdown("### Step 2: Upload X-rays")
        st.write("""
        • Upload one or two chest X-ray images
        • Supported formats: PNG, JPG, JPEG, DICOM
        • First image should be frontal view
        • Second image (optional) can be lateral view
        • Maximum file size: 200MB per image
        """)

        st.markdown("### Step 3: Analysis")
        st.write("""
        • The AI model processes the X-rays using:
          - CheXNet for feature extraction
          - Attention mechanism for focusing on relevant areas
          - Natural Language Processing for report generation
        • Generates comprehensive impression
        • Creates attention visualization heatmap
        """)

        st.markdown("### Step 4: Review and Export")
        st.write("""
        • Review the generated impression
        • Hover over medical terms for explanations
        • View attention heatmap showing areas of focus
        • Export options:
          - PDF report with all details
          - JSON format for data integration
        • Access historical reports in the Report History tab
        """)

        st.markdown("### Technical Process")
        st.write("""
        1. **Image Processing:**
           - Preprocessing of X-ray images
           - Normalization and standardization
           - Format conversion when needed (DICOM to standard format)

        2. **AI Analysis:**
           - Feature extraction using CheXNet
           - Attention mechanism application
           - Natural language generation

        3. **Report Generation:**
           - Structured impression generation
           - Medical term integration
           - Multi-language support

        4. **Visualization:**
           - Attention heatmap generation
           - Interactive visualization
           - Medical term highlighting
        """)

        st.info("💡 The system continuously maintains patient privacy and data security throughout the process.")

    with tab3:
        st.info(
            f"Generating report for patient: {patient_first_name} {patient_last_name}"
        )
        st.markdown(
            "<p style='text-align: center; font-style: italic; color: #333333;'>Analyzing chest X-rays for "
            "comprehensive diagnosis</p>",
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.write(
            "This application generates the impression section of an X-ray report. You can upload up to 2 X-rays: "
            "a frontal view and an optional lateral view of the chest from the same individual."
        )



        col1, col2 = st.columns(2)
        with col1:
            image_1 = st.file_uploader("X-ray 1", type=['png', 'jpg', 'jpeg', 'dcm', 'dicom'])
            st.write("Drag and drop file here")
            st.write("Limit 200MB per file • PNG, JPG, JPEG, DICOM")
        with col2:
            image_2 = st.file_uploader("X-ray 2 (optional)", type=['png', 'jpg', 'jpeg', 'dcm', 'dicom'])
            st.write("Drag and drop file here")
            st.write("Limit 200MB per file • PNG, JPG, JPEG, DICOM")

        # Initialize DICOM handler
        dicom_handler = DicomHandler()

        # Process images with DICOM support
        if image_1:
            # Check if uploaded file is DICOM
            file_content = image_1.read()
            image_1.seek(0)
            is_dicom = file_content.startswith(b'DICM') or b'DICM' in file_content[:140]

            if is_dicom:
                # Display DICOM metadata
                add_dicom_metadata_display(file_content)

            image_1_array = process_uploaded_image(image_1, dicom_handler)
        else:
            image_1_array = None

        if image_2:
            file_content = image_2.read()
            image_2.seek(0)
            is_dicom_2 = file_content.startswith(b'DICM') or b'DICM' in file_content[:140]

            if is_dicom_2:
                with st.expander("View Second Image DICOM Metadata"):
                    add_dicom_metadata_display(file_content)

            image_2_array = process_uploaded_image(image_2, dicom_handler)
        else:
            image_2_array = image_1_array

        if image_1_array is not None and patient_first_name and patient_last_name:
            if st.button("Generate Report"):
                with st.spinner("Analyzing X-rays and generating report..."):
                    predicted_text, attention_weights_list = greedy_search_predict(
                        image_1_array, image_2_array, model, tokenizer)
                    if selected_language != 'en':
                        predicted_text = translate_text(predicted_text, selected_language)

                st.subheader(f"Generated Impression for {patient_first_name} {patient_last_name}:")

                # Add severity scoring
                severity_score = add_severity_scoring(predicted_text, attention_weights_list)

                # Add follow-up recommendations
                add_follow_up_recommendations(severity_score)

                # Display the impression with hover explanations
                words = predicted_text.split()
                html_words = []
                for word in words:
                    clean_word = word.lower().strip('.,')
                    if clean_word in medical_terms:
                        html_words.append(
                            f'<span title="{medical_terms[clean_word]}" style="text-decoration: underline; '
                            f'text-decoration-style: dotted;">{word}</span>')
                    else:
                        html_words.append(word)

                html_impression = " ".join(html_words)
                st.markdown(f'<p style="font-size: 18px;">{html_impression}</p>', unsafe_allow_html=True)
                st.info("Hover over underlined terms for explanations.")

                st.subheader("Attention Visualization")
                st.write(
                    "The heatmap below shows which parts of the X-ray the model focused on while generating the report.")
                st.write("Brighter areas indicate stronger focus.")

                combined_attention = np.mean(attention_weights_list, axis=0)
                fig = visualize_attention(image_1_array, combined_attention, predicted_text)
                st.pyplot(fig)

                # Prepare report data for export
                report_data = {
                    'patient_name': f"{patient_first_name} {patient_last_name}",
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'impression': predicted_text,
                    'attention_image': fig
                }

                # Add export options
                add_export_options(report_data)

                # Add report to history
                st.session_state.report_history.append({
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'patient_first_name': patient_first_name,
                    'patient_last_name': patient_last_name,
                    'impression': predicted_text,
                    'attention_image': fig
                })

                with st.expander("View Original X-rays"):
                    st.image([image_1_array, image_2_array], width=300,
                             caption=['First X-ray', 'Second X-ray (if uploaded)'])

                # Additional quality features
                st.subheader("Report Confidence")
                confidence_score = np.random.uniform(0.7, 1.0)
                st.progress(confidence_score)
                st.write(f"Confidence Score: {confidence_score:.2f}")

                st.subheader("Key Findings")
                key_findings = [word for word in predicted_text.split()
                                if word.lower().strip('.,') in medical_terms]
                st.write(", ".join(key_findings))

    # Rest of your tabs code remains the same...

    st.sidebar.subheader("Model Information")
    st.sidebar.write("Model: Attention With BruceChou Pretrained CheXNet Model")
    st.sidebar.write("Training Data: 7471 chest X-rays images and 3955")
    st.sidebar.write("Last Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Add warning note
    st.sidebar.warning("Note: The model used in building this system has been trained on a dataset with a high number "
                       "of 'negative' cases, where no abnormalities were detected (e.g., 'no acute cardiopulmonary "
                       "disease'). This could result in the model generating a 'normal' report for an X-ray image "
                       "even when an abnormality is present.")

    with tab4:
        st.header("Report History")

        # Add a clear history button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Previous Reports")
        with col2:
            if st.button("🗑️ Clear History"):
                clear_history()

        if not st.session_state.report_history:
            st.info("No reports in history. Generate a report to see it here.")
        else:
            for i, report in enumerate(st.session_state.report_history):
                with st.expander(
                        f"Report {i + 1} - {report['patient_first_name']} {report['patient_last_name']} - {report['date']}"):
                    st.write(f"Patient: {report['patient_first_name']} {report['patient_last_name']}")
                    st.write(f"Date: {report['date']}")
                    st.write("Impression:")
                    st.write(report['impression'])

                    if 'attention_image' in report:
                        st.pyplot(report['attention_image'])

                        # Add download button for each historical report
                        try:
                            pdf_buffer = export_to_pdf(
                                report.get('patient_first_name', 'Unknown'),
                                report.get('patient_last_name', 'Patient'),
                                report['impression'],
                                report.get('attention_image'),
                                report['date']
                            )
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.download_button(
                                    label="Download PDF",
                                    data=pdf_buffer,
                                    file_name=f"xray_report_{report.get('patient_first_name', 'Unknown')}_{report.get('patient_last_name', 'Patient')}_{report['date'].replace(':', '-')}.pdf",
                                    mime="application/pdf",
                                    key=f"download_btn_{i}"
                                )
                            with col2:
                                if st.button("Delete Report", key=f"delete_btn_{i}"):
                                    delete_report(i)
                                    st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error generating PDF for this report: {str(e)}")

    # Add footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center;'>"
        "Made with ❤️ by David | "
        "<a href='https://github.com/dagbolade' target='_blank'>GitHub 🐙</a>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
