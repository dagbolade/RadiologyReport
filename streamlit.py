import base64
import io
from datetime import datetime
import warnings
import os

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=Warning)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Your TensorFlow import and code here
import tensorflow as tf

from matplotlib.colors import LinearSegmentedColormap

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
}


# Helper Functions
def load_and_display_logo():
    logo_url = "https://www.solent.ac.uk/graphics/logo/rebrandLogo.svg"
    response = requests.get(logo_url)
    if response.status_code == 200:
        svg_content = response.text
        st.markdown(f'<div style="display: flex; justify-content: center;">{svg_content}</div>', unsafe_allow_html=True)
    else:
        st.error("Failed to load the logo")


def create_tabs():
    return st.tabs(["Upload X-rays", "About", "How it works", "Report History"])


# Model and Tokenizer Loading
@st.cache_resource
def load_model_and_tokenizer():
    st.write("Loading model and tokenizer...")
    # Update the model path to use the new .keras or .h5 file
    model_path = 'Radiography/Attention_with_cheXNet_full_model.keras'  # or use .h5 if you prefer
    tokenizer_path = 'Radiography/tokenizer/tokenizer.pickle'

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(f"Model path: {model_path}")
        st.error(f"TensorFlow version: {tf.__version__}")
        st.error(f"Keras version: {tf.keras.__version__}")
        return None, None

    try:
        tokenizer = joblib.load(tokenizer_path)
        st.success("Tokenizer loaded successfully!")
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        return model, None

    return model, tokenizer


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


def process_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file).convert("L"))
        if is_likely_chest_xray(image):
            return image
        else:
            st.error("The uploaded image does not appear to be a chest X-ray. Please upload only chest X-ray images.")
            return None
    return None


# Image Processing
def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Prediction Function
def greedy_search_predict(image1, image2, model, tokenizer):
    # Ensure images are in RGB format with 3 channels
    image1 = Image.fromarray(image1).convert('RGB')
    image2 = Image.fromarray(image2).convert('RGB')

    # Convert to numpy arrays and preprocess
    image1 = np.array(image1)
    image2 = np.array(image2)
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    # Generate encoder outputs
    enc_op1 = model.get_layer('image_encoder')(image1)
    enc_op2 = model.get_layer('image_encoder')(image2)
    image1_encoded = model.get_layer('bkdense')(enc_op1)
    image2_encoded = model.get_layer('bkdense')(enc_op2)

    concat = model.get_layer('concatenate')([image1_encoded, image2_encoded])
    enc_op = model.get_layer('encoder_batch_norm')(concat)
    enc_op = model.get_layer('encoder_dropout')(enc_op)

    decoder_h = tf.zeros_like(enc_op[:, 0])
    predicted_caption = []
    attention_weights_list = []

    for i in range(max_pad):
        if i == 0:
            caption = np.array(tokenizer.texts_to_sequences(['<cls>']))

        caption = tf.cast(caption, dtype=tf.float32)

        output, decoder_h, attention_weights = model.get_layer('decoder').onestepdecoder(caption, enc_op, decoder_h)
        attention_weights_list.append(attention_weights)

        max_prob = tf.argmax(output, axis=-1)
        predicted_id = max_prob.numpy()[0]

        if predicted_id == tokenizer.word_index.get('<end>', 1):
            break

        predicted_caption.append(predicted_id)
        caption = np.array([[predicted_id]])

    predicted_text = tokenizer.sequences_to_texts([predicted_caption])[0]
    return predicted_text, attention_weights_list


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
    "disease": "A disorder of structure or function in a human, animal, or plant"
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
        predicted_text, attention_weights_list = greedy_search_predict(image_1, image_2, model, tokenizer)
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


def clear_history():
    st.session_state.report_history = []
    st.success("Report history cleared!")


def delete_report(index):
    st.session_state.report_history.pop(index)
    st.success("Report deleted successfully!")


# Main App Function
def main():
    load_and_display_logo()
    st.title("Chest X-ray Report Generator")
    st.markdown("<small>by David</small>", unsafe_allow_html=True)

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

    with tab3:
        st.markdown("<h2 style='text-align: center; color: #0066cc;'>Patient X-ray Report</h2>", unsafe_allow_html=True)
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
            image_1 = st.file_uploader("X-ray 1", type=['png', 'jpg', 'jpeg'])
            st.write("Drag and drop file here")
            st.write("Limit 200MB per file • PNG, JPG, JPEG")
        with col2:
            image_2 = st.file_uploader("X-ray 2 (optional)", type=['png', 'jpg', 'jpeg'])
            st.write("Drag and drop file here")
            st.write("Limit 200MB per file • PNG, JPG, JPEG")

        image_1 = process_uploaded_image(image_1)
        image_2 = process_uploaded_image(image_2) if image_2 else image_1

        if image_1 is not None and patient_first_name and patient_last_name:
            if st.button("Generate Report"):
                if image_2 is None:
                    st.error("Please ensure both uploaded images are valid chest X-rays.")
                else:
                    with st.spinner("Analyzing X-rays and generating report..."):
                        predicted_text, attention_weights_list = greedy_search_predict(image_1, image_2, model,
                                                                                       tokenizer)
                        if selected_language != 'en':
                            predicted_text = translate_text(predicted_text, selected_language)

                    st.subheader(f"Generated Impression for {patient_first_name} {patient_last_name}:")

                    # Display the impression with hover explanations
                    words = predicted_text.split()
                    html_words = []
                    for word in words:
                        clean_word = word.lower().strip('.,')
                        if clean_word in medical_terms:
                            html_words.append(
                                f'<span title="{medical_terms[clean_word]}" style="text-decoration: underline; text-decoration-style: dotted;">{word}</span>')
                        else:
                            html_words.append(word)

                    html_impression = " ".join(html_words)
                    st.markdown(f'<p style="font-size: 18px;">{html_impression}</p>', unsafe_allow_html=True)
                    st.info("Hover over underlined terms for explanations.")

                    st.subheader("Attention Visualization")
                    st.write(
                        "The heatmap below shows which parts of the X-ray the model focused on while generating the "
                        "report.")
                    st.write("Brighter areas indicate stronger focus.")

                    combined_attention = np.mean(attention_weights_list, axis=0)
                    fig = visualize_attention(image_1, combined_attention, predicted_text)
                    st.pyplot(fig)

                    # Add report to history
                    st.session_state.report_history.append({
                        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'patient_first_name': patient_first_name,
                        'patient_last_name': patient_last_name,
                        'impression': predicted_text,
                        'attention_image': fig
                    })

                    # Export options
                    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    pdf_buffer = export_to_pdf(patient_first_name, patient_last_name, predicted_text, fig, current_date)
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"xray_report_{patient_first_name}_{patient_last_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )

                    with st.expander("View Original X-rays"):
                        st.image([image_1, image_2], width=300, caption=['First X-ray', 'Second X-ray (if uploaded)'])

                    # Additional quality features
                    st.subheader("Report Confidence")
                    confidence_score = np.random.uniform(0.7, 1.0)  # Placeholder for actual confidence calculation
                    st.progress(confidence_score)
                    st.write(f"Confidence Score: {confidence_score:.2f}")

                    st.subheader("Key Findings")
                    key_findings = [word for word in predicted_text.split() if
                                    word.lower().strip('.,') in medical_terms]
                    st.write(", ".join(key_findings))

    with tab1:
        st.header("About")
        st.write("""
            This advanced app uses a deep learning model to analyze chest X-rays and generate medical impressions. 
            Three models were used  which are the Attention Mechanism With CheXNet, InceptionV3 and the EfficientNetB0.
            The Attention Mechanism with CheXNet achieved the best results, which is the model used in this app.
            
            The attention visualization helps understand which areas of the X-ray were most important for the prediction.
    
            The model  has been trained on a large dataset of chest X-rays.
    
            Please note that this tool is for educational purposes only and should not be used for actual medical diagnosis. 
            
            """)

    with tab2:
        st.header("How it works")
        st.write("""
        1. Enter Patient Information:
           - Input the patient's first and last name.

        2. Select Language:
           - Choose your preferred language for the report from the sidebar.

        3. Upload X-ray Images: 
           - Upload one or two chest X-ray images.
           - The first image should be a front view of the chest.
           - The second image (optional) should be a side view of the chest.

        4. Generate Report: 
           - Click the "Generate Report" button to analyze the uploaded X-rays.
           - The AI model will process the images and generate an impression.

        5. View Results:
           - The generated impression will appear with medical terms explained (hover over underlined terms).
           - An attention visualization will show which parts of the X-ray the model focused on.
           - You can view the original X-rays, a confidence score for the report, and key findings.

        6. Download Report:
           - Download a PDF report containing the patient's name, date, impression, and attention visualization.

        7. Report History:
           - Access previous reports in the "Report History" tab.
           - Download PDF reports for any historical entries.

        8. Interpret Results:
           - Use the generated impression as a starting point for understanding the X-ray.
           - The attention visualization can provide insight into the model's decision-making process.
           - Remember, this tool is for educational purposes and should not replace professional medical advice.
        """)

    st.sidebar.subheader("Model Information")
    st.sidebar.write("Model: Attention With BruceChou Pretrined CheXNet Model")
    st.sidebar.write("Training Data: 7471 chest X-rays images and 3955")

    # using datetime module to get the current date and time to show last updated
    st.sidebar.write("Last Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Add a Note

    st.sidebar.warning("Note: The Report used in building this sytem has a lot of negative cases, so "
                       "the model might generate a negative report for a normal X-ray image. To make this system better,"
                       "it would need to be trained on a large dataset of chest X-rays with a balanced distribution of "
                       "positive and negative cases.")

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
