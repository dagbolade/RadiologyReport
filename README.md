# Automated Radiology Report Generation using Deep Learning for Chest X-ray scans

(Demo.gif)![demo](https://github.com/user-attachments/assets/fd26a019-16c2-49ed-a447-6bb8b7c4754d)








## Project Overview

This project develops an automated system for generating radiology reports from chest X-ray images, focusing on the impression section. The system integrates Convolutional Neural Networks (CNNs) for image feature extraction with a global attention mechanism and a Gated Recurrent Unit (GRU) to generate coherent and clinically relevant reports.

The model was trained on a publicly available dataset from Indiana University, consisting of 7,470 chest X-ray images and their corresponding reports.

## Key Features

- Utilizes advanced CNN architectures (CheXNet, InceptionV3, EfficientNet) for feature extraction
- Implements a global attention mechanism for focusing on relevant image areas
- Generates coherent and clinically relevant radiology reports
- Provides visualization tools including attention maps and saliency maps
- Offers a user-friendly interface via a Streamlit web application

## Model Performance

The CheXNet Attention model with greedy search decoding achieved the following BLEU scores:

- BLEU-1: 0.279464
- BLEU-2: 0.176753
- BLEU-3: 0.115272
- BLEU-4: 0.062379

## Main Files

1. `app.py`: The main Streamlit application file for the user interface.
2. `1_Data_Preparation_EDA.ipynb`: Jupyter notebook for data preparation and exploratory data analysis.
3. `2_CheXNet_model.ipynb`: Implementation of the CheXNet model without attention.
4. `3_Inceptionv3.ipynb`: Implementation of the InceptionV3 model.
5. `4_EfficientNet_Model.ipynb`: Implementation of the EfficientNet model.
6. `5_Attention_Model_With_CheXNet.ipynb`: The final model implementing CheXNet with attention mechanism.

## Getting Started

1. Clone the repository:

```bash 
git clone https://github.com/dagbolade/RadiologyReport.git
cd RadiologyReport
```
2. Install dependencies:

```bash 
pip install -r requirements.txt
``` 
3. Run the Streamlit application:

```bash
streamlit run app.py
```
## Usage

1. Launch the Streamlit app using the command above.
2. Upload chest X-ray images using the file uploader in the app.
3. The system will generate an impression section of the radiology report.
4. View the generated report along with attention visualizations.
 5. Download the report as a pdf file.
