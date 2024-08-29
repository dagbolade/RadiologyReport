import os
import xml.etree.ElementTree as ET
import nltk
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from PIL import Image
import numpy as np

from data_exploration import calculate_image_sharpness


# Function to preprocess reports
def preprocess_reports(report_dir):
    preprocessed_reports = []
    image_report_map = {}
    report_filenames = []
    for filename in os.listdir(report_dir):
        if filename.endswith(".xml"):
            report_path = os.path.join(report_dir, filename)
            tree = ET.parse(report_path)
            root = tree.getroot()

            findings = ""
            impression = ""
            for abstract_text in root.findall(".//AbstractText"):
                if abstract_text.get("Label") == "FINDINGS":
                    findings = abstract_text.text if abstract_text.text else ""
                elif abstract_text.get("Label") == "IMPRESSION":
                    impression = abstract_text.text if abstract_text.text else ""

            findings_tokens = word_tokenize(findings) if findings else []
            impression_tokens = word_tokenize(impression) if impression else []
            preprocessed_reports.append((findings_tokens, impression_tokens))
            report_filenames.append(filename)

            # Extract image IDs from the XML
            image_ids = []
            for parent_image in root.findall(".//parentImage"):
                image_id = parent_image.get("id")
                image_ids.append(image_id)

            # Map report filename to image filenames
            report_id = filename.split(".")[0]
            for image_id in image_ids:
                image_report_map[image_id] = report_id

    return preprocessed_reports, image_report_map, report_filenames

# Function to preprocess images
def preprocess_images(image_dir, target_size=(224, 224)):
    preprocessed_images = []
    image_filenames = []

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)
            image = image.resize(target_size)
            image = np.array(image) / 255.0
            preprocessed_images.append(image)
            image_filenames.append(filename)

    preprocessed_images = np.array(preprocessed_images)
    return preprocessed_images, image_filenames

# check for the description of the reports
