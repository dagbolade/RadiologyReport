import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


def plot_image_distribution(preprocessed_images):
    # Plot the distribution of pixel values in the preprocessed images
    pixel_values = preprocessed_images.flatten()
    plt.hist(pixel_values, bins=50, density=True)
    plt.xlabel("Pixel Values")
    plt.ylabel("Frequency")
    plt.title("Distribution of Pixel Values")
    plt.show()


def plot_report_lengths(preprocessed_reports):
    # Plot the distribution of report lengths
    report_lengths = [len(report) for report in preprocessed_reports]
    plt.hist(report_lengths, bins=20)
    plt.xlabel("Report Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Report Lengths")
    plt.show()


def analyze_report_vocabulary(preprocessed_reports):
    # Analyze the vocabulary used in the reports
    all_tokens = [token for report in preprocessed_reports for token in report]
    token_counts = Counter(all_tokens)
    print("Total Vocabulary Size:", len(token_counts))
    print("Top 10 Most Common Tokens:", token_counts.most_common(10))


# image qulaity anlysis
def calculate_image_sharpness(image):
    image = np.array(image * 255, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian


def check_image_quality(preprocessed_images):
    sharpness_scores = [calculate_image_sharpness(image) for image in preprocessed_images]
    plt.hist(sharpness_scores, bins=50)
    plt.xlabel('Sharpness Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Image Sharpness Scores')
    plt.show()

    print("Mean Sharpness:", np.mean(sharpness_scores))
    print("Median Sharpness:", np.median(sharpness_scores))
    print("Standard Deviation of Sharpness:", np.std(sharpness_scores))


# class distribution
def plot_class_distribution(image_report_map):
    report_ids = list(image_report_map.values())
    report_counts = Counter(report_ids)
    plt.hist(report_counts.values(), bins=20)
    plt.xlabel("Number of Images")
    plt.ylabel("Frequency")
    plt.title("Distribution of Number of Images per Report")
    plt.show()

    print("Number of Unique Reports:", len(report_counts))
    print("Top 10 Reports with Most Images:")
    print(report_counts.most_common(10))

# plot the top 10 reports with most images and reports with least images

#report simillaritiws  analysis
def plot_report_similarity(preprocessed_reports):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_reports)
    cosine_sim = cosine_similarity(tfidf_matrix)

    plt.figure(figsize=(10, 10))
    sns.heatmap(cosine_sim, cmap='coolwarm')
    plt.title('Report Similarity Matrix')
    plt.show()

    similar_reports = []
    for i in range(len(cosine_sim)):
        for j in range(i + 1, len(cosine_sim)):
            if cosine_sim[i][j] > 0.9:
                similar_reports.append((i, j))
    print("Number of Similar Reports:", len(similar_reports))
    print("Similar Report Pairs:")
    print(similar_reports)


def explore_data(preprocessed_images, preprocessed_reports, image_report_map):
    print("Data Exploration")
    print("-----------------")

    print("Number of Images:", len(preprocessed_images))
    print("Image Shape:", preprocessed_images[0].shape)
    plot_image_distribution(preprocessed_images)

    print("\nNumber of Reports:", len(preprocessed_reports))
    plot_report_lengths(preprocessed_reports)

    print("\nVocabulary Analysis:")
    analyze_report_vocabulary(preprocessed_reports)

    print("\nImage Quality Analysis:")
    check_image_quality(preprocessed_images)

    print("\nClass Distribution:")
    plot_class_distribution(image_report_map)

    print("\nReport Similarity Analysis:")
    plot_report_similarity(preprocessed_reports)


