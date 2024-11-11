import numpy as np
from tf_keras.src.callbacks import EarlyStopping, ModelCheckpoint
from model_development import create_model, create_cnn_model, create_rnn_model, split_data, train_model, evaluate_model
from preprocessing import preprocess_reports, preprocess_images
from data_exploration import explore_data
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def prepare_labels(reports, tokenizer, max_length, num_classes):
    findings_labels = []
    impression_labels = []
    for report in reports:
        if "IMPRESSION" in report:
            findings, impression = report.split("IMPRESSION")[0], "IMPRESSION" + report.split("IMPRESSION")[1]
        else:
            findings, impression = report, ""
        findings_encoding = tokenizer.texts_to_sequences([findings])[0]
        impression_encoding = tokenizer.texts_to_sequences([impression])[0]
        findings_label = pad_sequences([findings_encoding], maxlen=max_length)[0]
        impression_label = pad_sequences([impression_encoding], maxlen=max_length)[0]
        findings_labels.append(to_categorical(findings_label, num_classes=num_classes))
        impression_labels.append(to_categorical(impression_label, num_classes=num_classes))
    return np.array(findings_labels), np.array(impression_labels)


def main():
    # Set the directories for reports and images
    report_dir = "report"
    image_dir = "images"

    # Preprocess reports
    preprocessed_reports, image_report_map, report_filenames = preprocess_reports(report_dir)

    # Preprocess images
    preprocessed_images, image_filenames = preprocess_images(image_dir)

    # #Print the preprocessed data
    # print("Preprocessed Reports:")
    # print(preprocessed_reports)
    # print("\nImage-Report Map:")
    # print(image_report_map)
    # print("\nPreprocessed Images:")
    # print(preprocessed_images.shape)
    #
    # # Count of the mapped images - reports
    # print("Number of Mapped Images - Reports:", len(image_report_map))
    #
    # # count of the reports
    # print("Number of Reports:", len(preprocessed_reports))
    #
    # # # Perform data exploration
    # explore_data(preprocessed_images, preprocessed_reports, image_report_map)

    # Set the maximum sequence length and vocabulary size for the transformer model
    max_length = 128
    vocab_size = 5000

    # Create the image-report mapping
    image_report_map = {image_filename.split(".")[0]: report_filename.split(".")[0]
                        for image_filename, report_filename in zip(image_filenames, report_filenames)}
    # Filter reports to ensure each report has a corresponding image
    valid_image_filenames = [fname for fname in image_filenames if fname.split(".")[0] in image_report_map]
    valid_preprocessed_images = [preprocessed_images[i] for i, fname in enumerate(image_filenames) if
                                 fname.split(".")[0] in image_report_map]
    valid_preprocessed_reports = [preprocessed_reports[i] for i, fname in enumerate(image_filenames) if
                                  fname.split(".")[0] in image_report_map]

    # Split the data
    train_images, test_images, train_reports, test_reports = split_data(
        valid_preprocessed_images, valid_preprocessed_reports)

    # Convert train and test reports to list of strings
    train_reports = [' '.join(findings + impression) for findings, impression in train_reports]
    test_reports = [' '.join(findings + impression) for findings, impression in test_reports]

    # Tokenize text data
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_reports)
    train_report_ids = pad_sequences(tokenizer.texts_to_sequences(train_reports), maxlen=max_length)
    test_report_ids = pad_sequences(tokenizer.texts_to_sequences(test_reports), maxlen=max_length)

    train_findings_labels, train_impression_labels = prepare_labels(train_reports, tokenizer, max_length, num_classes=len(tokenizer.word_index) + 1)

    # Create models
    cnn_model = create_cnn_model(input_shape=(224, 224, 3))
    rnn_model = create_rnn_model(max_length=max_length, vocab_size=len(tokenizer.word_index) + 1)
    model = create_model(cnn_model, rnn_model, vocab_size=len(tokenizer.word_index) + 1, max_length=max_length)

    # Train the model
    train_model(model, train_images, train_report_ids, epochs=1, batch_size=4,
                train_findings_labels=train_findings_labels, train_impression_labels=train_impression_labels)

    test_findings_labels, test_impression_labels = prepare_labels(test_reports, tokenizer, max_length, num_classes=len(tokenizer.word_index) + 1)

    # Evaluate the model
    evaluate_model(model, test_images, test_report_ids, test_findings_labels, test_impression_labels)

    # Save the model in h5 format
    model.save('final_model.h5')

    #streamlit


if __name__ == "__main__":
    main()
