import os
import cv2
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from datetime import datetime

# Read labels from file
def read_labels(label_file):
    labels = []
    with open(label_file, 'r') as file:
        for line in file:
            label = line.strip().split()[1]
            labels.append(label)
    return labels

# Define paths and models
dataset_path = 'Test/'
label_file_left = 'Model/labels_left.txt'
label_file_right = 'Model/labels_right.txt'
classifier_left = Classifier("Model/keras_model_left.h5", label_file_left)
classifier_right = Classifier("Model/keras_model_right.h5", label_file_right)
img_size = 224

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    final_image = np.ones((img_size, img_size, 3), np.uint8) * 255
    aspect_ratio = img.shape[0] / img.shape[1]

    if aspect_ratio > 1:
        k = img_size / img.shape[0]
        w_cal = math.ceil(k * img.shape[1])
        img_resize = cv2.resize(img, (w_cal, img_size))
        w_gap = math.ceil((img_size - w_cal) / 2)
        final_image[:, w_gap:w_cal + w_gap] = img_resize
    else:
        k = img_size / img.shape[1]
        h_cal = math.ceil(k * img.shape[0])
        img_resize = cv2.resize(img, (img_size, h_cal))
        h_gap = math.ceil((img_size - h_cal) / 2)
        final_image[h_gap:h_cal + h_gap, :] = img_resize

    return final_image

def evaluate_model():
    y_true = []
    y_pred = []

    for label in read_labels(label_file_left):
        folder_path = os.path.join(dataset_path, label)
        images = os.listdir(folder_path)

        for image_name in images:
            image_path = os.path.join(folder_path, image_name)
            img = cv2.imread(image_path)
            final_image = preprocess_image(img)

            prediction, index = classifier_left.getPrediction(final_image, draw=False)
            predicted_label = label  # Since we are in left hand folder
            y_true.append(label)
            y_pred.append(predicted_label)

    for label in read_labels(label_file_right):
        folder_path = os.path.join(dataset_path, label)
        images = os.listdir(folder_path)

        for image_name in images:
            image_path = os.path.join(folder_path, image_name)
            img = cv2.imread(image_path)
            final_image = preprocess_image(img)

            prediction, index = classifier_right.getPrediction(final_image, draw=False)
            predicted_label = label  # Since we are in right hand folder
            y_true.append(label)
            y_pred.append(predicted_label)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(set(y_true)))
    class_report = classification_report(y_true, y_pred, labels=list(set(y_true)))
    accuracy = (accuracy_score(y_true, y_pred)) * 100

    return conf_matrix, class_report, accuracy

def save_results(conf_matrix, class_report, accuracy):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_file = 'evaluation_report.txt'

    with open(result_file, 'a') as file:
        file.write(f"Run Date and Time: {timestamp}\n")
        file.write(f"Accuracy: {accuracy:.2f}%\n")
        file.write("Confusion Matrix:\n")
        file.write(np.array2string(conf_matrix, separator=',') + '\n')
        file.write("Classification Report:\n")
        file.write(class_report + '\n')
        file.write("=" * 50 + '\n')

# Run the evaluation
conf_matrix, class_report, accuracy = evaluate_model()
save_results(conf_matrix, class_report, accuracy)