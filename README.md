


[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=450&size=40&duration=2500&pause=1000&width=1500&height=70&lines=%F0%9F%A4%98+Sign+Language+To+Text+Convertor+for+Speech+Impaired)](https://git.io/typing-svg)

The **Sign Language to Text Converter for Speech Impaired** is a project aimed at translating Sign Language into written text. It uses machine learning and computer vision to understand sign language and provide accurate translations, making communication more accessible and easy for people with hearing and speech impairments.

## 🚀 Features

- **Real-time Gesture Recognition**: Translates  hand signs into text instantly.
- **High Accuracy**: Built on a robust machine learning model trained on a diverse dataset.
- **Intuitive Interface**: User-friendly design for seamless interaction.
- **Customizable Vocabulary**: Users can add or modify signs for personal use.

## 🛠 Built With

We have mainly used Python and some of its libraries and tools to build this project. Some of these are:

- **🟡 cvzone**: Used to access the camera to capture hand gestures, including the built-in `HandDetector` library for hand detection.
  
- **🔢 Numpy**: Used for model predictions, specifically for predicting different hand signs and the letters used.

- **🖐 Mediapipe**: A cross-platform framework developed by Google for processing video and multimedia. It provides advanced capabilities for hand tracking and gesture recognition, making it ideal for real-time Sign Language recognition.

- **🔍 OpenCV**: An open-source computer vision and machine learning library that provides tools for image and video processing, essential for capturing and manipulating frames from a webcam.

- **🧠 TensorFlow**: An open-source machine learning framework developed by Google, widely used for building and training machine learning models. In this project, TensorFlow runs the neural network model that recognizes Sign Language gestures.

- **🤖 Teachable Machine**: A user-friendly web tool that allows anyone to create machine learning models without coding, designed for tasks like image classification and pose detection.

## 🛠 Technologies Used

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0.82-blue.svg)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.1-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.15.0-red.svg)](https://keras.io/)
[![Mediapipe](https://img.shields.io/badge/Mediapipe-2.15.0-green.svg)](https://mediapipe.dev/)
[![Numpy](https://img.shields.io/badge/Numpy-1.26.4-blue.svg)](https://numpy.org/)
[![cvzone](https://img.shields.io/badge/cvzone-1.26.4-lightgrey.svg)](https://pypi.org/project/cvzone/)
[![Teachable Machine](https://img.shields.io/badge/Teachable%20Machine-ML-orange.svg)](https://teachablemachine.withgoogle.com/)


## 📦 Installation

1. Clone the repository:
   ```bash
    git clone https://github.com/ChirayuDodiya/Sign-Language-To-Text-Convertor-for-Speech-Impaired

2. Install required packages:
   ```bash
    pip install -r requirements.txt

## 🏗 Process Overview

The process of developing the Sign Language to Text Converter involves three main steps: data collection, model training, and testing. Here's how each step works:

### 1. Data Collection

- **File**: `collect.py`
- **Description**: This script is used to collect data for training the model. 
- **How It Works**:
  1. Run the `collect.py` script.
  2. The script opens a user interface where you can click on letters to indicate the corresponding Sign Language. The Data is stored in \Data folder.
  3. Each click records the hand gesture as a data sample, which is saved for training.

### 2. Model Training

- **Tool**: Teachable Machine by Google
- **Description**: The collected data is uploaded to Teachable Machine, where it is used to train the model.
- **How It Works**:
  1. Once you have collected sufficient data, export it from `collect.py`.
  2. Upload the dataset to Teachable Machine.
  3. Train the model using the uploaded data to recognize the Sign Language.
  4. After training, export the model as a Keras model file (usually in `.h5` format).

### 3. Testing

- **File**: `test.py`
- **Description**: This script is used to test the trained model.
- **How It Works**:
  1. Run the `test.py` script.
  2. The script loads the trained Keras model.
  3. It accesses the camera to capture live hand gestures.
  4. The model predicts the corresponding Sign Language in real-time, displaying the recognized letter on the screen.



## 🤝 Contributors

- [@Chirayu Dodiya](https://github.com/ChirayuDodiya)
- [@Mayank Bagul](https://github.com/likemacc)
- [@Nandini Gadhvi](https://github.com/NadiniGadhvi)
- [@Abhishek Guna](https://github.com/HINOKAM-ii)
- [@Harshil Vasava](https://github.com/harshilV14)


## 🎥 Demo

In this section, we provide a demonstration of the Sign Language to Text Converter. Below are the steps to showcase how the application works, along with visual aids.

### How to Use the Sign Language to Text Converter

1. **Start the Application**:
   - Run the `test.py` script to initiate the application.
   - Ensure your webcam is connected and accessible.

2. **User Interface**:
   - Upon running, a window will open showing the live feed from your webcam.

3. **Perform Hand Signs**:
   - Position your hand in front of the camera and perform the hand signs.
   - The model will recognize the sign in real-time and display the corresponding letter on the screen.

#### Video Demonstration

Watch the following video to see the Sign Language to Text Converter in action:

[Sign Language to Text Demo Video](https://drive.google.com/file/d/1ibvHMfsna9GtM5IRJya9GiTj9_yzX7bt/view)

### Notes

- Ensure you are in a well-lit area for better gesture recognition.
- If the model does not recognize your signs accurately, consider retraining the model with more diverse data samples.

### Feedback

We welcome feedback! If you have any suggestions or issues, please open an issue on our GitHub repository.



