# Image_Classifier_for_Cifar-10

This project implements an image classification model using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. The goal is to classify images into one of the ten predefined categories such as airplanes, cars, birds, dogs and more. The repository contains the trained model, the GUI script for testing images and supporting documentation.

---

## 📌 Project Overview

CIFAR-10 is a benchmark dataset commonly used in machine learning research. It contains small RGB images across ten classes. This project focuses on designing and training a CNN to achieve reliable classification performance.

The repository includes:
- A trained model (`cifar10_cnn.h5`)
- A GUI script (`cifar10_gui.py`)
- A complete project description

---

## 🧠 Features
- CNN built with TensorFlow/Keras  
- Image preprocessing and normalization  
- Training with accuracy and loss visualization  
- GUI to test custom images  
- Pretrained model available for direct use  

---

## 📂 Dataset Information

**CIFAR-10 Dataset**  
- 60,000 RGB images (32x32)  
- 10 categories: `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`  
- 50,000 training images  
- 10,000 test images  

Dataset Source:  
https://www.cs.toronto.edu/~kriz/cifar.html

---

## ⚙️ Model Architecture

The CNN includes:
- Convolution layers  
- ReLU activation  
- MaxPooling layers  
- Dropout layer to reduce overfitting  
- Dense layers  
- Softmax classifier for 10 output classes  

**Optimizer:** Adam  
**Loss:** Categorical Crossentropy  
**Metric:** Accuracy  

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/p-kumar-d3f4ult/Image_Classifier_for_Cifar-10.git
cd Image_Classifier_for_Cifar-10
