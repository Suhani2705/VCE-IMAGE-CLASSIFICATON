# **Attention-Driven Multi-Class Abnormality Detection in Video Capsule Endoscopy (VCE) Using Enhanced InceptionResNetV2**

## Overview
This project focuses on building an advanced image classification model to detect various abnormalities from video capsule endoscopy (VCE) images. Utilizing a hybrid architecture based on InceptionResNetV2, the model integrates custom CNN layers with attention mechanisms to improve feature extraction and classification performance.

## Table of Contents
- [Introduction](#Introduction)
- [Setup](#Setup)
- [Dataset Preparation](#Dataset-Preparation)
- [Model Architecture](#Model-Architecture)
- [Training and Evaluation](#Training-and-Evaluation)
- [Results](#Results)

## Introduction
The project aims to automatically classify abnormalities in VCE images, promoting the development of vendor-independent, generalized AI-based models for medical applications. The hybrid CNN architecture includes:
- Pretrained **InceptionResNetV2** backbone.
- Custom convolutional layers.
- Attention mechanisms for focused feature extraction.

## Setup

### Prerequisites
- Python 3.x
- TensorFlow 2.x
- Jupyter Notebook
- Required libraries: `numpy`, `tensorflow`, `sklearn`, `matplotlib`, `seaborn`.

## Dataset-Preparation

The dataset used for training and validation can be found at the following link:

[Training and Validation Dataset of Capsule Vision 2024 Challenge](https://figshare.com/articles/dataset/Training_and_Validation_Dataset_of_Capsule_Vision_2024_Challenge/26403469)

### Class Folders
The dataset consists of various classes of abnormalities found in Video Capsule Endoscopy (VCE) images, including:

- Angioectasia
- Bleeding
- Erosion
- Erythema
- Foreign Body
- Lymphangiectasia
- Normal
- Polyp
- Ulcer
- Worms

## Model-Architecture

The model employed for detecting abnormalities in Video Capsule Endoscopy (VCE) images is a hybrid architecture that combines the **InceptionResNetV2** model with custom Convolutional Neural Network (CNN) blocks and attention mechanisms. This design is aimed at enhancing feature extraction and classification performance.

### Key Components

1. **InceptionResNetV2**:
   - This pre-trained model serves as the backbone of our architecture. It is known for its deep residual learning capabilities and efficient training process, making it suitable for complex image classification tasks.
   - We utilize the model with the top layer removed (i.e., `include_top=False`) to leverage its feature extraction capabilities while allowing us to add our custom layers.

2. **Custom CNN Blocks**:
   - After the InceptionResNetV2 base, we added two custom CNN blocks, each consisting of:
     - **Convolutional Layer**: Extracts features from the input images using filters.
     - **Batch Normalization**: Normalizes the output of the convolutional layers to improve training speed and stability.
     - **Activation Function**: We use ReLU (Rectified Linear Unit) activation for introducing non-linearity into the model.
     - **MaxPooling Layer**: Reduces the spatial dimensions of the feature maps, retaining the most significant features and reducing computational complexity.
     - **Dropout Layer**: Regularization technique to prevent overfitting by randomly setting a fraction of input units to 0 during training.

3. **Attention Mechanism**:
   - We implement an attention block after each custom CNN block. The attention mechanism helps the model focus on relevant features in the input images by:
     - Applying a global average pooling operation to capture the spatial information.
     - Using dense layers to generate attention scores, which are multiplied with the input features to enhance important features and suppress irrelevant ones.

4. **Output Layer**:
   - Finally, we utilize a **Global Average Pooling Layer** to convert the feature maps into a one-dimensional vector, followed by a **Dense Layer** with a softmax activation function. This layer outputs the probabilities for each class of abnormalities.

### Model Summary
- The model is fine-tuned by training the last 20 layers of the InceptionResNetV2 base while keeping the earlier layers frozen. This allows the model to adapt to the specific characteristics of VCE images while leveraging the pre-learned features from the ImageNet dataset.

This architecture balances the strengths of a pre-trained model with customized components tailored for the specific task of abnormality detection in VCE images.

## Training-and-Evaluation

### Training Setup
The model was trained using the following parameters:
- **Epochs**: 20
- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Class Weights**: Calculated to address class imbalance in the dataset, ensuring that the model gives appropriate importance to each class during training.

### Data Augmentation
To enhance the robustness of the model and prevent overfitting, we applied data augmentation techniques during training:
- **Rotation**: Images were randomly rotated up to 30 degrees.
- **Width Shift**: Randomly shifted along the width (up to 20%).
- **Height Shift**: Randomly shifted along the height (up to 20%).
- **Shear**: Random shearing transformations applied.
- **Zoom**: Random zooming applied (up to 30%).
- **Horizontal Flip**: Randomly flipping images horizontally.

These transformations help the model generalize better by exposing it to variations of the training data.

### Callbacks
We implemented several callbacks to optimize the training process:
- **EarlyStopping**: Monitors validation loss, stopping training when the model performance stops improving for a specified number of epochs (patience set to 10).
- **ModelCheckpoint**: Saves model weights at specific intervals, allowing recovery of the best-performing model.
- **ReduceLROnPlateau**: Reduces the learning rate when the validation loss plateaus, helping the model escape local minima.

### Evaluation Metrics
To comprehensively evaluate the model's performance, the following metrics were calculated:

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **AUC-ROC Score**: The area under the receiver operating characteristic curve, providing insight into the model's ability to discriminate between classes.
- **Precision and Recall**: These metrics help in understanding the model's performance on positive class predictions.
- **F1 Score**: The harmonic mean of precision and recall, giving a balanced measure of performance.
- **Specificity**: The model's ability to correctly identify negative cases.
- **Average Precision**: Calculated using the precision-recall curve, indicating the trade-off between precision and recall at various thresholds.

### Confusion Matrix
- A confusion matrix was generated to visualize the performance across different classes. This matrix shows true positives, false positives, true negatives, and false negatives for each class, allowing for an intuitive understanding of where the model performs well and where it struggles.

## Results
The model achieved the following performance metrics:
- **Mean AUC**: 0.9906
- **Mean Specificity**: 0.9917
- **Mean Average Precision**: 0.8820
- **Mean Sensitivity**: 0.8358
- **Mean F1 Score**: 0.8186
- **Balanced Accuracy**: 0.8474
