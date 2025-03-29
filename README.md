# Face Mask Detection, Classification, and Segmentation

This project implements a computer vision solution to classify and segment face masks in images, utilizing both traditional machine learning methods and deep learning techniques.

## Introduction

The goal of this project is to develop methods for:
1. Binary classification of images into "with mask" and "without mask" categories
2. Segmentation of face mask regions in images

The project explores multiple approaches:
- Handcrafted feature extraction with traditional ML classifiers
- Convolutional Neural Networks (CNNs) for classification
- Traditional image processing techniques for segmentation
- U-Net architecture for advanced mask segmentation

## Dataset

Two datasets were used for this project:

1. **Classification Dataset**: [Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)
   - Contains images of people with and without face masks
   - Used for binary classification tasks

2. **Segmentation Dataset**: [MFSD](https://github.com/sadjadrz/MFSD)
   - Includes images of masked faces with corresponding ground truth segmentation masks
   - Used for mask segmentation tasks

## Methodology

### 1. Binary Classification Using Handcrafted Features

For this task, we extracted the following handcrafted features:
- Color histograms: Capturing the color distribution in the images
- Histogram of Oriented Gradients (HOG): Capturing shape information

These features were then used to train two machine learning classifiers:
- Support Vector Machine (SVM) with RBF kernel
- Neural Network (Multi-layer Perceptron)

### 2. Binary Classification Using CNN

We designed a custom CNN architecture for face mask classification:
- 3 convolutional blocks with max pooling
- Dropout layers to prevent overfitting
- Fully connected layers for final classification

We experimented with different hyperparameters:
- Learning rates: 0.01, 0.001, 0.0001
- Optimizers: SGD (with momentum), Adam
- Batch sizes: 16, 32, 64

### 3. Region Segmentation Using Traditional Techniques

We implemented several traditional segmentation methods:
- Color thresholding: Using HSV color space to identify mask regions
- Edge-based segmentation: Using Canny edge detection and contour finding
- Watershed segmentation: Using watershed algorithm for region identification

### 4. Mask Segmentation Using U-Net

We implemented the U-Net architecture for precise mask segmentation:
- Encoder-decoder structure with skip connections
- Dice loss function to optimize segmentation accuracy
- Evaluation using IoU and Dice score metrics

## Hyperparameters and Experiments

### CNN Hyperparameter Tuning

We experimented with different configurations:

| Learning Rate | Optimizer | Batch Size | Accuracy |
|---------------|-----------|------------|----------|
| 0.001         | Adam      | 32         | 0.96     |
| 0.001         | SGD       | 32         | 0.92     |
| 0.0001        | Adam      | 64         | 0.95     |
| ...           | ...       | ...        | ...      |

The best hyperparameter combination was:
- Learning Rate: 0.001
- Optimizer: Adam
- Batch Size: 32

### U-Net Training

For U-Net, we used:
- Batch size: 4
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: Dice Loss
- Number of epochs: 10

## Results

### Classification Results

| Method                    | Accuracy |
|---------------------------|----------|
| SVM (Handcrafted features)| 93.28 %  |
| NN (Handcrafted features) | 90.84 %  |
| CNN                       | 96.46 %  |

The CNN outperformed the traditional machine learning classifiers, demonstrating the power of deep learning for this task.

### Segmentation Results

| Method             | IoU    | Dice Score |
|--------------------|--------|------------|
| Color Thresholding | 0.253  | 0.339      |
| Edge-based         | 0.269  | 0.378      |
| Watershed          | 0.214  | 0.27       |
| U-Net              | 0.782  | 0.869      |

U-Net significantly outperformed traditional segmentation methods, providing much more precise mask segmentation.

## Observations and Analysis

- CNN classification was robust to variations in mask types and face orientations
- Handcrafted features worked reasonably well but required more engineering
- Traditional segmentation methods struggled with complex backgrounds and lighting variations
- U-Net produced clean and accurate segmentation masks, even for partially occluded faces
- Data augmentation helped improve the robustness of both classification and segmentation models

Challenges faced:
- Variability in mask colors and types
- Diverse lighting conditions and backgrounds
- Limited dataset size for U-Net training
- Balancing model complexity with performance


## How to Run the Code

### Binary Classification
1. Run the notebook

### Region Segmentation
1. Download the dataset using the link given above
2. Extract the contents of the 'dataset/1' folder into the dataset folde
3. Rename 'face-crop' and 'face-crop-segmentation' folders to 'images' and 'masks'
4. Run the notebook 