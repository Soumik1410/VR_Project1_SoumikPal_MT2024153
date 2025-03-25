import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import zipfile
import requests
from io import BytesIO
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

# Download and extract dataset
def download_and_extract_dataset():
    # Download face mask classification dataset
    url = "https://github.com/chandrikadeb7/Face-Mask-Detection/archive/refs/heads/master.zip"
    print("Downloading face mask dataset...")
    r = requests.get(url, stream=True)
    with zipfile.ZipFile(BytesIO(r.content)) as zip_ref:
        zip_ref.extractall("./")
    
    # Create directories for processed data
    os.makedirs("data/with_mask", exist_ok=True)
    os.makedirs("data/without_mask", exist_ok=True)
    
    # Copy files to our working directory
    source_with_mask = "./Face-Mask-Detection-master/dataset/with_mask"
    source_without_mask = "./Face-Mask-Detection-master/dataset/without_mask"
    
    for filename in os.listdir(source_with_mask):
        src_path = os.path.join(source_with_mask, filename)
        dst_path = os.path.join("data/with_mask", filename)
        shutil.copy(src_path, dst_path)
    
    for filename in os.listdir(source_without_mask):
        src_path = os.path.join(source_without_mask, filename)
        dst_path = os.path.join("data/without_mask", filename)
        shutil.copy(src_path, dst_path)
    
    print("Dataset downloaded and extracted successfully.")

# Run the download function
download_and_extract_dataset()

# Load and preprocess images
def load_images(with_mask_dir, without_mask_dir):
    images = []
    labels = []
    
    # Load with_mask images
    for filename in tqdm(os.listdir(with_mask_dir), desc="Loading with_mask images"):
        img_path = os.path.join(with_mask_dir, filename)
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 64))  # Resize for consistency
            images.append(img)
            labels.append(1)  # 1 for with_mask
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    # Load without_mask images
    for filename in tqdm(os.listdir(without_mask_dir), desc="Loading without_mask images"):
        img_path = os.path.join(without_mask_dir, filename)
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 64))  # Resize for consistency
            images.append(img)
            labels.append(0)  # 0 for without_mask
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)

# Load dataset
images, labels = load_images("data/with_mask", "data/without_mask")

# Display some sample images
def plot_sample_images(images, labels, num_samples=5):
    plt.figure(figsize=(15, 6))
    mask_samples = np.where(labels == 1)[0][:num_samples]
    no_mask_samples = np.where(labels == 0)[0][:num_samples]
    
    for i, idx in enumerate(mask_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(images[idx])
        plt.title("With Mask")
        plt.axis("off")
    
    for i, idx in enumerate(no_mask_samples):
        plt.subplot(2, num_samples, i + num_samples + 1)
        plt.imshow(images[idx])
        plt.title("Without Mask")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

plot_sample_images(images, labels)

# Feature extraction functions
def extract_histogram_features(img):
    """Extract color histogram features"""
    features = []
    for channel in range(3):  # RGB channels
        hist = cv2.calcHist([img], [channel], None, [32], [0, 256])
        features.extend(hist.flatten())
    return features

def extract_hog_features(img):
    """Extract HOG (Histogram of Oriented Gradients) features"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Parameters for HOG
    cell_size = (8, 8)
    block_size = (2, 2)
    nbins = 9
    
    # Calculate gradient
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    
    # Calculate gradient magnitude and orientation
    magnitude, orientation = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # Quantize orientations into bins
    orientation = orientation / 180 * nbins
    
    # Simple HOG implementation for feature extraction
    hog_features = []
    for i in range(0, h, cell_size[0]):
        for j in range(0, w, cell_size[1]):
            if i + cell_size[0] <= h and j + cell_size[1] <= w:
                cell_magnitude = magnitude[i:i+cell_size[0], j:j+cell_size[1]]
                cell_orientation = orientation[i:i+cell_size[0], j:j+cell_size[1]]
                
                hist = np.zeros(nbins)
                for o_idx in range(nbins):
                    # Find pixels with orientations in the current bin
                    mask = ((cell_orientation >= o_idx) & (cell_orientation < (o_idx + 1)))
                    hist[o_idx] = np.sum(cell_magnitude[mask])
                
                hog_features.extend(hist)
    
    return hog_features

def extract_features(images):
    """Extract combined features from images"""
    all_features = []
    for img in tqdm(images, desc="Extracting features"):
        # Get histogram features
        hist_features = extract_histogram_features(img)
        
        # Get HOG features
        hog_features = extract_hog_features(img)
        
        # Combine features
        combined_features = np.concatenate([hist_features, hog_features])
        all_features.append(combined_features)
    
    return np.array(all_features)

# Extract features from all images
features = extract_features(images)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate SVM classifier
def train_svm(X_train, y_train, X_test, y_test):
    print("Training SVM classifier...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    svm.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"SVM Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Without Mask', 'With Mask']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Without Mask', 'With Mask'],
                yticklabels=['Without Mask', 'With Mask'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - SVM')
    plt.show()
    
    return svm, accuracy

# Train and evaluate Neural Network classifier
def train_nn(X_train, y_train, X_test, y_test):
    print("Training Neural Network classifier...")
    nn = MLPClassifier(hidden_layer_sizes=(100, 50), 
                       activation='relu', 
                       solver='adam',
                       alpha=0.0001,
                       max_iter=300,
                       random_state=42)
    
    nn.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = nn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Neural Network Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Without Mask', 'With Mask']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Without Mask', 'With Mask'],
                yticklabels=['Without Mask', 'With Mask'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Neural Network')
    plt.show()
    
    return nn, accuracy

# Train and evaluate classifiers
svm_model, svm_accuracy = train_svm(X_train_scaled, y_train, X_test_scaled, y_test)
nn_model, nn_accuracy = train_nn(X_train_scaled, y_train, X_test_scaled, y_test)

# Compare classifier performances
def compare_classifiers(classifiers, accuracies):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classifiers, accuracies, color=['blue', 'green'])
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2 - 0.1, 
                 bar.get_height() + 0.01, 
                 f'{acc:.4f}', 
                 fontsize=12)
    
    plt.ylim(0, 1.1)
    plt.title('Classifier Accuracy Comparison (Handcrafted Features)')
    plt.ylabel('Accuracy')
    plt.show()

compare_classifiers(['SVM', 'Neural Network'], [svm_accuracy, nn_accuracy])


# Define a custom dataset class
class FaceMaskDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Define data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Split data into training and testing sets
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    images, labels, test_size=0.2, random_state=42)

# Create datasets
train_dataset = FaceMaskDataset(X_train_img, y_train_img, transform=transform)
test_dataset = FaceMaskDataset(X_test_img, y_test_img, transform=transform)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define CNN model
class FaceMaskCNN(nn.Module):
    def __init__(self):
        super(FaceMaskCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # First convolutional block
        x = self.pool(F.relu(self.conv1(x)))
        # Second convolutional block
        x = self.pool(F.relu(self.conv2(x)))
        # Third convolutional block
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the output
        x = x.view(-1, 128 * 8 * 8)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function to train the model
def train_cnn(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device="cpu"):
    model = model.to(device)
    
    # Training history
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Testing phase
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_test_loss = running_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)
        
        accuracy = accuracy_score(all_labels, all_preds)
        test_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    return train_losses, test_losses, test_accuracies, all_preds, all_labels

# Function to plot training history
def plot_training_history(train_losses, test_losses, test_accuracies):
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    
    plt.tight_layout()
    plt.show()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model
model = FaceMaskCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
train_losses, test_losses, test_accuracies, y_pred_cnn, y_test_cnn = train_cnn(
    model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

# Plot training history
plot_training_history(train_losses, test_losses, test_accuracies)

# Evaluation
print("CNN Final Accuracy:", test_accuracies[-1])
print("\nClassification Report:")
print(classification_report(y_test_cnn, y_pred_cnn, target_names=['Without Mask', 'With Mask']))

# Plot confusion matrix
cm = confusion_matrix(y_test_cnn, y_pred_cnn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Without Mask', 'With Mask'],
            yticklabels=['Without Mask', 'With Mask'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - CNN')
plt.show()

# Hyperparameter experimentation
def cnn_hyperparameter_experiment():
    # Define hyperparameter combinations to try
    learning_rates = [0.01, 0.001, 0.0001]
    optimizers = [
        ('SGD', optim.SGD),
        ('Adam', optim.Adam)
    ]
    batch_sizes = [16, 32, 64]
    
    # Dictionary to store results
    results = []
    
    # Iterate through hyperparameter combinations
    for lr in learning_rates:
        for opt_name, opt_class in optimizers:
            for bs in batch_sizes:
                print(f"\nTesting hyperparameters: LR={lr}, Optimizer={opt_name}, Batch Size={bs}")
                
                # Create data loaders with the current batch size
                train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
                
                # Initialize model
                model = FaceMaskCNN()
                
                # Define optimizer with current learning rate
                if opt_name == 'SGD':
                    optimizer = opt_class(model.parameters(), lr=lr, momentum=0.9)
                else:
                    optimizer = opt_class(model.parameters(), lr=lr)
                
                # Train for fewer epochs to save time
                num_epochs = 5
                _, _, test_accuracies, _, _ = train_cnn(
                    model, train_loader, test_loader, criterion, optimizer, num_epochs, device)
                
                # Store the final accuracy
                final_accuracy = test_accuracies[-1]
                results.append({
                    'learning_rate': lr,
                    'optimizer': opt_name,
                    'batch_size': bs,
                    'accuracy': final_accuracy
                })
                
                print(f"Final accuracy: {final_accuracy:.4f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find the best hyperparameter combination
    best_idx = results_df['accuracy'].idxmax()
    best_params = results_df.iloc[best_idx]
    
    print("\nBest Hyperparameter Combination:")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"Optimizer: {best_params['optimizer']}")
    print(f"Batch Size: {best_params['batch_size']}")
    print(f"Accuracy: {best_params['accuracy']:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    for i, opt in enumerate(['SGD', 'Adam']):
        plt.subplot(2, 1, i+1)
        for bs in batch_sizes:
            df_subset = results_df[(results_df['optimizer'] == opt) & (results_df['batch_size'] == bs)]
            plt.plot(df_subset['learning_rate'], df_subset['accuracy'], 
                     marker='o', label=f'Batch Size={bs}')
        
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Accuracy')
        plt.title(f'Hyperparameter Tuning - {opt}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results_df, best_params

# Run hyperparameter experiment
results_df, best_params = cnn_hyperparameter_experiment()

# Compare all classifier performances (handcrafted+ML vs CNN)
def compare_all_classifiers(classifiers, accuracies):
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'red']
    bars = plt.bar(classifiers, accuracies, color=colors)
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2 - 0.1, 
                 bar.get_height() + 0.01, 
                 f'{acc:.4f}', 
                 fontsize=12)
    
    plt.ylim(0, 1.1)
    plt.title('Classifier Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.show()

# Compare SVM, NN, and CNN performances
compare_all_classifiers(['SVM (Handcrafted)', 'NN (Handcrafted)', 'CNN'], 
                        [svm_accuracy, nn_accuracy, test_accuracies[-1]])

# Save best models for later use
torch.save(model.state_dict(), 'cnn_face_mask_model.pth')