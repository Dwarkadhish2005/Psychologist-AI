"""
Training Script for Facial Emotion Recognition
===============================================
Complete training pipeline for EmotionCNN model.

Usage:
    python train_emotion_model.py

Before running:
    1. Place your dataset in data/face_emotion/
    2. Organize images into train/val/test folders
    3. Each folder should have subfolders: angry, happy, sad, surprise, neutral

Author: Psychologist AI Team
Phase: 1 (Facial Emotion Recognition)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime
from pathlib import Path

# Import custom modules
from model import EmotionCNN, count_parameters
from preprocessing import get_train_transforms, get_val_transforms


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Training configuration"""
    
    # Paths
    DATA_DIR = 'data/face_emotion'
    MODEL_DIR = 'models/face_emotion'
    REPORTS_DIR = 'reports'
    
    # Model
    INPUT_SIZE = 48
    NUM_CLASSES = None  # Will be auto-detected from dataset
    CLASS_NAMES = None  # Will be auto-detected from dataset
    
    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Early stopping
    PATIENCE = 10
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================
# DATASET CLASS
# ============================================

class EmotionDataset(Dataset):
    """
    Custom dataset for emotion recognition from folders.
    
    Expected structure:
        data/face_emotion/train/
            ├── angry/
            ├── happy/
            ├── sad/
            ├── surprise/
            └── neutral/
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Root directory (e.g., 'data/face_emotion/train')
            transform: Optional transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"[OK] Loaded {len(self.images)} images from {root_dir}")
        print(f"  Class distribution: {self._get_class_distribution()}")
    
    def _get_class_distribution(self):
        """Get count of images per class"""
        from collections import Counter
        label_counts = Counter(self.labels)
        return {self.classes[idx]: count for idx, count in label_counts.items()}
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================
# TRAINING FUNCTIONS
# ============================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Returns:
        Average loss and accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Validate model.
    
    Returns:
        Average loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def evaluate_model(model, dataloader, device, class_names):
    """
    Comprehensive model evaluation with confusion matrix.
    
    Returns:
        Test accuracy and confusion matrix
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    test_acc = 100 * correct / len(all_labels)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    
    return test_acc, cm


# ============================================
# VISUALIZATION
# ============================================

def plot_training_history(history, save_path):
    """Plot training curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f" Training plot saved to {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix heatmap"""
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f" Confusion matrix saved to {save_path}")


# ============================================
# MAIN TRAINING FUNCTION
# ============================================

def main():
    """Main training pipeline"""
    
    config = Config()
    
    # Create directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("FACIAL EMOTION RECOGNITION - TRAINING")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("=" * 60)
    
    # ========== Load Data ==========
    print("\nLoading datasets...")
    
    train_transform = get_train_transforms(target_size=(config.INPUT_SIZE, config.INPUT_SIZE))
    val_transform = get_val_transforms(target_size=(config.INPUT_SIZE, config.INPUT_SIZE))
    
    train_dataset = EmotionDataset(
        os.path.join(config.DATA_DIR, 'train'),
        transform=train_transform
    )
    val_dataset = EmotionDataset(
        os.path.join(config.DATA_DIR, 'val'),
        transform=val_transform
    )
    test_dataset = EmotionDataset(
        os.path.join(config.DATA_DIR, 'test'),
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Auto-detect number of classes and class names from dataset
    config.NUM_CLASSES = len(train_dataset.classes)
    config.CLASS_NAMES = train_dataset.classes
    print(f"\n[OK] Detected {config.NUM_CLASSES} classes: {', '.join(config.CLASS_NAMES)}")
    
    # ========== Build Model ==========
    print("\nBuilding model...")
    model = EmotionCNN(num_classes=config.NUM_CLASSES, input_size=config.INPUT_SIZE)
    model = model.to(config.DEVICE)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # ========== Loss & Optimizer ==========
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # ========== Training Loop ==========
    print(f"\nTraining for {config.NUM_EPOCHS} epochs...")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'emotion_cnn_best.pth'))
            print(f"  Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # ========== Final Evaluation ==========
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'emotion_cnn_best.pth')))
    
    # Evaluate on test set
    test_acc, cm = evaluate_model(model, test_loader, config.DEVICE, config.CLASS_NAMES)
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    
    # ========== Save Everything ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f'emotion_cnn_final_{timestamp}.pth'))
    
    # Save labels
    labels_dict = {idx: name for idx, name in enumerate(config.CLASS_NAMES)}
    with open(os.path.join(config.MODEL_DIR, 'labels.json'), 'w') as f:
        json.dump(labels_dict, f, indent=4)
    
    # Save config
    config_dict = {
        'input_size': [1, config.INPUT_SIZE, config.INPUT_SIZE],
        'num_classes': config.NUM_CLASSES,
        'class_names': config.CLASS_NAMES,
        'architecture': 'EmotionCNN',
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'timestamp': timestamp
    }
    with open(os.path.join(config.MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # Save training history
    with open(os.path.join(config.REPORTS_DIR, f'training_history_{timestamp}.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Save plots
    plot_training_history(history, os.path.join(config.REPORTS_DIR, f'training_plot_{timestamp}.png'))
    plot_confusion_matrix(cm, config.CLASS_NAMES, os.path.join(config.REPORTS_DIR, f'confusion_matrix_{timestamp}.png'))
    
    print("\n" + "=" * 60)
    print("[OK] TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best model saved to: {config.MODEL_DIR}/emotion_cnn_best.pth")
    print(f"Reports saved to: {config.REPORTS_DIR}/")
    print("=" * 60)


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()
