"""
Universal Training Pipeline Template
=====================================
This is the MASTER training loop that works for ANY model.
Copy this for each phase and modify only the model & dataset parts.

Author: Psychologist AI Team
Phase: 0 (Foundation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime


# ============================================
# 1. DATASET CLASS (Modify for each phase)
# ============================================
class CustomDataset(Dataset):
    """
    Template dataset class.
    Replace this with your specific data loading logic.
    """
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


# ============================================
# 2. MODEL CLASS (Modify for each phase)
# ============================================
class SimpleModel(nn.Module):
    """
    Template model class.
    Replace this with your CNN, RNN, Transformer, etc.
    """
    def __init__(self, input_size, num_classes):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.fc(x)


# ============================================
# 3. TRAINING FUNCTION (Universal - Don't Touch)
# ============================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    Returns average loss.
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
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
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


# ============================================
# 4. VALIDATION FUNCTION (Universal - Don't Touch)
# ============================================
def validate(model, dataloader, criterion, device):
    """
    Validate model on validation set.
    Returns average loss and accuracy.
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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


# ============================================
# 5. TESTING FUNCTION (Universal - Don't Touch)
# ============================================
def test(model, dataloader, device, class_names=None):
    """
    Test model on test set.
    Returns accuracy and confusion matrix.
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100 * correct / total
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    
    if class_names:
        report = classification_report(all_labels, all_preds, target_names=class_names)
        print("\nClassification Report:")
        print(report)
    
    return test_acc, cm


# ============================================
# 6. MAIN TRAINING LOOP (Universal - Modify hyperparams only)
# ============================================
def main():
    # ========== Hyperparameters ==========
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    INPUT_SIZE = 784  # Example: 28x28 for MNIST
    NUM_CLASSES = 10
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ========== Load Data (Replace with your data) ==========
    # Example: Replace with your actual data loading
    print("Loading data...")
    # train_dataset = CustomDataset(...)
    # val_dataset = CustomDataset(...)
    # test_dataset = CustomDataset(...)
    
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ========== Build Model ==========
    print("Building model...")
    model = SimpleModel(INPUT_SIZE, NUM_CLASSES).to(device)
    
    # ========== Loss & Optimizer ==========
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ========== Training Loop ==========
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    # ========== Test ==========
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load('models/best_model.pth'))
    test_acc, cm = test(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    
    # ========== Save Training History ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'reports/training_history_{timestamp}.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # ========== Plot Results ==========
    plot_history(history)
    
    print("\n✅ Training complete!")


# ============================================
# 7. VISUALIZATION (Universal - Don't Touch)
# ============================================
def plot_history(history):
    """
    Plot training and validation loss/accuracy.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'reports/training_plot_{timestamp}.png')
    print(f"✓ Saved training plot to reports/")


# ============================================
# 8. RUN
# ============================================
if __name__ == "__main__":
    main()
