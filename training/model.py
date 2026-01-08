"""
CNN Model Architecture for Facial Emotion Recognition
======================================================
Simple but effective CNN for 5-class emotion classification.

Architecture:
    Conv → ReLU → MaxPool (32 filters)
    Conv → ReLU → MaxPool (64 filters)
    Conv → ReLU → MaxPool (128 filters)
    Flatten
    FC → ReLU → Dropout (256 units)
    FC → Softmax (5 classes)

Author: Psychologist AI Team
Phase: 1 (Facial Emotion Recognition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# MAIN MODEL
# ============================================

class EmotionCNN(nn.Module):
    """
    Simple CNN for facial emotion recognition.
    
    Input: (batch, 1, 48, 48) grayscale images
    Output: (batch, 5) class logits
    
    Parameters: ~300K (lightweight and fast)
    """
    
    def __init__(self, num_classes=5, input_size=48, dropout=0.5):
        super(EmotionCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Conv Block 1: 1 → 32 channels
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 48x48 → 24x24
        
        # Conv Block 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 24x24 → 12x12
        
        # Conv Block 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 12x12 → 6x6
        
        # Calculate flattened size
        # After 3 pooling layers: 48 → 24 → 12 → 6
        flattened_size = 128 * (input_size // 8) * (input_size // 8)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 1, H, W)
        
        Returns:
            Logits (batch, num_classes)
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """
        Predict with softmax probabilities.
        
        Args:
            x: Input tensor (batch, 1, H, W)
        
        Returns:
            Probabilities (batch, num_classes)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs


# ============================================
# DEEPER MODEL (Optional - If You Need More Power)
# ============================================

class EmotionCNNDeep(nn.Module):
    """
    Deeper CNN with residual-like connections.
    Use this if simple CNN plateaus < 70% accuracy.
    
    Input: (batch, 1, 48, 48)
    Output: (batch, 5)
    Parameters: ~800K
    """
    
    def __init__(self, num_classes=5, input_size=48):
        super(EmotionCNNDeep, self).__init__()
        
        # Conv Block 1
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv Block 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv Block 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # FC Layers
        flattened_size = 256 * (input_size // 8) * (input_size // 8)
        self.fc1 = nn.Linear(flattened_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.bn1(x)
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.bn3(x)
        x = self.pool3(x)
        
        # Flatten & FC
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


# ============================================
# MODEL UTILITIES
# ============================================

def count_parameters(model):
    """
    Count total trainable parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_size=(1, 1, 48, 48)):
    """
    Print model architecture summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
    """
    print("=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    print(model)
    print("=" * 60)
    print(f"Total Parameters: {count_parameters(model):,}")
    print("=" * 60)
    
    # Test forward pass
    dummy_input = torch.randn(input_size)
    output = model(dummy_input)
    print(f"Input Shape:  {tuple(dummy_input.shape)}")
    print(f"Output Shape: {tuple(output.shape)}")
    print("=" * 60)


def save_model(model, save_path, config=None):
    """
    Save model weights and configuration.
    
    Args:
        model: PyTorch model
        save_path: Path to save .pth file
        config: Optional config dictionary
    """
    import json
    import os
    
    # Save model weights
    torch.save(model.state_dict(), save_path)
    print(f"✓ Model weights saved to {save_path}")
    
    # Save config
    if config is not None:
        config_path = save_path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✓ Config saved to {config_path}")


def load_model(model_class, weights_path, num_classes=5, input_size=48):
    """
    Load trained model from weights.
    
    Args:
        model_class: Model class (EmotionCNN or EmotionCNNDeep)
        weights_path: Path to .pth file
        num_classes: Number of output classes
        input_size: Input image size
    
    Returns:
        Loaded model in eval mode
    """
    model = model_class(num_classes=num_classes, input_size=input_size)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    print(f"✓ Model loaded from {weights_path}")
    return model


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    print("Testing EmotionCNN model...")
    
    # Create model
    model = EmotionCNN(num_classes=5, input_size=48)
    
    # Print summary
    print_model_summary(model)
    
    # Test forward pass
    dummy_input = torch.randn(4, 1, 48, 48)  # Batch of 4 images
    output = model(dummy_input)
    
    print("\n" + "=" * 60)
    print("FORWARD PASS TEST")
    print("=" * 60)
    print(f"Input batch: {dummy_input.shape}")
    print(f"Output logits: {output.shape}")
    print(f"Sample logits: {output[0].detach().numpy()}")
    
    # Test prediction (with softmax)
    probs = model.predict(dummy_input)
    print(f"\nProbabilities: {probs[0].detach().numpy()}")
    print(f"Sum of probabilities: {probs[0].sum().item():.4f}")
    print(f"Predicted class: {probs[0].argmax().item()}")
    
    print("\n✅ Model test passed!")
    
    # Test deeper model
    print("\n" + "=" * 60)
    print("Testing EmotionCNNDeep model...")
    print("=" * 60)
    
    deep_model = EmotionCNNDeep(num_classes=5, input_size=48)
    print(f"Deep model parameters: {count_parameters(deep_model):,}")
    
    output_deep = deep_model(dummy_input)
    print(f"Deep model output: {output_deep.shape}")
    
    print("\n✅ All models working correctly!")
