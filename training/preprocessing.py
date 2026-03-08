"""
Preprocessing Utilities for Facial Emotion Recognition
========================================================
Standard preprocessing pipeline that MUST be used consistently
during training, validation, and inference.

Author: Psychologist AI Team
Phase: 1 (Facial Emotion Recognition)
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


# ============================================
# CORE PREPROCESSING (Must Stay Consistent!)
# ============================================

def preprocess_face(image, target_size=(48, 48)):
    """
    Standard preprocessing for facial emotion recognition.
    Use this EVERYWHERE (training, validation, inference).
    
    Args:
        image: Input image (BGR, RGB, or grayscale)
        target_size: Target dimensions (width, height)
    
    Returns:
        Preprocessed image tensor (1, H, W) for PyTorch
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            # Assume BGR (OpenCV default)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:, :, 0]  # Take first channel
    else:
        gray = image
    
    # Resize to target size
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = resized.astype('float32') / 255.0
    
    # Add channel dimension (1, H, W)
    tensor = np.expand_dims(normalized, axis=0)
    
    return tensor


def preprocess_face_torch(image, target_size=(48, 48)):
    """
    PyTorch-native preprocessing with tensor output.
    
    Args:
        image: NumPy array or PIL Image
        target_size: Target dimensions
    
    Returns:
        PyTorch tensor (1, 1, H, W) ready for model input
    """
    # Convert to grayscale
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        pil_image = Image.fromarray(gray)
    else:
        pil_image = image.convert('L')  # Convert PIL to grayscale
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
    ])
    
    # Apply transforms
    tensor = transform(pil_image)
    
    # Add batch dimension (1, C, H, W)
    tensor = tensor.unsqueeze(0)
    
    return tensor


# ============================================
# DATA AUGMENTATION (Training Only!)
# ============================================

def get_train_transforms(target_size=(48, 48)):
    """
    Data augmentation pipeline for training.
    DO NOT use during validation/testing/inference!
    
    Returns:
        torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            shear=5
        ),
        # Brightness/contrast jitter — effective even on grayscale images.
        # Simulates different lighting conditions across subjects.
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        # RandomErasing masks out a random rectangle, forcing the model to
        # not rely on any single facial region (e.g. always using the mouth for happy).
        # Applied after ToTensor on the [0-1] normalized tensor.
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])


def get_val_transforms(target_size=(48, 48)):
    """
    Validation/Test preprocessing (no augmentation).
    
    Returns:
        torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


# ============================================
# FACE DETECTION UTILITIES
# ============================================

def detect_faces_haar(image, scale_factor=1.3, min_neighbors=5):
    """
    Detect faces using Haar Cascade (fast but less accurate).
    
    Args:
        image: Input image (BGR)
        scale_factor: Image pyramid scale
        min_neighbors: Quality threshold
    
    Returns:
        List of (x, y, w, h) bounding boxes
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    
    return faces


def extract_face_roi(image, bbox, padding=10):
    """
    Extract face region of interest with padding.
    
    Args:
        image: Input image
        bbox: (x, y, w, h) bounding box
        padding: Extra pixels around face
    
    Returns:
        Cropped face image
    """
    x, y, w, h = bbox
    
    # Add padding
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(image.shape[1], x + w + padding)
    y_end = min(image.shape[0], y + h + padding)
    
    # Extract ROI
    face_roi = image[y_start:y_end, x_start:x_end]
    
    return face_roi


# ============================================
# BATCH PREPROCESSING FOR DATASETS
# ============================================

def preprocess_dataset_folder(input_folder, output_folder, target_size=(48, 48)):
    """
    Preprocess all images in a folder (batch processing).
    Useful for preparing downloaded datasets.
    
    Args:
        input_folder: Path to raw images
        output_folder: Path to save preprocessed images
        target_size: Target dimensions
    """
    import os
    from pathlib import Path
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    for img_file in os.listdir(input_folder):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            # Read image
            img_path = os.path.join(input_folder, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Skipping {img_file} (couldn't read)")
                continue
            
            # Preprocess
            preprocessed = preprocess_face(image, target_size)
            
            # Convert back to uint8 for saving
            preprocessed_uint8 = (preprocessed[0] * 255).astype('uint8')
            
            # Save
            output_path = os.path.join(output_folder, img_file)
            cv2.imwrite(output_path, preprocessed_uint8)
    
    print(f"Preprocessed images saved to {output_folder}")


# ============================================
# VISUALIZATION UTILITIES
# ============================================

def visualize_preprocessing(image_path, target_size=(48, 48)):
    """
    Visualize preprocessing steps for debugging.
    
    Args:
        image_path: Path to test image
        target_size: Target dimensions
    """
    import matplotlib.pyplot as plt
    
    # Load image
    image = cv2.imread(image_path)
    
    # Original
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Preprocessed
    preprocessed = preprocess_face(image, target_size)[0]
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('Grayscale')
    axes[1].axis('off')
    
    axes[2].imshow(preprocessed, cmap='gray')
    axes[2].set_title(f'Preprocessed ({target_size[0]}x{target_size[1]})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    # Test preprocessing with a dummy image
    print("Testing preprocessing utilities...")
    
    # Create dummy grayscale image
    dummy_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    # Test preprocessing
    preprocessed = preprocess_face(dummy_image, target_size=(48, 48))
    print(f"✓ Preprocessed shape: {preprocessed.shape}")
    print(f"✓ Value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    # Test PyTorch preprocessing
    preprocessed_torch = preprocess_face_torch(dummy_image, target_size=(48, 48))
    print(f"✓ PyTorch tensor shape: {preprocessed_torch.shape}")
    
    # Test augmentation transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    print(f"✓ Train transforms: {len(train_transform.transforms)} steps")
    print(f"✓ Val transforms: {len(val_transform.transforms)} steps")
    
    print("\n All preprocessing utilities working!")
