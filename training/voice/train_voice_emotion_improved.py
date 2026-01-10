"""
IMPROVED Voice Emotion Training Script
=======================================
Fixes for low happy accuracy:
1. Class weights to fix imbalance
2. More epochs (100 instead of 50)
3. Lower learning rate (0.0005)
4. Better early stopping patience
5. Data augmentation

Based on diagnosis: Current accuracy 40% → Target 60-70%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from collections import Counter
import sys

# Add modules to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'training' / 'voice'))

from voice_emotion_model import VoiceEmotionModel, StressDetector
from feature_extraction import extract_all_features
from audio_preprocessing import preprocess_audio
import librosa.effects as effects


# ============================================
# DATA AUGMENTATION
# ============================================

def augment_audio(audio, sr, augmentation_prob=0.5):
    """
    Apply audio augmentation for training.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        augmentation_prob: Probability of applying each augmentation
    
    Returns:
        augmented_audio: Augmented audio signal
    """
    # Pitch shift (±2 semitones)
    if np.random.random() < augmentation_prob:
        n_steps = np.random.uniform(-2, 2)
        audio = effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    # Time stretch (±10%)
    if np.random.random() < augmentation_prob:
        rate = np.random.uniform(0.9, 1.1)
        audio = effects.time_stretch(audio, rate=rate)
    
    # Add noise (±5% volume)
    if np.random.random() < augmentation_prob:
        noise = np.random.randn(len(audio)) * 0.005
        audio = audio + noise
        audio = np.clip(audio, -1.0, 1.0)
    
    return audio


# ============================================
# DATASET CLASS
# ============================================

class VoiceEmotionDataset(Dataset):
    """Dataset for voice emotion recognition with augmentation."""
    
    def __init__(self, file_paths, cache_dir, use_augmentation=False):
        """
        Initialize dataset.
        
        Args:
            file_paths: List of audio file paths
            cache_dir: Directory for cached features
            use_augmentation: Whether to use data augmentation
        """
        self.file_paths = file_paths
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_augmentation = use_augmentation
        
        # Load emotion labels
        from dataset_utils import TARGET_EMOTION_TO_IDX
        self.emotion_to_idx = TARGET_EMOTION_TO_IDX
        
        # Extract labels from filenames
        self.labels = []
        for file_path in file_paths:
            label = self._extract_label(file_path)
            self.labels.append(label)
    
    def _extract_label(self, file_path):
        """Extract emotion label from filename."""
        from dataset_utils import map_ravdess_label, map_tess_label
        
        filename = Path(file_path).name
        
        # RAVDESS
        if 'Actor_' in file_path:
            parts = filename.replace('.wav', '').split('-')
            if len(parts) == 7:
                emotion_map = {
                    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
                }
                original_emotion = emotion_map.get(parts[2], 'unknown')
                target_emotion, _ = map_ravdess_label(original_emotion)
                return self.emotion_to_idx.get(target_emotion, 3)  # Default to neutral
        
        # TESS
        elif '_' in filename:
            parts = filename.replace('.wav', '').split('_')
            if len(parts) >= 3:
                original_emotion = parts[-1]
                target_emotion, _ = map_tess_label(original_emotion)
                return self.emotion_to_idx.get(target_emotion, 3)
        
        return 3  # Default to neutral
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Get item with caching and optional augmentation."""
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Create cache filename
        cache_filename = Path(file_path).stem + '.npy'
        cache_path = self.cache_dir / cache_filename
        
        # Try to load from cache
        if cache_path.exists() and not self.use_augmentation:
            features = np.load(cache_path)
        else:
            # Load and preprocess audio
            audio, sr, quality = preprocess_audio(file_path)
            
            # Apply augmentation if training
            if self.use_augmentation:
                audio = augment_audio(audio, sr)
            
            # Extract features
            _, features = extract_all_features(audio, sr)
            
            # Cache if not using augmentation
            if not self.use_augmentation:
                np.save(cache_path, features)
        
        return torch.FloatTensor(features), label


# ============================================
# TRAINING FUNCTION
# ============================================

def train_improved_model(
    train_paths,
    val_paths,
    model_save_dir,
    epochs=100,
    batch_size=16,
    lr=0.0005,
    use_augmentation=True
):
    """
    Train voice emotion model with improvements.
    
    Args:
        train_paths: Training file paths
        val_paths: Validation file paths
        model_save_dir: Directory to save models
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        use_augmentation: Whether to use data augmentation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    cache_dir = project_root / 'data' / 'voice_emotion' / 'feature_cache'
    train_dataset = VoiceEmotionDataset(train_paths, cache_dir, use_augmentation=use_augmentation)
    val_dataset = VoiceEmotionDataset(val_paths, cache_dir, use_augmentation=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Calculate class weights
    label_counts = Counter(train_dataset.labels)
    print(f"\nTraining label distribution: {label_counts}")
    
    # Weights inversely proportional to frequency
    total = sum(label_counts.values())
    class_weights = []
    emotion_names = ['angry', 'fear', 'happy', 'neutral', 'sad']
    
    for i in range(5):
        count = label_counts.get(i, 1)
        weight = total / (5 * count)
        class_weights.append(weight)
        print(f"  {emotion_names[i]}: {count} samples, weight: {weight:.2f}")
    
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Create model
    model = VoiceEmotionModel(input_dim=48, num_classes=5)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10
    
    model_save_path = Path(model_save_dir)
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("STARTING IMPROVED TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Data augmentation: {use_augmentation}")
    print(f"Class weights: {class_weights.cpu().numpy()}")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Per-class accuracy
        from sklearn.metrics import classification_report
        report = classification_report(
            val_labels_list, val_preds,
            target_names=emotion_names,
            zero_division=0
        )
        print("\nValidation Classification Report:")
        print(report)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save(model.state_dict(), model_save_path / 'emotion_model_best_improved.pth')
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_save_path / 'emotion_model_best_improved.pth'}")
    print(f"{'='*70}\n")
    
    return best_val_acc


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    # Load data splits
    splits_path = project_root / 'data' / 'voice_emotion' / 'dataset_splits.json'
    
    with open(splits_path) as f:
        splits = json.load(f)
    
    train_paths = splits['train']
    val_paths = splits['val']
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Train improved model
    model_save_dir = project_root / 'models' / 'voice_emotion'
    
    best_acc = train_improved_model(
        train_paths,
        val_paths,
        model_save_dir,
        epochs=100,
        batch_size=16,
        lr=0.0005,
        use_augmentation=False  # Disabled due to librosa/numba issues
    )
    
    print(f"\n🎉 Final validation accuracy: {best_acc:.2f}%")
    print(f"\nNext steps:")
    print(f"  1. Test improved model: python diagnostics/test_happy_audio.py")
    print(f"  2. Update microphone detection to use new model")
    print(f"  3. If accuracy still low, try transfer learning or more data")
