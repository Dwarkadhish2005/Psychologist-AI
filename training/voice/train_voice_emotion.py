"""
Training Script for Voice Emotion Recognition
==============================================
Phase 2: Train voice emotion and stress detection models

Architecture:
  - Emotion Model: 5 classes (angry, fear, happy, neutral, sad)
  - Stress Model: 3 levels (low, medium, high)
  - GPU-accelerated feature extraction and training

Author: Psychologist AI Team
Phase: 2 (Voice Emotion Recognition)
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.voice.audio_preprocessing import preprocess_audio
from training.voice.feature_extraction import extract_all_features, extract_stress_features
from training.voice.voice_emotion_model import VoiceEmotionModel, StressDetector, VoiceEmotionSystem
from training.voice.dataset_utils import TARGET_EMOTIONS, STRESS_LEVELS


# ============================================
# CONFIGURATION
# ============================================

class Config:
    # Paths
    DATA_ROOT = Path(r"C:\Dwarka\Machiene Learning\Psycologist AI\data\voice_emotion")
    MODEL_ROOT = Path(r"C:\Dwarka\Machiene Learning\Psycologist AI\models\voice_emotion")
    SPLITS_FILE = DATA_ROOT / "dataset_splits.json"
    METADATA_FILE = DATA_ROOT / "dataset_metadata.json"
    CACHE_DIR = DATA_ROOT / "feature_cache"
    
    # Model parameters
    FEATURE_DIM = 48
    NUM_EMOTIONS = 5
    NUM_STRESS_LEVELS = 3
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Feature extraction
    SAMPLE_RATE = 16000
    USE_CACHE = True  # Cache features to speed up training


# ============================================
# DATASET CLASS
# ============================================

class VoiceEmotionDataset(Dataset):
    """
    Voice emotion dataset with on-the-fly or cached feature extraction.
    """
    
    def __init__(self, file_paths, emotions, stress_levels, use_cache=True, cache_dir=None):
        """
        Initialize dataset.
        
        Args:
            file_paths: List of audio file paths
            emotions: List of emotion labels
            stress_levels: List of stress level labels
            use_cache: Whether to use cached features
            cache_dir: Directory for feature cache
        """
        self.file_paths = file_paths
        self.emotions = emotions
        self.stress_levels = stress_levels
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Label to index mapping
        self.emotion_to_idx = {e: i for i, e in enumerate(TARGET_EMOTIONS)}
        self.stress_to_idx = {s: i for i, s in enumerate(STRESS_LEVELS)}
    
    def __len__(self):
        return len(self.file_paths)
    
    def _get_cache_path(self, file_path):
        """Get cache file path for features."""
        if not self.cache_dir:
            return None
        file_hash = str(hash(file_path))
        return self.cache_dir / f"{file_hash}.pkl"
    
    def _extract_features(self, file_path):
        """Extract features from audio file."""
        # Preprocess audio
        audio, sr, quality = preprocess_audio(file_path, target_sr=Config.SAMPLE_RATE)
        
        if not quality['is_valid']:
            # Return zero features if audio is invalid
            return np.zeros(Config.FEATURE_DIM), np.zeros(4)
        
        # Extract all features
        _, feature_vector = extract_all_features(audio, sr)
        
        # Extract stress features
        stress_features_dict = extract_stress_features(audio, sr)
        stress_feature_vector = np.array([
            stress_features_dict['jitter'],
            stress_features_dict['shimmer'],
            stress_features_dict['spectral_flatness'],
            stress_features_dict['pitch_variance']
        ])
        
        return feature_vector, stress_feature_vector
    
    def __getitem__(self, idx):
        """Get a single sample."""
        file_path = self.file_paths[idx]
        emotion = self.emotions[idx]
        stress_level = self.stress_levels[idx]
        
        # Try to load from cache
        cache_path = self._get_cache_path(file_path)
        if self.use_cache and cache_path and cache_path.exists():
            with open(cache_path, 'rb') as f:
                features, stress_features = pickle.load(f)
        else:
            # Extract features
            features, stress_features = self._extract_features(file_path)
            
            # Save to cache
            if self.use_cache and cache_path:
                with open(cache_path, 'wb') as f:
                    pickle.dump((features, stress_features), f)
        
        # Convert to tensors
        features = torch.FloatTensor(features)
        stress_features = torch.FloatTensor(stress_features)
        emotion_label = torch.LongTensor([self.emotion_to_idx[emotion]])[0]
        stress_label = torch.LongTensor([self.stress_to_idx[stress_level]])[0]
        
        return features, stress_features, emotion_label, stress_label


# ============================================
# TRAINING FUNCTIONS
# ============================================

def train_epoch(emotion_model, stress_model, dataloader, emotion_criterion, 
                stress_criterion, emotion_optimizer, stress_optimizer, device):
    """Train for one epoch."""
    emotion_model.train()
    stress_model.train()
    
    total_emotion_loss = 0
    total_stress_loss = 0
    emotion_correct = 0
    stress_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for features, stress_features, emotion_labels, stress_labels in pbar:
        features = features.to(device)
        stress_features = stress_features.to(device)
        emotion_labels = emotion_labels.to(device)
        stress_labels = stress_labels.to(device)
        
        # Emotion model
        emotion_optimizer.zero_grad()
        emotion_outputs = emotion_model(features)
        emotion_loss = emotion_criterion(emotion_outputs, emotion_labels)
        emotion_loss.backward()
        emotion_optimizer.step()
        
        # Stress model
        stress_optimizer.zero_grad()
        stress_outputs = stress_model(stress_features)
        stress_loss = stress_criterion(stress_outputs, stress_labels)
        stress_loss.backward()
        stress_optimizer.step()
        
        # Statistics
        _, emotion_preds = torch.max(emotion_outputs, 1)
        _, stress_preds = torch.max(stress_outputs, 1)
        
        total_emotion_loss += emotion_loss.item()
        total_stress_loss += stress_loss.item()
        emotion_correct += (emotion_preds == emotion_labels).sum().item()
        stress_correct += (stress_preds == stress_labels).sum().item()
        total_samples += emotion_labels.size(0)
        
        pbar.set_postfix({
            'emotion_loss': f'{emotion_loss.item():.4f}',
            'stress_loss': f'{stress_loss.item():.4f}',
            'emotion_acc': f'{100 * emotion_correct / total_samples:.2f}%',
            'stress_acc': f'{100 * stress_correct / total_samples:.2f}%'
        })
    
    avg_emotion_loss = total_emotion_loss / len(dataloader)
    avg_stress_loss = total_stress_loss / len(dataloader)
    emotion_acc = 100 * emotion_correct / total_samples
    stress_acc = 100 * stress_correct / total_samples
    
    return avg_emotion_loss, avg_stress_loss, emotion_acc, stress_acc


def validate(emotion_model, stress_model, dataloader, emotion_criterion, 
             stress_criterion, device):
    """Validate models."""
    emotion_model.eval()
    stress_model.eval()
    
    total_emotion_loss = 0
    total_stress_loss = 0
    emotion_correct = 0
    stress_correct = 0
    total_samples = 0
    
    all_emotion_preds = []
    all_emotion_labels = []
    all_stress_preds = []
    all_stress_labels = []
    
    with torch.no_grad():
        for features, stress_features, emotion_labels, stress_labels in tqdm(dataloader, desc="Validating"):
            features = features.to(device)
            stress_features = stress_features.to(device)
            emotion_labels = emotion_labels.to(device)
            stress_labels = stress_labels.to(device)
            
            # Forward pass
            emotion_outputs = emotion_model(features)
            stress_outputs = stress_model(stress_features)
            
            # Loss
            emotion_loss = emotion_criterion(emotion_outputs, emotion_labels)
            stress_loss = stress_criterion(stress_outputs, stress_labels)
            
            # Predictions
            _, emotion_preds = torch.max(emotion_outputs, 1)
            _, stress_preds = torch.max(stress_outputs, 1)
            
            # Statistics
            total_emotion_loss += emotion_loss.item()
            total_stress_loss += stress_loss.item()
            emotion_correct += (emotion_preds == emotion_labels).sum().item()
            stress_correct += (stress_preds == stress_labels).sum().item()
            total_samples += emotion_labels.size(0)
            
            # Store for metrics
            all_emotion_preds.extend(emotion_preds.cpu().numpy())
            all_emotion_labels.extend(emotion_labels.cpu().numpy())
            all_stress_preds.extend(stress_preds.cpu().numpy())
            all_stress_labels.extend(stress_labels.cpu().numpy())
    
    avg_emotion_loss = total_emotion_loss / len(dataloader)
    avg_stress_loss = total_stress_loss / len(dataloader)
    emotion_acc = 100 * emotion_correct / total_samples
    stress_acc = 100 * stress_correct / total_samples
    
    return (avg_emotion_loss, avg_stress_loss, emotion_acc, stress_acc,
            all_emotion_preds, all_emotion_labels, all_stress_preds, all_stress_labels)


# ============================================
# MAIN TRAINING LOOP
# ============================================

def main():
    print("=" * 60)
    print("VOICE EMOTION RECOGNITION TRAINING")
    print("=" * 60)
    
    config = Config()
    
    # Device
    print(f"\n[Device] Using: {config.DEVICE}")
    if config.DEVICE.type == 'cuda':
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset splits
    print(f"\n[Data] Loading dataset splits...")
    with open(config.SPLITS_FILE, 'r') as f:
        splits = json.load(f)
    
    with open(config.METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    print(f"[Data] Train: {len(splits['train'])} samples")
    print(f"[Data] Val: {len(splits['val'])} samples")
    print(f"[Data] Test: {len(splits['test'])} samples")
    
    # Load labels (parse from file paths - they're stored in dataset_metadata)
    # For now, we'll re-parse the dataset
    print(f"\n[Data] Parsing dataset metadata...")
    
    from download_datasets import load_ravdess_dataset, load_tess_dataset
    
    ravdess_data = load_ravdess_dataset(config.DATA_ROOT / 'RAVDESS')
    tess_data = load_tess_dataset(config.DATA_ROOT / 'TESS')
    all_data = ravdess_data + tess_data
    
    # Create file_path to metadata mapping
    file_to_meta = {item['file_path']: item for item in all_data}
    
    # Create datasets
    def get_labels(file_paths):
        emotions = []
        stress_levels = []
        for fp in file_paths:
            meta = file_to_meta.get(fp)
            if meta:
                emotions.append(meta['emotion'])
                stress_levels.append(meta['stress_level'])
            else:
                # Skip if not found
                emotions.append('neutral')
                stress_levels.append('low')
        return emotions, stress_levels
    
    train_emotions, train_stress = get_labels(splits['train'])
    val_emotions, val_stress = get_labels(splits['val'])
    test_emotions, test_stress = get_labels(splits['test'])
    
    print(f"\n[Data] Creating datasets...")
    train_dataset = VoiceEmotionDataset(
        splits['train'], train_emotions, train_stress,
        use_cache=config.USE_CACHE, cache_dir=config.CACHE_DIR
    )
    val_dataset = VoiceEmotionDataset(
        splits['val'], val_emotions, val_stress,
        use_cache=config.USE_CACHE, cache_dir=config.CACHE_DIR
    )
    test_dataset = VoiceEmotionDataset(
        splits['test'], test_emotions, test_stress,
        use_cache=config.USE_CACHE, cache_dir=config.CACHE_DIR
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create models
    print(f"\n[Model] Creating models...")
    emotion_model = VoiceEmotionModel(input_dim=config.FEATURE_DIM, num_classes=config.NUM_EMOTIONS).to(config.DEVICE)
    stress_model = StressDetector(input_dim=4, num_levels=config.NUM_STRESS_LEVELS).to(config.DEVICE)
    
    print(f"[Model] Emotion model: {sum(p.numel() for p in emotion_model.parameters()):,} parameters")
    print(f"[Model] Stress model: {sum(p.numel() for p in stress_model.parameters()):,} parameters")
    
    # Loss and optimizers
    emotion_criterion = nn.CrossEntropyLoss()
    stress_criterion = nn.CrossEntropyLoss()
    
    emotion_optimizer = optim.Adam(emotion_model.parameters(), lr=config.LEARNING_RATE)
    stress_optimizer = optim.Adam(stress_model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate schedulers
    emotion_scheduler = optim.lr_scheduler.ReduceLROnPlateau(emotion_optimizer, mode='max', patience=5, factor=0.5)
    stress_scheduler = optim.lr_scheduler.ReduceLROnPlateau(stress_optimizer, mode='max', patience=5, factor=0.5)
    
    # Training loop
    print(f"\n[Train] Starting training for {config.NUM_EPOCHS} epochs...")
    
    best_emotion_acc = 0
    best_stress_acc = 0
    patience_counter = 0
    
    history = {
        'train_emotion_loss': [],
        'train_stress_loss': [],
        'train_emotion_acc': [],
        'train_stress_acc': [],
        'val_emotion_loss': [],
        'val_stress_loss': [],
        'val_emotion_acc': [],
        'val_stress_acc': []
    }
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_emotion_loss, train_stress_loss, train_emotion_acc, train_stress_acc = train_epoch(
            emotion_model, stress_model, train_loader,
            emotion_criterion, stress_criterion,
            emotion_optimizer, stress_optimizer,
            config.DEVICE
        )
        
        # Validate
        val_results = validate(
            emotion_model, stress_model, val_loader,
            emotion_criterion, stress_criterion,
            config.DEVICE
        )
        val_emotion_loss, val_stress_loss, val_emotion_acc, val_stress_acc = val_results[:4]
        
        # Update history
        history['train_emotion_loss'].append(train_emotion_loss)
        history['train_stress_loss'].append(train_stress_loss)
        history['train_emotion_acc'].append(train_emotion_acc)
        history['train_stress_acc'].append(train_stress_acc)
        history['val_emotion_loss'].append(val_emotion_loss)
        history['val_stress_loss'].append(val_stress_loss)
        history['val_emotion_acc'].append(val_emotion_acc)
        history['val_stress_acc'].append(val_stress_acc)
        
        # Print results
        print(f"\n[Results] Epoch {epoch+1}")
        print(f"  Emotion: Train Loss={train_emotion_loss:.4f}, Acc={train_emotion_acc:.2f}% | Val Loss={val_emotion_loss:.4f}, Acc={val_emotion_acc:.2f}%")
        print(f"  Stress:  Train Loss={train_stress_loss:.4f}, Acc={train_stress_acc:.2f}% | Val Loss={val_stress_loss:.4f}, Acc={val_stress_acc:.2f}%")
        
        # Learning rate scheduling
        emotion_scheduler.step(val_emotion_acc)
        stress_scheduler.step(val_stress_acc)
        
        # Save best models
        if val_emotion_acc > best_emotion_acc:
            best_emotion_acc = val_emotion_acc
            torch.save(emotion_model.state_dict(), config.MODEL_ROOT / 'emotion_model_best.pth')
            print(f"[Save] New best emotion model: {best_emotion_acc:.2f}%")
            patience_counter = 0
        
        if val_stress_acc > best_stress_acc:
            best_stress_acc = val_stress_acc
            torch.save(stress_model.state_dict(), config.MODEL_ROOT / 'stress_model_best.pth')
            print(f"[Save] New best stress model: {best_stress_acc:.2f}%")
        
        # Early stopping
        patience_counter += 1
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n[Early Stop] No improvement for {config.EARLY_STOPPING_PATIENCE} epochs")
            break
    
    # Load best models for testing
    print(f"\n{'='*60}")
    print("TESTING")
    print(f"{'='*60}")
    
    emotion_model.load_state_dict(torch.load(config.MODEL_ROOT / 'emotion_model_best.pth'))
    stress_model.load_state_dict(torch.load(config.MODEL_ROOT / 'stress_model_best.pth'))
    
    test_results = validate(
        emotion_model, stress_model, test_loader,
        emotion_criterion, stress_criterion,
        config.DEVICE
    )
    
    test_emotion_loss, test_stress_loss, test_emotion_acc, test_stress_acc, \
        emotion_preds, emotion_labels, stress_preds, stress_labels = test_results
    
    print(f"\n[Test Results]")
    print(f"  Emotion Accuracy: {test_emotion_acc:.2f}%")
    print(f"  Stress Accuracy: {test_stress_acc:.2f}%")
    
    # Classification reports
    print(f"\n[Emotion Classification Report]")
    print(classification_report(emotion_labels, emotion_preds, target_names=TARGET_EMOTIONS))
    
    print(f"\n[Stress Classification Report]")
    print(classification_report(stress_labels, stress_preds, target_names=STRESS_LEVELS))
    
    print(f"\n{'='*60}")
    print("✓ Training complete!")
    print(f"  Best Emotion Model: {best_emotion_acc:.2f}% (saved)")
    print(f"  Best Stress Model: {best_stress_acc:.2f}% (saved)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
