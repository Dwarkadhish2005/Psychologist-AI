"""
BALANCED Voice Emotion Training Script
========================================
Fixes overfitting and angry bias while maintaining happy detection:
1. Stronger regularization (dropout 0.5, weight decay 1e-4)
2. Balanced class weights (reduce angry bias)
3. Smaller learning rate with cosine annealing
4. Gradient clipping
5. Label smoothing

Goal: Maintain 100% happy detection while improving other classes
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

from voice_emotion_model import VoiceEmotionModel
from feature_extraction import extract_all_features
from audio_preprocessing import preprocess_audio


# ============================================
# DATASET CLASS
# ============================================

class VoiceEmotionDataset(Dataset):
    """Dataset for voice emotion recognition with caching."""
    
    def __init__(self, file_paths, cache_dir):
        self.file_paths = file_paths
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load emotion labels
        from dataset_utils import TARGET_EMOTION_TO_IDX
        self.emotion_to_idx = TARGET_EMOTION_TO_IDX
        
        # Extract labels
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
                return self.emotion_to_idx.get(target_emotion, 3)
        
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
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Create cache filename
        cache_filename = Path(file_path).stem + '.npy'
        cache_path = self.cache_dir / cache_filename
        
        # Try to load from cache
        if cache_path.exists():
            features = np.load(cache_path)
        else:
            # Load and preprocess audio
            audio, sr, quality = preprocess_audio(file_path)
            
            # Extract features
            _, features = extract_all_features(audio, sr)
            
            # Cache features
            np.save(cache_path, features)
        
        return torch.FloatTensor(features), label


# ============================================
# LABEL SMOOTHING LOSS
# ============================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing to prevent overconfidence."""
    
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_probs = torch.log_softmax(pred, dim=-1)
        
        # Apply label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Apply class weights if provided
        if self.weight is not None:
            weight_mask = self.weight[target]
            loss = -(true_dist * log_probs).sum(dim=-1) * weight_mask
        else:
            loss = -(true_dist * log_probs).sum(dim=-1)
        
        return loss.mean()


# ============================================
# TRAINING FUNCTION
# ============================================

def train_balanced_model(
    train_paths,
    val_paths,
    model_save_dir,
    epochs=80,
    batch_size=32,
    lr=0.0003
):
    """Train balanced voice emotion model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    cache_dir = project_root / 'data' / 'voice_emotion' / 'feature_cache'
    train_dataset = VoiceEmotionDataset(train_paths, cache_dir)
    val_dataset = VoiceEmotionDataset(val_paths, cache_dir)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Calculate balanced class weights (reduce angry bias)
    label_counts = Counter(train_dataset.labels)
    print(f"\nTraining label distribution: {label_counts}")
    
    # Manual weights to balance angry overprediction
    # angry=0, fear=1, happy=2, neutral=3, sad=4
    class_weights = torch.FloatTensor([
        0.7,  # angry - reduce (was dominating)
        1.5,  # fear - increase (low recall)
        1.3,  # happy - increase slightly (maintain good detection)
        1.2,  # neutral - increase (low recall)
        1.5   # sad - increase (very low recall)
    ]).to(device)
    
    print("\nBalanced Class Weights:")
    emotion_names = ['angry', 'fear', 'happy', 'neutral', 'sad']
    for i, (emotion, weight) in enumerate(zip(emotion_names, class_weights)):
        print(f"  {emotion}: weight={weight:.2f}")
    
    # Create model with HIGHER dropout
    model = VoiceEmotionModel(
        input_dim=48,
        num_classes=5,
        hidden_dims=[256, 128, 64],
        dropout=0.5  # Increased from 0.3
    )
    model = model.to(device)
    
    # Loss with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights)
    
    # Optimizer with stronger weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )
    
    # Training loop
    best_val_acc = 0.0
    best_happy_recall = 0.0
    patience_counter = 0
    patience = 15
    
    model_save_path = Path(model_save_dir)
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("STARTING BALANCED TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Dropout: 0.5 (high regularization)")
    print(f"Weight decay: 1e-4")
    print(f"Label smoothing: 0.1")
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
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
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
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate per-class metrics
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(
            val_labels_list, val_preds,
            target_names=emotion_names,
            output_dict=True,
            zero_division=0
        )
        
        happy_recall = report['happy']['recall']
        happy_precision = report['happy']['precision']
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Happy: Precision={happy_precision:.2%}, Recall={happy_recall:.2%}")
        
        # Print full classification report every 5 epochs
        if (epoch + 1) % 5 == 0:
            report_str = classification_report(
                val_labels_list, val_preds,
                target_names=emotion_names,
                zero_division=0
            )
            print("\nValidation Classification Report:")
            print(report_str)
        
        # Save best model (prioritize happy recall + overall accuracy)
        combined_metric = val_acc + (happy_recall * 50)  # Weight happy recall heavily
        best_combined = best_val_acc + (best_happy_recall * 50)
        
        if combined_metric > best_combined:
            best_val_acc = val_acc
            best_happy_recall = happy_recall
            patience_counter = 0
            
            torch.save(model.state_dict(), model_save_path / 'emotion_model_best_balanced.pth')
            print(f"✅ New best model! Val Acc: {val_acc:.2f}%, Happy Recall: {happy_recall:.2%}")
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
    print(f"Best Happy Recall: {best_happy_recall:.2%}")
    print(f"Model saved to: {model_save_path / 'emotion_model_best_balanced.pth'}")
    print(f"{'='*70}\n")
    
    return best_val_acc, best_happy_recall


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
    
    # Train balanced model
    model_save_dir = project_root / 'models' / 'voice_emotion'
    
    best_acc, best_happy = train_balanced_model(
        train_paths,
        val_paths,
        model_save_dir,
        epochs=80,
        batch_size=32,
        lr=0.0003
    )
    
    print(f"\n🎉 Final results:")
    print(f"  Validation accuracy: {best_acc:.2f}%")
    print(f"  Happy recall: {best_happy:.2%}")
    print(f"\nNext steps:")
    print(f"  1. Test: python diagnostics/test_happy_audio.py")
    print(f"  2. If happy recall > 80% and val acc > 45%, use this model")
    print(f"  3. Test with microphone: python inference/microphone_emotion_detection.py")
