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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from collections import Counter
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

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
    
    def __init__(self, file_paths, cache_dir, scaler=None, training=False):
        self.file_paths = file_paths
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = scaler
        self.training = training  # enables augmentation during training only

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

        # Apply feature normalization if scaler is fitted
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()

        # Gaussian noise augmentation — training only.
        # Adds random perturbations (std=0.1 in normalized space) so the model
        # sees slightly different feature values each epoch, reducing overfitting on
        # only 336 samples per class.
        if self.training:
            features = features + np.random.normal(0, 0.1, features.shape).astype(np.float32)

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
    
    # Create datasets (no scaler yet — scaler fitted on training features below)
    cache_dir = project_root / 'data' / 'voice_emotion' / 'feature_cache'
    train_dataset = VoiceEmotionDataset(train_paths, cache_dir, training=True)
    val_dataset = VoiceEmotionDataset(val_paths, cache_dir, training=False)

    # Fit StandardScaler on all training features so train and inference use
    # the same feature distribution. Save to disk so test scripts can load it.
    print("\nFitting StandardScaler on training features...")
    all_train_feats = np.vstack([
        train_dataset[i][0].numpy() for i in range(len(train_dataset))
    ])
    scaler = StandardScaler()
    scaler.fit(all_train_feats)
    joblib.dump(scaler, Path(model_save_dir) / 'feature_scaler.pkl')
    print(f"  Scaler fitted on {len(all_train_feats)} samples — saved to feature_scaler.pkl")

    # Apply scaler to both datasets
    train_dataset.scaler = scaler
    val_dataset.scaler = scaler

    # Compute class weights directly from training label distribution
    emotion_names = ['angry', 'fear', 'happy', 'neutral', 'sad']
    label_counts = Counter(train_dataset.labels)
    print(f"\nTraining label distribution: {label_counts}")

    # Create data loaders
    # WeightedRandomSampler ensures each mini-batch has equal class representation.
    # This is more effective than loss weighting alone because the model sees each
    # class equally often per update rather than down-weighting majority classes.
    sample_weights = [1.0 / label_counts[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    label_array = np.array(train_dataset.labels)
    unique_classes = np.unique(label_array)
    computed_weights = compute_class_weight('balanced', classes=unique_classes, y=label_array)
    class_weight_array = np.ones(5, dtype=np.float32)
    for i, cls in enumerate(unique_classes):
        class_weight_array[cls] = computed_weights[i]
    class_weights = torch.FloatTensor(class_weight_array).to(device)

    print("\nAuto-Computed Class Weights (balanced):")
    for i, (emotion, weight) in enumerate(zip(emotion_names, class_weights)):
        print(f"  {emotion}: weight={weight:.3f}  (count: {label_counts.get(i, 0)})")
    
    # Create model — [256,128,64] gives sufficient capacity to separate 5 classes
    # in 48D feature space. Overfitting is controlled by dropout=0.5, weight_decay=5e-4,
    # and noise augmentation rather than shrinking capacity.
    model = VoiceEmotionModel(
        input_dim=48,
        num_classes=5,
        hidden_dims=[256, 128, 64],
        dropout=0.5
    )
    model = model.to(device)
    
    # Loss with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights)
    
    # Optimizer — lr=0.0003 converges to the best solution quickly (peaked at epoch 4
    # in previous run). Higher weight_decay=5e-4 (vs 1e-4 before) adds stronger L2
    # regularization to slow post-convergence overfitting.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    
    # ReduceLROnPlateau: only decays when val macro F1 stops improving.
    # Replaces CosineAnnealingWarmRestarts which would abruptly reset LR to
    # the initial value mid-training and destabilize the converged model.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=8, factor=0.5, min_lr=1e-5, verbose=False
    )
    
    # Training loop
    best_val_acc = 0.0
    best_macro_f1 = 0.0
    patience_counter = 0
    patience = 20
    
    model_save_path = Path(model_save_dir)
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("STARTING BALANCED TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Model dims: [256, 128, 64]")
    print(f"Dropout: 0.5 (high regularization)")
    print(f"Weight decay: 5e-4 (stronger L2)")
    print(f"Label smoothing: 0.1")
    print(f"Feature augmentation: Gaussian noise std=0.1 (training only)")
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

        # Calculate per-class metrics
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(
            val_labels_list, val_preds,
            target_names=emotion_names,
            output_dict=True,
            zero_division=0
        )

        macro_f1 = report['macro avg']['f1-score']

        # Learning rate scheduling — step on macro F1 (higher = better)
        scheduler.step(macro_f1)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Macro F1: {macro_f1:.4f}")

        # Print full classification report every 5 epochs
        if (epoch + 1) % 5 == 0:
            report_str = classification_report(
                val_labels_list, val_preds,
                target_names=emotion_names,
                zero_division=0
            )
            print("\nValidation Classification Report:")
            print(report_str)

        # Save best model using macro F1 — unbiased across all classes
        if macro_f1 > best_macro_f1:
            best_val_acc = val_acc
            best_macro_f1 = macro_f1
            patience_counter = 0

            torch.save(model.state_dict(), model_save_path / 'emotion_model_best_balanced.pth')
            print(f"✅ New best model! Val Acc: {val_acc:.2f}%, Macro F1: {macro_f1:.4f}")
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
    print(f"Best Macro F1: {best_macro_f1:.4f}")
    print(f"Model saved to: {model_save_path / 'emotion_model_best_balanced.pth'}")
    print(f"{'='*70}\n")

    return best_val_acc, best_macro_f1


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
    
    best_acc, best_f1 = train_balanced_model(
        train_paths,
        val_paths,
        model_save_dir,
        epochs=80,
        batch_size=32,
        lr=0.0003
    )

    print(f"\n🎉 Final results:")
    print(f"  Validation accuracy: {best_acc:.2f}%")
    print(f"  Macro F1: {best_f1:.4f}")
    print(f"\nNext steps:")
    print(f"  1. Test: python diagnostics/test_happy_audio.py")
    print(f"  2. If happy recall > 80% and val acc > 45%, use this model")
    print(f"  3. Test with microphone: python inference/microphone_emotion_detection.py")
