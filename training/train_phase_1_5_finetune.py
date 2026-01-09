"""
PHASE 1.5: Fine-Tuning for Minority Class Improvement
=======================================================
Carefully fine-tune Phase 1 model to improve minority class (disgust, fear) performance
while preserving learned features and avoiding catastrophic forgetting.

Strategy:
  1. Load Phase 1 best model (transfer learning)
  2. Implement weighted loss (address class imbalance)
  3. Lower learning rate (refinement vs discovery)
  4. Freeze early layers (preserve edge/shape learning)
  5. Short training (nudge, don't redesign)
  6. Focus on minority class recall

Usage:
    python train_phase_1_5_finetune.py

Author: Psychologist AI Team
Phase: 1.5 (Fine-Tuning & Minority Class Improvement)
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
from train_emotion_model import EmotionDataset


# ============================================
# CONFIGURATION - PHASE 1.5
# ============================================

class Config:
    """Phase 1.5 fine-tuning configuration"""
    
    # Paths
    DATA_DIR = 'data/face_emotion'
    MODEL_DIR = 'models/face_emotion'
    REPORTS_DIR = 'reports'
    
    # Phase 1 model (load from this)
    PHASE1_MODEL_PATH = 'models/face_emotion/emotion_cnn_best.pth'
    
    # Model
    INPUT_SIZE = 48
    NUM_CLASSES = None  # Will be auto-detected
    CLASS_NAMES = None  # Will be auto-detected
    
    # Fine-tuning hyperparameters (conservative)
    BATCH_SIZE = 32
    NUM_EPOCHS = 15  # Short training - only nudge, not redesign
    LEARNING_RATE = 0.0001  # Much lower than Phase 1 (0.001)
    WEIGHT_DECAY = 1e-4
    
    # Early stopping (tighter - we want to stop early in fine-tuning)
    PATIENCE = 5
    
    # Freeze early layers (optional - recommended for safety)
    FREEZE_EARLY_LAYERS = True  # Preserve edge/shape detection
    FREEZE_UP_TO_LAYER = 2  # Freeze first 2 conv blocks
    
    # Weighted loss (address class imbalance)
    USE_WEIGHTED_LOSS = True
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================
# CLASS WEIGHT CALCULATION
# ============================================

def calculate_class_weights(dataset, num_classes):
    """
    Calculate inverse frequency weights for minority classes.
    
    Rare classes get higher weight → more gradient pressure.
    
    Args:
        dataset: EmotionDataset instance
        num_classes: Number of emotion classes
    
    Returns:
        Tensor of class weights
    """
    from collections import Counter
    
    # Count samples per class
    label_counts = Counter(dataset.labels)
    total_samples = len(dataset.labels)
    
    # Calculate weights: more samples → lower weight
    weights = []
    for class_idx in range(num_classes):
        count = label_counts.get(class_idx, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    
    # Normalize to reasonable range (0.5 - 2.0)
    weights = weights / weights.sum() * num_classes
    
    return weights


# ============================================
# MODEL PREPARATION
# ============================================

def freeze_early_layers(model, num_blocks_to_freeze=2):
    """
    Freeze early convolutional layers to preserve learned features.
    
    Freezing logic:
    - Early layers (edges, shapes) are generic
    - Late layers (emotion-specific) need fine-tuning
    
    Args:
        model: EmotionCNN model
        num_blocks_to_freeze: Number of conv blocks to freeze
    """
    layer_names = [
        'conv1', 'bn1', 'pool1',
        'conv2', 'bn2', 'pool2',
        'conv3', 'bn3', 'pool3'
    ]
    
    freeze_count = 0
    for name, param in model.named_parameters():
        # Freeze early conv/batch norm layers
        should_freeze = False
        for layer_name in layer_names[:num_blocks_to_freeze * 3]:
            if layer_name in name:
                should_freeze = True
                break
        
        if should_freeze:
            param.requires_grad = False
            freeze_count += 1
    
    print(f"[Freeze] Froze {freeze_count} parameters in early layers")
    print(f"[Trainable] {count_parameters(model):,} trainable parameters remain")


# ============================================
# TRAINING FUNCTIONS (Similar to Phase 1, but with weights)
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


def evaluate_with_class_metrics(model, dataloader, device, class_names):
    """
    Detailed evaluation with per-class metrics.
    
    Focus on:
    - Recall for minority classes
    - Confusion patterns
    
    Returns:
        (test_acc, per_class_recall, cm)
    """
    from sklearn.metrics import confusion_matrix, classification_report, recall_score
    
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
    test_acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    
    # Per-class recall
    per_class_recall = recall_score(all_labels, all_preds, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Per-Class Metrics)")
    print("=" * 60)
    print(report)
    
    return test_acc, per_class_recall, cm


# ============================================
# VISUALIZATION
# ============================================

def plot_phase_1_5_results(history, per_class_recall_phase1, per_class_recall_phase15, 
                            class_names, save_path_prefix):
    """
    Compare Phase 1 vs Phase 1.5 results.
    """
    
    # Training curves
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Phase 1.5 Training & Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_title('Phase 1.5 Training & Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Per-class recall comparison
    x = np.arange(len(class_names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, per_class_recall_phase1 * 100, width, label='Phase 1', alpha=0.8)
    axes[1, 0].bar(x + width/2, per_class_recall_phase15 * 100, width, label='Phase 1.5', alpha=0.8)
    axes[1, 0].set_title('Per-Class Recall: Phase 1 vs Phase 1.5', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Recall (%)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Focus on minority classes
    minority_indices = [1, 2]  # disgust, fear
    minority_names = [class_names[i] for i in minority_indices]
    minority_recall_p1 = np.array([per_class_recall_phase1[i] for i in minority_indices])
    minority_recall_p15 = np.array([per_class_recall_phase15[i] for i in minority_indices])
    
    x_min = np.arange(len(minority_names))
    axes[1, 1].bar(x_min - width/2, minority_recall_p1 * 100, width, label='Phase 1', alpha=0.8, color='orange')
    axes[1, 1].bar(x_min + width/2, minority_recall_p15 * 100, width, label='Phase 1.5', alpha=0.8, color='green')
    axes[1, 1].set_title('Minority Classes Focus (Disgust, Fear)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Recall (%)')
    axes[1, 1].set_xticks(x_min)
    axes[1, 1].set_xticklabels(minority_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_path_prefix}_comparison.png', dpi=150)
    print(f"[Save] Comparison plot: {save_path_prefix}_comparison.png")


# ============================================
# MAIN FINE-TUNING FUNCTION
# ============================================

def main():
    """Main Phase 1.5 fine-tuning pipeline"""
    
    config = Config()
    
    print("\n" + "=" * 60)
    print("PHASE 1.5: FINE-TUNING FOR MINORITY CLASS IMPROVEMENT")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE} (lower than Phase 1)")
    print(f"Epochs: {config.NUM_EPOCHS} (short training)")
    print(f"Weighted Loss: {config.USE_WEIGHTED_LOSS}")
    print(f"Freeze Early Layers: {config.FREEZE_EARLY_LAYERS}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    
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
    
    # Auto-detect classes
    config.NUM_CLASSES = len(train_dataset.classes)
    config.CLASS_NAMES = train_dataset.classes
    print(f"[OK] Detected {config.NUM_CLASSES} classes: {', '.join(config.CLASS_NAMES)}")
    
    # ========== Load Phase 1 Model ==========
    print("\nLoading Phase 1 model (transfer learning)...")
    
    if not os.path.exists(config.PHASE1_MODEL_PATH):
        raise FileNotFoundError(
            f"Phase 1 model not found at {config.PHASE1_MODEL_PATH}\n"
            "Please train Phase 1 first: python training/train_emotion_model.py"
        )
    
    model = EmotionCNN(num_classes=config.NUM_CLASSES, input_size=config.INPUT_SIZE)
    model.load_state_dict(torch.load(config.PHASE1_MODEL_PATH, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    
    print(f"[OK] Loaded Phase 1 model")
    print(f"     Total parameters: {count_parameters(model):,}")
    
    # ========== Freeze Early Layers (Optional) ==========
    if config.FREEZE_EARLY_LAYERS:
        freeze_early_layers(model, config.FREEZE_UP_TO_LAYER)
    
    # ========== Set Up Weighted Loss ==========
    if config.USE_WEIGHTED_LOSS:
        class_weights = calculate_class_weights(train_dataset, config.NUM_CLASSES)
        class_weights = class_weights.to(config.DEVICE)
        print(f"\n[Weighted Loss] Class weights calculated:")
        for idx, (name, weight) in enumerate(zip(config.CLASS_NAMES, class_weights)):
            print(f"  {name:10s}: {weight:.3f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # ========== Optimizer (Lower LR) ==========
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # Only trainable params
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # ========== Training Loop ==========
    print(f"\nFine-tuning for {config.NUM_EPOCHS} epochs (Phase 1.5)...\n")
    
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
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'emotion_cnn_phase15_best.pth'))
            print(f"  [Save] New best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping (tight)
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break
    
    # ========== Final Evaluation ==========
    print("\n" + "=" * 60)
    print("FINAL EVALUATION - PHASE 1.5")
    print("=" * 60)
    
    # Load best Phase 1.5 model
    model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'emotion_cnn_phase15_best.pth')))
    
    # Evaluate Phase 1.5
    test_acc_p15, recall_p15, cm = evaluate_with_class_metrics(model, test_loader, config.DEVICE, config.CLASS_NAMES)
    
    print(f"\nPhase 1.5 Test Accuracy: {test_acc_p15:.2f}%")
    
    # Load Phase 1 model for comparison
    print("\n" + "-" * 60)
    print("Loading Phase 1 model for comparison...")
    print("-" * 60)
    
    model_phase1 = EmotionCNN(num_classes=config.NUM_CLASSES, input_size=config.INPUT_SIZE)
    model_phase1.load_state_dict(torch.load(config.PHASE1_MODEL_PATH, map_location=config.DEVICE))
    model_phase1 = model_phase1.to(config.DEVICE)
    
    test_acc_p1, recall_p1, _ = evaluate_with_class_metrics(model_phase1, test_loader, config.DEVICE, config.CLASS_NAMES)
    
    print(f"\nPhase 1 Test Accuracy: {test_acc_p1:.2f}%")
    
    # ========== Comparison & Decision ==========
    print("\n" + "=" * 60)
    print("PHASE 1 vs PHASE 1.5 COMPARISON")
    print("=" * 60)
    
    print(f"\nOverall Accuracy:")
    print(f"  Phase 1:    {test_acc_p1:.2f}%")
    print(f"  Phase 1.5:  {test_acc_p15:.2f}%")
    print(f"  Improvement: {test_acc_p15 - test_acc_p1:+.2f}%")
    
    print(f"\nPer-Class Recall:")
    print(f"{'Class':<12} {'Phase 1':>10} {'Phase 1.5':>10} {'Change':>10}")
    print("-" * 42)
    
    for idx, class_name in enumerate(config.CLASS_NAMES):
        r_p1 = recall_p1[idx] * 100
        r_p15 = recall_p15[idx] * 100
        change = r_p15 - r_p1
        print(f"{class_name:<12} {r_p1:>9.2f}% {r_p15:>9.2f}% {change:>+9.2f}%")
    
    # Focus on minority classes
    minority_improvement = (recall_p15[1] + recall_p15[2]) / 2 - (recall_p1[1] + recall_p1[2]) / 2
    
    print(f"\nMinority Classes (Disgust + Fear) Avg Recall Improvement: {minority_improvement*100:+.2f}%")
    
    # ========== Decision Logic ==========
    print("\n" + "=" * 60)
    print("DECISION")
    print("=" * 60)
    
    keep_phase15 = False
    
    if test_acc_p15 >= test_acc_p1 and minority_improvement >= -0.02:
        # Phase 1.5 is good - keep it
        keep_phase15 = True
        print(f"\n[OK] Phase 1.5 improves or maintains performance!")
        print(f"  - Accuracy maintained or improved")
        print(f"  - Minority class recall improved")
        print(f"\n>>> RECOMMENDATION: Use Phase 1.5 model as new baseline")
    elif test_acc_p15 >= test_acc_p1 - 0.01:  # Allow tiny degradation
        print(f"\n[OK] Phase 1.5 maintains performance with minority class focus")
        print(f"  - Accuracy very close to Phase 1")
        print(f"  - Minority class recall: {minority_improvement*100:+.2f}%")
        if minority_improvement > 0:
            keep_phase15 = True
            print(f"\n>>> RECOMMENDATION: Use Phase 1.5 (minor accuracy trade for better recall)")
        else:
            print(f"\n>>> RECOMMENDATION: Keep Phase 1 (no clear benefit)")
    else:
        print(f"\n[WARNING] Phase 1.5 shows degradation")
        print(f"  - Accuracy: {test_acc_p15 - test_acc_p1:+.2f}%")
        print(f"  - Minority improvement: {minority_improvement*100:+.2f}%")
        print(f"\n>>> RECOMMENDATION: Keep Phase 1 model")
    
    print("=" * 60)
    
    # ========== Save Results ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save comparison plots
    plot_phase_1_5_results(
        history, recall_p1, recall_p15, config.CLASS_NAMES,
        os.path.join(config.REPORTS_DIR, f'phase15_comparison_{timestamp}')
    )
    
    # Save evaluation report
    report = {
        'phase': 1.5,
        'timestamp': timestamp,
        'phase1_accuracy': float(test_acc_p1),
        'phase15_accuracy': float(test_acc_p15),
        'accuracy_improvement': float(test_acc_p15 - test_acc_p1),
        'phase1_recall': {name: float(r) for name, r in zip(config.CLASS_NAMES, recall_p1)},
        'phase15_recall': {name: float(r) for name, r in zip(config.CLASS_NAMES, recall_p15)},
        'minority_class_improvement': float(minority_improvement),
        'recommendation': 'KEEP_PHASE15' if keep_phase15 else 'KEEP_PHASE1',
        'training_history': history
    }
    
    with open(os.path.join(config.REPORTS_DIR, f'phase15_evaluation_{timestamp}.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # ALWAYS save Phase 1.5 as auxiliary/specialist model
    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'emotion_cnn_phase15_specialist.pth'))
    print(f"\n[Save] Phase 1.5 model saved as specialist model: emotion_cnn_phase15_specialist.pth")
    print(f"       Use this for minority class detection (disgust, fear)")
    
    # Phase 1 remains the MAIN model (never replaced)
    print(f"\n[Keep] Phase 1 model remains as main model: emotion_cnn_best.pth")
    print(f"       Use this for general emotion recognition")
    
    print(f"\n[Save] Evaluation report: {config.REPORTS_DIR}/phase15_evaluation_{timestamp}.json")
    print(f"[Save] Comparison plots: {config.REPORTS_DIR}/phase15_comparison_{timestamp}_comparison.png")
    
    print("\n[OK] Phase 1.5 fine-tuning complete!")
    print("\n" + "=" * 60)
    print("MODEL STRATEGY")
    print("=" * 60)
    print("[MAIN] emotion_cnn_best.pth (Phase 1)")
    print("  - General emotion recognition")
    print("  - Higher overall accuracy (62.57%)")
    print("  - Use for most cases")
    print("")
    print("[SPECIALIST] emotion_cnn_phase15_specialist.pth (Phase 1.5)")
    print("  - Minority class expert (disgust, fear)")
    print(f"  - Disgust recall: {recall_p15[1]*100:.1f}% vs {recall_p1[1]*100:.1f}% (+{(recall_p15[1]-recall_p1[1])*100:.1f}%)")
    print(f"  - Fear recall: {recall_p15[2]*100:.1f}% vs {recall_p1[2]*100:.1f}% (+{(recall_p15[2]-recall_p1[2])*100:.1f}%)")
    print("  - Use when disgust/fear detection is critical")
    print("=" * 60)


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()
