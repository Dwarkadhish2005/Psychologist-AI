"""
Dataset Utilities for Voice Emotion Recognition
================================================
Phase 2: Label mapping, stress derivation, speaker-independent splits

Critical Concepts:
  - Map dataset labels to 5 target classes
  - Derive stress levels from emotion (rule-based initially)
  - Speaker-independent splits (no speaker leakage)
  - Data balancing (equal samples per emotion)

Author: Psychologist AI Team
Phase: 2 (Voice Emotion Recognition)
"""

import numpy as np
from collections import defaultdict, Counter


# ============================================
# TARGET CLASSES
# ============================================

TARGET_EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad']
TARGET_EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(TARGET_EMOTIONS)}
TARGET_IDX_TO_EMOTION = {idx: emotion for emotion, idx in TARGET_EMOTION_TO_IDX.items()}

STRESS_LEVELS = ['low', 'medium', 'high']
STRESS_LEVEL_TO_IDX = {level: idx for idx, level in enumerate(STRESS_LEVELS)}
STRESS_IDX_TO_LEVEL = {idx: level for level, idx in STRESS_LEVEL_TO_IDX.items()}


# ============================================
# LABEL MAPPING
# ============================================

def map_ravdess_label(ravdess_label):
    """
    Map RAVDESS emotion label to target label.
    
    RAVDESS emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
    
    Mapping:
      - angry → angry
      - fearful → fear
      - happy → happy
      - sad → sad
      - neutral → neutral
      - calm → neutral (low-stress variant)
      - disgust → angry (closest match)
      - surprised → happy (positive arousal)
    
    Args:
        ravdess_label: Original RAVDESS label
    
    Returns:
        target_label: Mapped target label
        note: Mapping note (if special case)
    """
    label_lower = ravdess_label.lower().strip()
    
    mapping = {
        'angry': 'angry',
        'fearful': 'fear',
        'fear': 'fear',
        'happy': 'happy',
        'sad': 'sad',
        'neutral': 'neutral',
        'calm': 'neutral',  # Low-stress neutral
        'disgust': 'angry',  # Closest match
        'surprised': 'happy',  # Positive arousal
        'surprise': 'happy'
    }
    
    if label_lower in mapping:
        target = mapping[label_lower]
        note = "calm→neutral" if label_lower == 'calm' else None
        return target, note
    else:
        # Unknown label - skip or default to neutral
        return None, f"unknown:{ravdess_label}"


def map_tess_label(tess_label):
    """
    Map TESS emotion label to target label.
    
    TESS emotions: angry, disgust, fear, happy, sad, neutral, pleasant_surprised
    
    Mapping (mostly direct):
      - angry → angry
      - fear → fear
      - happy → happy
      - sad → sad
      - neutral → neutral
      - disgust → angry
      - pleasant_surprised → happy
    
    Args:
        tess_label: Original TESS label
    
    Returns:
        target_label: Mapped target label
        note: Mapping note (if special case)
    """
    label_lower = tess_label.lower().strip()
    
    mapping = {
        'angry': 'angry',
        'fear': 'fear',
        'happy': 'happy',
        'sad': 'sad',
        'neutral': 'neutral',
        'disgust': 'angry',  # Not in our target set
        'pleasant_surprised': 'happy',  # Positive arousal
        'ps': 'happy'  # Abbreviation
    }
    
    if label_lower in mapping:
        return mapping[label_lower], None
    else:
        return None, f"unknown:{tess_label}"


def map_cremad_label(cremad_label):
    """
    Map CREMA-D emotion label to target label.
    
    CREMA-D emotions: Angry, Disgust, Fear, Happy, Neutral, Sad
    (with intensity levels: Low, Medium, High, Unspecified)
    
    Mapping:
      - Angry → angry
      - Fear → fear
      - Happy → happy
      - Neutral → neutral
      - Sad → sad
      - Disgust → angry
    
    Note: Intensity is ignored initially
    
    Args:
        cremad_label: Original CREMA-D label
    
    Returns:
        target_label: Mapped target label
        note: Mapping note
    """
    label_lower = cremad_label.lower().strip()
    
    mapping = {
        'angry': 'angry',
        'fear': 'fear',
        'happy': 'happy',
        'neutral': 'neutral',
        'sad': 'sad',
        'disgust': 'angry'
    }
    
    if label_lower in mapping:
        return mapping[label_lower], None
    else:
        return None, f"unknown:{cremad_label}"


# ============================================
# STRESS DERIVATION
# ============================================

def derive_stress_from_emotion(emotion_label, source_label=None):
    """
    Derive stress level from emotion label.
    
    Rule-based stress labeling (psychologically valid):
      - calm → low
      - neutral → low-medium
      - happy → medium
      - sad → medium
      - fear → high
      - angry → high
    
    This is NOT guesswork - it's based on:
      - Arousal theory
      - Valence-arousal model
      - Standard psychology research
    
    Args:
        emotion_label: Target emotion label
        source_label: Original dataset label (for calm detection)
    
    Returns:
        stress_level: 'low', 'medium', or 'high'
        stress_score: Continuous score 0-1
    """
    emotion_lower = emotion_label.lower().strip()
    
    # Check if source was 'calm' (special case)
    is_calm = source_label and source_label.lower().strip() == 'calm'
    
    # Rule-based stress mapping
    stress_rules = {
        'angry': ('high', 0.85),
        'fear': ('high', 0.90),
        'sad': ('medium', 0.50),
        'happy': ('medium', 0.45),
        'neutral': ('low', 0.20) if not is_calm else ('low', 0.10)
    }
    
    if emotion_lower in stress_rules:
        return stress_rules[emotion_lower]
    else:
        # Default to medium
        return ('medium', 0.50)


# ============================================
# SPEAKER-INDEPENDENT SPLITS
# ============================================

def create_speaker_independent_split(data_dict, train_ratio=0.7, val_ratio=0.15, 
                                     test_ratio=0.15, random_seed=42):
    """
    Create speaker-independent train/val/test splits.
    
    CRITICAL: Same speaker must NOT appear in train and test!
    
    This prevents speaker leakage and ensures generalization.
    
    Args:
        data_dict: Dictionary with 'file_path', 'emotion', 'speaker_id', etc.
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
    
    Returns:
        train_data: Training samples
        val_data: Validation samples
        test_data: Test samples
    """
    np.random.seed(random_seed)
    
    # Group samples by speaker
    speaker_to_samples = defaultdict(list)
    for idx, item in enumerate(data_dict):
        speaker_id = item.get('speaker_id', 'unknown')
        speaker_to_samples[speaker_id].append(idx)
    
    # Get all speaker IDs
    all_speakers = list(speaker_to_samples.keys())
    np.random.shuffle(all_speakers)
    
    # Split speakers (not samples!)
    num_speakers = len(all_speakers)
    num_train = int(num_speakers * train_ratio)
    num_val = int(num_speakers * val_ratio)
    
    train_speakers = all_speakers[:num_train]
    val_speakers = all_speakers[num_train:num_train + num_val]
    test_speakers = all_speakers[num_train + num_val:]
    
    # Collect sample indices
    train_indices = []
    val_indices = []
    test_indices = []
    
    for speaker in train_speakers:
        train_indices.extend(speaker_to_samples[speaker])
    for speaker in val_speakers:
        val_indices.extend(speaker_to_samples[speaker])
    for speaker in test_speakers:
        test_indices.extend(speaker_to_samples[speaker])
    
    # Create split datasets
    train_data = [data_dict[i] for i in train_indices]
    val_data = [data_dict[i] for i in val_indices]
    test_data = [data_dict[i] for i in test_indices]
    
    print(f"[Split] Speaker-independent split:")
    print(f"  Train: {len(train_speakers)} speakers, {len(train_data)} samples")
    print(f"  Val: {len(val_speakers)} speakers, {len(val_data)} samples")
    print(f"  Test: {len(test_speakers)} speakers, {len(test_data)} samples")
    
    return train_data, val_data, test_data


# ============================================
# DATA BALANCING
# ============================================

def balance_dataset_by_emotion(data_dict, strategy='equal', target_count=None):
    """
    Balance dataset to prevent emotion bias.
    
    Problem: Voice datasets are imbalanced
      - happy / neutral dominate
      - fear / angry are rare
    
    Strategies:
      - 'equal': Equal samples per emotion (downsample majority)
      - 'upsample': Upsample minority to match majority
      - 'weighted': Keep all, but use weighted loss later
    
    Args:
        data_dict: List of data samples
        strategy: 'equal', 'upsample', or 'weighted'
        target_count: Target count per emotion (for equal)
    
    Returns:
        balanced_data: Balanced dataset
        weights: Class weights (for weighted strategy)
    """
    # Count samples per emotion
    emotion_to_samples = defaultdict(list)
    for idx, item in enumerate(data_dict):
        emotion = item['emotion']
        emotion_to_samples[emotion].append(idx)
    
    emotion_counts = {e: len(s) for e, s in emotion_to_samples.items()}
    print(f"\n[Balance] Original distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count}")
    
    if strategy == 'equal':
        # Downsample to minimum or target count
        if target_count is None:
            target_count = min(emotion_counts.values())
        
        balanced_indices = []
        for emotion, sample_indices in emotion_to_samples.items():
            if len(sample_indices) > target_count:
                # Downsample
                selected = np.random.choice(sample_indices, target_count, replace=False)
            else:
                # Keep all (or upsample if needed)
                selected = sample_indices
            balanced_indices.extend(selected)
        
        balanced_data = [data_dict[i] for i in balanced_indices]
        
        print(f"\n[Balance] After equal sampling ({target_count} per class):")
        new_counts = Counter([item['emotion'] for item in balanced_data])
        for emotion, count in sorted(new_counts.items()):
            print(f"  {emotion}: {count}")
        
        return balanced_data, None
    
    elif strategy == 'upsample':
        # Upsample minorities to match majority
        max_count = max(emotion_counts.values())
        
        balanced_indices = []
        for emotion, sample_indices in emotion_to_samples.items():
            if len(sample_indices) < max_count:
                # Upsample with replacement
                selected = np.random.choice(sample_indices, max_count, replace=True)
            else:
                selected = sample_indices
            balanced_indices.extend(selected)
        
        balanced_data = [data_dict[i] for i in balanced_indices]
        
        print(f"\n[Balance] After upsampling to {max_count}:")
        new_counts = Counter([item['emotion'] for item in balanced_data])
        for emotion, count in sorted(new_counts.items()):
            print(f"  {emotion}: {count}")
        
        return balanced_data, None
    
    elif strategy == 'weighted':
        # Keep all data, but compute weights for loss
        total_samples = len(data_dict)
        num_classes = len(emotion_counts)
        
        # Compute inverse frequency weights
        weights = {}
        for emotion, count in emotion_counts.items():
            weights[emotion] = total_samples / (num_classes * count)
        
        print(f"\n[Balance] Class weights (for weighted loss):")
        for emotion, weight in sorted(weights.items()):
            print(f"  {emotion}: {weight:.3f}")
        
        return data_dict, weights
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================
# UTILITY FUNCTIONS
# ============================================

def validate_dataset_labels(data_dict):
    """
    Validate that all labels are in target set.
    
    Args:
        data_dict: List of data samples
    
    Returns:
        is_valid: Whether all labels are valid
        invalid_samples: List of invalid samples
    """
    invalid_samples = []
    
    for idx, item in enumerate(data_dict):
        emotion = item.get('emotion')
        if emotion not in TARGET_EMOTIONS:
            invalid_samples.append({
                'index': idx,
                'file': item.get('file_path'),
                'emotion': emotion
            })
    
    is_valid = len(invalid_samples) == 0
    
    if not is_valid:
        print(f"\n[Warning] Found {len(invalid_samples)} invalid labels:")
        for sample in invalid_samples[:5]:  # Show first 5
            print(f"  {sample['file']}: {sample['emotion']}")
    
    return is_valid, invalid_samples


def get_dataset_statistics(data_dict):
    """
    Get comprehensive dataset statistics.
    
    Args:
        data_dict: List of data samples
    
    Returns:
        stats: Dictionary of statistics
    """
    emotion_counts = Counter([item['emotion'] for item in data_dict])
    stress_counts = Counter([item.get('stress_level', 'unknown') for item in data_dict])
    speaker_counts = len(set([item.get('speaker_id', 'unknown') for item in data_dict]))
    
    stats = {
        'total_samples': len(data_dict),
        'num_speakers': speaker_counts,
        'emotion_distribution': dict(emotion_counts),
        'stress_distribution': dict(stress_counts),
        'balance_ratio': min(emotion_counts.values()) / max(emotion_counts.values())
    }
    
    return stats


def print_dataset_info(data_dict, name="Dataset"):
    """Print formatted dataset information."""
    stats = get_dataset_statistics(data_dict)
    
    print(f"\n{'='*60}")
    print(f"{name.upper()} STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Num speakers: {stats['num_speakers']}")
    print(f"Balance ratio: {stats['balance_ratio']:.3f}")
    
    print(f"\nEmotion distribution:")
    for emotion, count in sorted(stats['emotion_distribution'].items()):
        pct = 100 * count / stats['total_samples']
        print(f"  {emotion:8s}: {count:4d} ({pct:5.1f}%)")
    
    if 'unknown' not in stats['stress_distribution']:
        print(f"\nStress distribution:")
        for level, count in sorted(stats['stress_distribution'].items()):
            pct = 100 * count / stats['total_samples']
            print(f"  {level:8s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test label mapping
    print("=" * 60)
    print("LABEL MAPPING TEST")
    print("=" * 60)
    
    # Test RAVDESS mapping
    print("\n[RAVDESS Mapping]")
    ravdess_labels = ['angry', 'fearful', 'happy', 'sad', 'neutral', 'calm', 'disgust']
    for label in ravdess_labels:
        target, note = map_ravdess_label(label)
        print(f"  {label:10s} → {target:8s} {f'({note})' if note else ''}")
    
    # Test TESS mapping
    print("\n[TESS Mapping]")
    tess_labels = ['angry', 'fear', 'happy', 'sad', 'neutral', 'disgust', 'pleasant_surprised']
    for label in tess_labels:
        target, note = map_tess_label(label)
        print(f"  {label:20s} → {target:8s}")
    
    # Test stress derivation
    print("\n[Stress Derivation]")
    for emotion in TARGET_EMOTIONS:
        stress_level, stress_score = derive_stress_from_emotion(emotion)
        print(f"  {emotion:8s} → stress: {stress_level:6s} (score: {stress_score:.2f})")
    
    # Special case: calm
    stress_level, stress_score = derive_stress_from_emotion('neutral', source_label='calm')
    print(f"  calm     → stress: {stress_level:6s} (score: {stress_score:.2f})")
    
    print("\n" + "=" * 60)
    print("✓ Label mapping and stress derivation ready!")
    print("=" * 60)
