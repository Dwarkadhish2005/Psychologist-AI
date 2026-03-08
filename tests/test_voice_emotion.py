"""
Voice Emotion Model Test
========================
Standalone test script for the Phase 2 voice emotion model.

Usage:
    python tests/test_voice_emotion.py
    python tests/test_voice_emotion.py --model balanced
    python tests/test_voice_emotion.py --model all
    python tests/test_voice_emotion.py --limit 100

Models:
    balanced  - emotion_model_best_balanced.pth  (default, trained with class balancing)
    improved  - emotion_model_best_improved.pth  (improved training)
    original  - emotion_model_best.pth           (original training)
    all       - test all three models and compare
"""

import sys
import os
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'training', 'voice'))

import numpy as np
import torch
import librosa

from voice_emotion_model import VoiceEmotionModel
from feature_extraction import extract_all_features
from audio_preprocessing import preprocess_audio
from utils import (
    confusion_matrix, print_confusion_matrix,
    per_class_stats, print_per_class_stats,
    macro_avg, print_summary_box
)


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = ['angry', 'fear', 'happy', 'neutral', 'sad']

MODEL_FILES = {
    'balanced': os.path.join(ROOT, 'models', 'voice_emotion', 'emotion_model_best_balanced.pth'),
    'improved': os.path.join(ROOT, 'models', 'voice_emotion', 'emotion_model_best_improved.pth'),
    'original': os.path.join(ROOT, 'models', 'voice_emotion', 'emotion_model_best.pth'),
}

SPLITS_FILE = os.path.join(ROOT, 'data', 'voice_emotion', 'dataset_splits.json')

# Load feature scaler that was saved during training.
# Critical: test must use the same feature distribution as training.
_scaler = None
_SCALER_PATH = os.path.join(ROOT, 'models', 'voice_emotion', 'feature_scaler.pkl')
if os.path.exists(_SCALER_PATH):
    try:
        import joblib
        _scaler = joblib.load(_SCALER_PATH)
    except Exception as e:
        print(f"[WARN] Could not load feature scaler: {e}")

# RAVDESS emotion code → class name
# 01=neutral, 02=calm(→neutral), 03=happy, 04=sad, 05=angry, 06=fearful(→fear), 07=disgust(skip), 08=surprised(skip)
RAVDESS_MAP = {
    '01': 'neutral',
    '02': 'neutral',   # calm → neutral
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',      # fearful → fear
    '07': None,        # disgust → not in model classes, skip
    '08': None,        # surprised → not in model classes, skip
}

# TESS suffix → class name
TESS_SUFFIX_MAP = {
    'neutral': 'neutral',
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'fear': 'fear',
    'fearful': 'fear',
    'disgust': None,   # skip
    'ps': None,        # "pleasant surprise" → skip
}


# ============================================================
# LABEL EXTRACTION
# ============================================================

def extract_label_ravdess(filename):
    """
    Extract emotion label from RAVDESS filename.
    Format: 03-01-05-01-01-01-01.wav
    3rd segment (index 2) is the emotion code.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split('-')
    if len(parts) < 3:
        return None
    code = parts[2]
    return RAVDESS_MAP.get(code, None)


def extract_label_tess(filename):
    """
    Extract emotion label from TESS filename.
    Format: OAF_word_emotion.wav  or  YAF_word_emotion.wav
    Last segment before extension is the emotion.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split('_')
    if len(parts) < 3:
        return None
    emotion = parts[-1].lower()
    return TESS_SUFFIX_MAP.get(emotion, None)


def extract_label(filepath):
    """Determine dataset source and extract label."""
    if 'RAVDESS' in filepath.replace('\\', '/').replace('//', '/'):
        return extract_label_ravdess(filepath)
    elif 'TESS' in filepath.replace('\\', '/'):
        return extract_label_tess(filepath)
    return None


# ============================================================
# DATA LOADING
# ============================================================

def load_test_samples(limit=None):
    """
    Load test sample paths and labels from dataset_splits.json.

    Returns:
        samples: list of (filepath, label_str) tuples
        skipped: count of skipped/unlabeled files
    """
    if not os.path.exists(SPLITS_FILE):
        print(f"[ERROR] dataset_splits.json not found at:\n  {SPLITS_FILE}")
        sys.exit(1)

    with open(SPLITS_FILE, 'r') as f:
        splits = json.load(f)

    test_paths = splits.get('test', [])
    if not test_paths:
        print("[ERROR] No 'test' key found in dataset_splits.json")
        sys.exit(1)

    samples = []
    skipped = 0

    for path in test_paths:
        label = extract_label(path)
        if label is None:
            skipped += 1
            continue
        if not os.path.exists(path):
            skipped += 1
            continue
        samples.append((path, label))

    if limit:
        # Balanced subsample: pick up to limit//5 per class
        per_class = limit // len(CLASS_NAMES)
        buckets = {c: [] for c in CLASS_NAMES}
        for path, label in samples:
            if label in buckets and len(buckets[label]) < per_class:
                buckets[label].append((path, label))
        samples = [s for bucket in buckets.values() for s in bucket]

    print(f"  Loaded {len(samples)} test samples  (skipped {skipped} — disgust/surprise/missing)")

    # Show per-class distribution
    from collections import Counter
    dist = Counter(label for _, label in samples)
    for cls in CLASS_NAMES:
        print(f"    {cls:<10} {dist.get(cls, 0):>4} samples")

    return samples, skipped


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def load_and_extract(filepath):
    """
    Load audio through the SAME pipeline used during training:
      sr=16000, noise reduction, RMS normalization, pre-emphasis.
    Then extract the 48-dim feature vector and apply the saved StandardScaler.
    Matching this pipeline to training is critical — features at sr=22050
    have completely different distributions and cause model collapse.
    """
    try:
        audio, sr, _ = preprocess_audio(filepath)
        _, feature_vector = extract_all_features(audio, sr)
        if feature_vector.shape[0] != 48:
            return None
        if _scaler is not None:
            feature_vector = _scaler.transform(feature_vector.reshape(1, -1)).flatten()
        return feature_vector
    except Exception:
        return None


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_key):
    """Load a VoiceEmotionModel from disk."""
    model_path = MODEL_FILES[model_key]
    if not os.path.exists(model_path):
        return None, f"File not found: {model_path}"

    try:
        model = VoiceEmotionModel(input_dim=48, num_classes=5)
        state = torch.load(model_path, map_location='cpu', weights_only=True)

        # State dict may be wrapped in 'model_state_dict'
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        elif isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']

        model.load_state_dict(state)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, samples, device='cpu'):
    """
    Run inference on all samples.
    Returns y_true (list of int) and y_pred (list of int).
    """
    model = model.to(device)
    model.eval()

    label_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    y_true = []
    y_pred = []
    errors = 0

    # Try importing tqdm for progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(samples, desc="  Evaluating", ncols=70, unit="clip")
    except ImportError:
        iterator = samples

    for filepath, label in iterator:
        features = load_and_extract(filepath)
        if features is None:
            errors += 1
            continue

        feat_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        pred_idx, confidence, _ = model.predict_emotion(feat_tensor)

        y_true.append(label_to_idx[label])
        y_pred.append(pred_idx)

    if errors > 0:
        print(f"  [WARN] {errors} files skipped due to feature extraction errors")

    return y_true, y_pred


# ============================================================
# RESULTS PRINTING
# ============================================================

def print_results(model_name, y_true, y_pred):
    """Print full evaluation results for a model."""
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0.0

    # Convert integer indices → class name strings for utils functions
    y_true_str = [CLASS_NAMES[i] for i in y_true]
    y_pred_str = [CLASS_NAMES[i] for i in y_pred]

    cm = confusion_matrix(y_true_str, y_pred_str, CLASS_NAMES)
    stats = per_class_stats(cm, CLASS_NAMES)
    m_prec, m_rec, m_f1 = macro_avg(stats)

    print_summary_box(
        f"VOICE EMOTION — {model_name.upper()} MODEL",
        [
            f"Total samples : {total}",
            f"Correct       : {correct}",
            f"Overall acc   : {accuracy * 100:.2f}%",
            f"Macro-Precision: {m_prec * 100:.2f}%",
            f"Macro-Recall   : {m_rec * 100:.2f}%",
            f"Macro-F1       : {m_f1 * 100:.2f}%",
        ]
    )

    print_confusion_matrix(cm, CLASS_NAMES)
    print_per_class_stats(stats)

    # Best and worst class
    sorted_by_f1 = sorted(stats.items(), key=lambda x: x[1]['f1'], reverse=True)
    print(f"\n  Best class  : {sorted_by_f1[0][0]}  (F1 {sorted_by_f1[0][1]['f1']*100:.1f}%)")
    print(f"  Worst class : {sorted_by_f1[-1][0]} (F1 {sorted_by_f1[-1][1]['f1']*100:.1f}%)")

    return accuracy, m_f1


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test voice emotion model accuracy")
    parser.add_argument(
        '--model', choices=['balanced', 'improved', 'original', 'all'],
        default='balanced',
        help='Which model checkpoint to evaluate (default: balanced)'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Max test samples per class (approx). E.g. --limit 250'
    )
    parser.add_argument(
        '--device', choices=['cpu', 'cuda', 'auto'], default='auto',
        help='Device to run inference on (default: auto)'
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"\n  Device: {device}")

    # Load all test samples once
    print("\n[1] Loading test samples ...")
    samples, skipped = load_test_samples(limit=args.limit)

    if not samples:
        print("[ERROR] No samples loaded. Check dataset_splits.json and file paths.")
        sys.exit(1)

    # Determine which models to evaluate
    if args.model == 'all':
        model_keys = ['balanced', 'improved', 'original']
    else:
        model_keys = [args.model]

    results = {}

    for model_key in model_keys:
        print(f"\n[2] Loading model: {model_key}")
        model, err = load_model(model_key)
        if model is None:
            print(f"  [SKIP] {model_key}: {err}")
            continue

        print(f"  OK — {MODEL_FILES[model_key]}")
        print(f"\n[3] Running inference on {len(samples)} clips ...")

        y_true, y_pred = evaluate(model, samples, device=device)

        if not y_true:
            print("  [ERROR] No predictions produced.")
            continue

        print(f"\n[4] Results for '{model_key}' model:")
        acc, f1 = print_results(model_key, y_true, y_pred)
        results[model_key] = {'accuracy': acc, 'f1': f1}

    # If testing all models, print comparison table
    if len(results) > 1:
        print("\n" + "=" * 50)
        print("  MODEL COMPARISON")
        print("=" * 50)
        print(f"  {'Model':<12}  {'Accuracy':>10}  {'Macro-F1':>10}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*10}")
        for key, res in results.items():
            print(f"  {key:<12}  {res['accuracy']*100:>9.2f}%  {res['f1']*100:>9.2f}%")
        best = max(results, key=lambda k: results[k]['f1'])
        print(f"\n  Best model: {best} (F1 {results[best]['f1']*100:.2f}%)")

    print("\nDone.\n")


if __name__ == '__main__':
    main()
