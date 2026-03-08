"""
╔══════════════════════════════════════════════════════════════════╗
║         FACE EMOTION MODEL — ACCURACY TEST                      ║
║  Tests both main model and specialist model on the held-out     ║
║  test set (data/face_emotion/test/).                            ║
║                                                                  ║
║  Usage:                                                          ║
║    python tests/test_face_emotion.py                            ║
║    python tests/test_face_emotion.py --limit 500  (quick run)  ║
║    python tests/test_face_emotion.py --model main               ║
║    python tests/test_face_emotion.py --model specialist         ║
║    python tests/test_face_emotion.py --model both               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ── project root on path ─────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from training.model import EmotionCNN, EmotionCNNDeep
from training.preprocessing import preprocess_face
from utils import (
    confusion_matrix,
    print_confusion_matrix,
    per_class_stats,
    print_per_class_stats,
    macro_avg,
    print_summary_box,
)

# ── constants ────────────────────────────────────────────────────
FACE_MODEL_DIR = ROOT / "models" / "face_emotion"
TEST_DATA_DIR  = ROOT / "data"  / "face_emotion" / "test"
CONFIG_FILE    = FACE_MODEL_DIR / "config.json"

MINORITY_CLASSES = {"disgust", "fear"}
SPECIALIST_THRESHOLD = 0.60


# ══════════════════════════════════════════════════════════════════
# Loader helpers
# ══════════════════════════════════════════════════════════════════

def load_model(model_path: Path, num_classes: int, device, architecture: str = 'EmotionCNN'):
    _ARCH_MAP = {'EmotionCNNDeep': EmotionCNNDeep, 'EmotionCNN': EmotionCNN}
    ModelClass = _ARCH_MAP.get(architecture, EmotionCNN)
    model = ModelClass(num_classes=num_classes, input_size=48)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_config():
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    return cfg["class_names"], cfg["num_classes"], cfg.get("architecture", "EmotionCNN")


# ══════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════

def predict_single(model, face_img, device):
    """Return (emotion, confidence) for one BGR face image."""
    tensor = torch.from_numpy(preprocess_face(face_img)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])


def predict_dual(main_model, specialist_model, face_img, class_names, device):
    """Dual-model strategy mirroring integrated_psychologist_ai.py."""
    idx, conf = predict_single(main_model, face_img, device)
    emotion = class_names[idx]

    if emotion in MINORITY_CLASSES or conf < SPECIALIST_THRESHOLD:
        s_idx, s_conf = predict_single(specialist_model, face_img, device)
        s_emotion = class_names[s_idx]
        if s_emotion in MINORITY_CLASSES and s_conf > 0.50:
            return s_emotion, s_conf

    return emotion, conf


# ══════════════════════════════════════════════════════════════════
# Collect test images
# ══════════════════════════════════════════════════════════════════

def gather_test_samples(class_names, limit=None):
    """
    Returns list of (image_path, true_label) pairs.
    When limit is set, samples evenly across classes.
    """
    samples = []
    per_class = (limit // len(class_names)) if limit else None

    for cls in class_names:
        cls_dir = TEST_DATA_DIR / cls
        if not cls_dir.exists():
            print(f"[Warning] Missing class folder: {cls_dir}")
            continue
        files = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        if per_class:
            files = files[:per_class]
        for f in files:
            samples.append((f, cls))

    return samples


# ══════════════════════════════════════════════════════════════════
# Evaluate one model
# ══════════════════════════════════════════════════════════════════

def evaluate_model(model, class_names, samples, device, label="Model"):
    y_true, y_pred = [], []
    errors = 0
    t0 = time.time()

    for img_path, true_label in tqdm(samples, desc=f"  {label}", ncols=70):
        img = cv2.imread(str(img_path))
        if img is None:
            errors += 1
            continue

        try:
            idx, _ = predict_single(model, img, device)
            pred_label = class_names[idx]
        except Exception:
            errors += 1
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)

    elapsed = time.time() - t0
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total   = len(y_true)
    acc     = correct / total if total else 0

    return y_true, y_pred, acc, elapsed, errors


def evaluate_dual(main_model, specialist_model, class_names, samples, device):
    y_true, y_pred = [], []
    errors = 0
    t0 = time.time()

    for img_path, true_label in tqdm(samples, desc="  Dual-model", ncols=70):
        img = cv2.imread(str(img_path))
        if img is None:
            errors += 1
            continue

        try:
            pred_label, _ = predict_dual(main_model, specialist_model, img, class_names, device)
        except Exception:
            errors += 1
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)

    elapsed = time.time() - t0
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total   = len(y_true)
    acc     = correct / total if total else 0

    return y_true, y_pred, acc, elapsed, errors


# ══════════════════════════════════════════════════════════════════
# Print results
# ══════════════════════════════════════════════════════════════════

def print_results(y_true, y_pred, acc, elapsed, errors, class_names, title):
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total   = len(y_true)
    cm      = confusion_matrix(y_true, y_pred, class_names)
    stats   = per_class_stats(cm, class_names)
    prec, rec, f1 = macro_avg(stats)

    sep = "─" * 64
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(f"{'═'*64}")

    print(f"\n  Overall Accuracy : {acc*100:.2f}%  ({correct}/{total})")
    print(f"  Macro Precision  : {prec*100:.2f}%")
    print(f"  Macro Recall     : {rec*100:.2f}%")
    print(f"  Macro F1-Score   : {f1*100:.2f}%")
    print(f"  Inference time   : {elapsed:.1f}s  ({total/elapsed:.1f} imgs/sec)")
    if errors:
        print(f"  Skipped (errors) : {errors}")

    print(f"\n{sep}")
    print("  Per-class stats:")
    print(sep)
    print_per_class_stats(stats)

    # Highlight worst class
    worst = min(stats, key=lambda c: stats[c]["recall"])
    best  = max(stats, key=lambda c: stats[c]["recall"])
    print(f"\n  Best  class: {best}  (recall {stats[best]['recall']*100:.1f}%)")
    print(f"  Worst class: {worst} (recall {stats[worst]['recall']*100:.1f}%)")

    print(f"\n{sep}")
    print("  Confusion Matrix (rows=true, cols=predicted):")
    print(sep)
    print_confusion_matrix(cm, class_names)

    return stats


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Face emotion model accuracy test")
    parser.add_argument("--model",  choices=["main", "specialist", "both", "dual"], default="dual",
                        help="Which model(s) to test (default: dual — tests with the live dual-model strategy)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Max images to test (default: all ~7178). Use e.g. 700 for a quick run.")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    print(f"\n{'═'*64}")
    print("  FACE EMOTION MODEL — ACCURACY TEST")
    print(f"{'═'*64}")
    print(f"  Device  : {device}")
    print(f"  Test dir: {TEST_DATA_DIR}")

    # Load config
    class_names, num_classes, main_architecture = load_config()
    print(f"  Classes : {class_names}")

    # Gather samples
    samples = gather_test_samples(class_names, limit=args.limit)
    print(f"  Samples : {len(samples)}")
    if not samples:
        print("[ERROR] No test images found. Check data/face_emotion/test/")
        return

    # Count per class
    from collections import Counter
    cls_count = Counter(label for _, label in samples)
    print(f"  Per class: " + ", ".join(f"{c}:{n}" for c, n in sorted(cls_count.items())))

    # Load models
    main_path       = FACE_MODEL_DIR / "emotion_cnn_best.pth"
    specialist_path = FACE_MODEL_DIR / "emotion_cnn_phase15_specialist.pth"

    run_main       = args.model in ("main", "both", "dual")
    run_specialist = args.model in ("specialist", "both", "dual")

    main_model       = load_model(main_path,       num_classes, device, main_architecture) if run_main       else None
    specialist_model = load_model(specialist_path, num_classes, device, 'EmotionCNN')        if run_specialist else None

    # ── Test main model ────────────────────────────────────────────
    if args.model in ("main", "both"):
        print(f"\n[1/2] Testing MAIN model: {main_path.name}")
        y_true, y_pred, acc, elapsed, errors = evaluate_model(
            main_model, class_names, samples, device, "Main model"
        )
        print_results(y_true, y_pred, acc, elapsed, errors, class_names,
                      f"MAIN MODEL — {main_path.name}")

    # ── Test specialist model ──────────────────────────────────────
    if args.model in ("specialist", "both"):
        print(f"\n[{'2/2' if args.model == 'both' else '1/1'}] Testing SPECIALIST model: {specialist_path.name}")
        y_true, y_pred, acc, elapsed, errors = evaluate_model(
            specialist_model, class_names, samples, device, "Specialist"
        )
        print_results(y_true, y_pred, acc, elapsed, errors, class_names,
                      f"SPECIALIST MODEL — {specialist_path.name}")

    # ── Test dual-model strategy (default, mirrors live inference) ─
    if args.model == "dual":
        print(f"\n[DUAL] Testing DUAL-MODEL strategy (mirrors live inference):")
        print(f"       Main: {main_path.name}")
        print(f"       Specialist: {specialist_path.name}")
        y_true, y_pred, acc, elapsed, errors = evaluate_dual(
            main_model, specialist_model, class_names, samples, device
        )
        print_results(y_true, y_pred, acc, elapsed, errors, class_names,
                      "DUAL-MODEL STRATEGY (live inference replication)")

    print(f"\n{'═'*64}")
    print("  DONE")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()
