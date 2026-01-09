# PHASE 1.5: Fine-Tuning for Minority Class Improvement

**Objective:** Improve Phase 1 model generalization on minority emotion classes (disgust, fear) while preserving learned features and maintaining or improving overall accuracy.

**Target:** +3-6% accuracy improvement, particularly better recall for minority classes.

---

## Problem Statement

### Class Imbalance in Training Data

The FER2013 dataset has significant class imbalance:

```
Class Distribution (Training Set):
  angry:    2,887 samples (13.9%)
  disgust:    316 samples (1.5%)  ← MINORITY
  fear:     2,961 samples (14.3%)  ← MINORITY
  happy:    5,214 samples (25.1%)  ← MAJORITY
  neutral:  3,588 samples (17.3%)
  sad:      3,491 samples (16.8%)
  surprise: 2,292 samples (11.0%)
```

**Impact:**
- Model optimizes for majority class (happy)
- Minority classes (disgust: 1.5%, fear: 14.3%) get less training signal
- Model struggles to distinguish disgust/fear in real-world scenarios
- Overall accuracy looks good but useless on rare emotions

---

## Strategy: Conservative Fine-Tuning

We are **NOT** redesigning the model or chasing raw accuracy. Instead, we focus on **decision logic rebalancing** using four key levers:

### 1. **Weighted Loss (Highest Impact)**
- Assign higher loss weight to minority classes
- Formula: `weight[class] = total_samples / (num_classes * count[class])`
- Effect: Each minority class sample contributes more gradient pressure
- Example: Disgust (1.5%) → weight ~16x higher than happy (25%)

### 2. **Frozen Early Layers (Safety)**
- Freeze first 2 convolutional blocks (edge/shape detection)
- Allow only later layers (emotion-specific features) to train
- Benefit: Preserve learned low-level features, prevent catastrophic forgetting

### 3. **Lower Learning Rate (Refinement)**
- Use 0.0001 instead of Phase 1's 0.001 (10x reduction)
- Effect: Small, careful nudges to decision boundaries
- Prevents overwriting Phase 1's learned patterns

### 4. **Short Training (Nudge, Not Redesign)**
- Train for 15 epochs instead of Phase 1's 50
- Early stopping patience: 5 (stop if no improvement)
- Effect: Gentle refinement, avoid overfitting

---

## Architecture

### Phase 1.5 Training Pipeline

```
Load Phase 1 Best Model
    ↓
Calculate Class Weights (inverse frequency)
    ↓
Freeze First 2 Conv Blocks (optional)
    ↓
Train with:
  - Weighted CrossEntropyLoss
  - Learning Rate = 0.0001
  - Adam optimizer
  - 15 epochs max
    ↓
Save Best Model
    ↓
Compare Phase 1 vs Phase 1.5:
  - Overall accuracy
  - Per-class recall (focus on disgust, fear)
  - Confusion matrix changes
    ↓
Decision: Keep Phase 1.5 or revert?
```

### Implementation Details

**File:** `training/train_phase_1_5_finetune.py`

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `Config` | Phase 1.5 hyperparameters and paths |
| `EmotionDataset` | Data loading (reused from Phase 1) |
| `EmotionCNN` | Model architecture (reused from Phase 1) |

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `calculate_class_weights()` | Compute inverse-frequency weights |
| `freeze_early_layers()` | Freeze first N conv blocks |
| `train_one_epoch()` | Training step with progress bar |
| `validate()` | Validation step |
| `evaluate_with_class_metrics()` | Per-class recall + confusion matrix |
| `plot_phase_1_5_results()` | Compare Phase 1 vs 1.5 visually |

---

## Hyperparameter Justification

### Learning Rate: 0.0001 (10x lower)
- **Rationale:** Phase 1.5 is refinement, not discovery
- **Effect:** Smaller parameter updates preserve learned features
- **Phase 1:** Started at 0.001 for initial learning
- **Phase 1.5:** Use 0.0001 for careful fine-tuning

### Epochs: 15 (vs Phase 1's 50)
- **Rationale:** Short training prevents overfitting and catastrophic forgetting
- **Effect:** Each class "seen" ~3x during Phase 1.5 (vs ~10x in Phase 1)
- **Early Stopping:** If validation doesn't improve for 5 epochs, stop
- **Expected:** 8-12 epochs in practice (early stop kicks in)

### Weighted Loss Calculation
```
For each class:
  weight = total_samples / (num_classes * count[class])
  
Example:
  happy (5214/20749):   20749 / (7 * 5214) = 0.567
  disgust (316/20749):  20749 / (7 * 316)  = 9.395  ← 16.5x higher!
  fear (2961/20749):    20749 / (7 * 2961) = 0.996
```

Weights normalized to prevent training instability.

### Frozen Layers
- **Freeze up to:** Block 2 (6 parameters frozen)
- **Trainable:** Block 3, FC layers (~1M parameters)
- **Rationale:** Edges/shapes (Block 1-2) generic across emotions
  - Disgust/fear need different high-level features, not edge detection

---

## Evaluation Metrics

### Phase 1 (Baseline)
- Recorded from Phase 1 training
- Test set accuracy + per-class recall

### Phase 1.5 (Fine-tuned)
- Same test set for fair comparison
- Per-class recall (focus on disgust + fear)
- Confusion matrix changes

### Comparison Framework

```
Decision Tree:
├─ If test_acc[1.5] >= test_acc[1] AND minority_recall improved
│  └─ KEEP Phase 1.5 (clear winner)
├─ Else if test_acc[1.5] >= test_acc[1] - 1% AND minority_recall +2%+
│  └─ KEEP Phase 1.5 (acceptable trade)
└─ Else
   └─ REVERT Phase 1 (no benefit)
```

### Minority Class Focus
- **Disgust:** Class index 1 (most critical - only 316 samples)
- **Fear:** Class index 2 (challenging - 14.3% of data)
- **Target:** Recall improvement of +2-5% without hurting others

---

## Expected Results

### Best Case (Likely)
- **Overall Accuracy:** +1-3% improvement
- **Disgust Recall:** +5-10% (from low baseline)
- **Fear Recall:** +2-4%
- **Happy Recall:** -1-2% (acceptable trade-off)
- **Confusion Matrix:** Fewer false negatives on disgust/fear

### Moderate Case
- **Overall Accuracy:** ±1% (maintained)
- **Minority Recall:** +2-3%
- **Action:** KEEP Phase 1.5 (minority improvement justifies tiny accuracy trade)

### Worst Case (Unlikely with conservative settings)
- **Overall Accuracy:** -2-3%
- **Minority Recall:** No improvement
- **Action:** REVERT Phase 1

---

## Running Phase 1.5

### Prerequisites
1. Phase 1 training complete
2. Best Phase 1 model saved: `models/face_emotion/emotion_cnn_best.pth`
3. Dataset in place: `data/face_emotion/{train,val,test}/`

### Command
```bash
python training/train_phase_1_5_finetune.py
```

### Expected Output
```
============================================================
PHASE 1.5: FINE-TUNING FOR MINORITY CLASS IMPROVEMENT
============================================================
Device: cuda (or cpu)
Batch Size: 32
Learning Rate: 0.0001 (lower than Phase 1)
Epochs: 15 (short training)
Weighted Loss: True
Freeze Early Layers: True
============================================================

Loading datasets...
[OK] Detected 7 classes: angry, disgust, fear, happy, neutral, sad, surprise

Loading Phase 1 model (transfer learning)...
[OK] Loaded Phase 1 model
     Total parameters: 1,274,823

[Weighted Loss] Class weights calculated:
  angry     : 0.992
  disgust   : 9.395  ← High weight for minority
  fear      : 0.996
  happy     : 0.567  ← Low weight for majority
  neutral   : 0.813
  sad       : 0.827
  surprise  : 1.089

Fine-tuning for 15 epochs (Phase 1.5)...

Epoch 1/15
...
Results:
  Train Loss: 1.2341 | Train Acc: 42.15%
  Val Loss:   1.0987 | Val Acc:   48.32%
  [Save] New best model (val_acc: 48.32%)

[... epochs 2-15 ...]

============================================================
FINAL EVALUATION - PHASE 1.5
============================================================
[Test Results]

Phase 1 Test Accuracy: 68.42%
Phase 1.5 Test Accuracy: 70.15%
Improvement: +1.73%

============================================================
PHASE 1 vs PHASE 1.5 COMPARISON
============================================================

Overall Accuracy:
  Phase 1:    68.42%
  Phase 1.5:  70.15%
  Improvement: +1.73%

Per-Class Recall:
Class          Phase 1    Phase 1.5      Change
------------------------------------------
angry            65.23%     66.14%      +0.91%
disgust          31.45%     39.62%      +8.17% ← Significant improvement!
fear             58.23%     61.85%      +3.62%
happy            82.15%     80.94%      -1.21%
neutral          72.31%     73.45%      +1.14%
sad              68.45%     70.12%      +1.67%
surprise         74.23%     75.31%      +1.08%

Minority Classes (Disgust + Fear) Avg Recall Improvement: +5.90%

============================================================
DECISION
============================================================

[OK] Phase 1.5 improves or maintains performance!
  - Accuracy maintained or improved
  - Minority class recall improved

>>> RECOMMENDATION: Use Phase 1.5 model as new baseline

============================================================

[Save] Phase 1.5 model saved as new best model
[Save] Evaluation report: reports/phase15_evaluation_20240115_093542.json
[Save] Comparison plots: reports/phase15_comparison_20240115_093542_comparison.png

[OK] Phase 1.5 fine-tuning complete!
```

---

## Output Files

### Evaluation Report
**File:** `reports/phase15_evaluation_<timestamp>.json`

```json
{
  "phase": 1.5,
  "timestamp": "20240115_093542",
  "phase1_accuracy": 68.42,
  "phase15_accuracy": 70.15,
  "accuracy_improvement": 1.73,
  "phase1_recall": {
    "angry": 0.6523,
    "disgust": 0.3145,
    "fear": 0.5823,
    "happy": 0.8215,
    "neutral": 0.7231,
    "sad": 0.6845,
    "surprise": 0.7423
  },
  "phase15_recall": {
    "angry": 0.6614,
    "disgust": 0.3962,
    "fear": 0.6185,
    "happy": 0.8094,
    "neutral": 0.7345,
    "sad": 0.7012,
    "surprise": 0.7531
  },
  "minority_class_improvement": 0.0590,
  "recommendation": "KEEP_PHASE15",
  "training_history": {
    "train_loss": [...],
    "train_acc": [...],
    "val_loss": [...],
    "val_acc": [...]
  }
}
```

### Comparison Plots
**File:** `reports/phase15_comparison_<timestamp>_comparison.png`

4 subplots:
1. Training & validation loss curves
2. Training & validation accuracy curves
3. Per-class recall comparison (all emotions)
4. Minority classes focus (disgust, fear only)

### Model File
- **If KEEP_PHASE15:** `models/face_emotion/emotion_cnn_best.pth` (replaced)
- **If REVERT_PHASE1:** Original Phase 1 model remains unchanged
- **Always saved:** `models/face_emotion/emotion_cnn_phase15_best.pth` (for reference)

---

## Decision Criteria

### Keep Phase 1.5?

✅ **YES, Keep Phase 1.5** if:
- Overall accuracy ≥ Phase 1 accuracy, AND
- Minority class (disgust/fear) recall improved by 2%+

✓ **MAYBE Keep Phase 1.5** if:
- Overall accuracy within 1% of Phase 1, AND
- Minority class recall improved by 3%+
- (Accept small accuracy trade for much better minority recall)

❌ **NO, Revert to Phase 1** if:
- Overall accuracy drops >1%, AND
- Minority recall unchanged or worse
- (No clear benefit, revert for stability)

---

## Troubleshooting

### Phase 1.5 Accuracy Drops Significantly
- **Cause:** Learning rate too high or training too long
- **Fix:** Reduce `LEARNING_RATE` to 0.00005 or lower `NUM_EPOCHS` to 10

### Minority Classes Still Not Improving
- **Cause:** Weighted loss not strong enough
- **Fix:** Multiply disgust weight by additional factor (e.g., 2.0x)
- **Alternative:** Use focal loss instead of weighted CE

### Validation Loss Increases (Overfitting)
- **Cause:** Model memorizing training data
- **Fix:** Increase `PATIENCE` to 6-7 for earlier stopping

### GPU Memory Issues
- **Cause:** Batch size too large with frozen layers
- **Fix:** Reduce `BATCH_SIZE` to 16

---

## Next Steps

### If Phase 1.5 Successful:
1. ✅ Update production model: `emotion_cnn_best.pth`
2. ✅ Document improvements in project README
3. ✅ Test updated webcam inference with new model
4. ⏭️ Proceed to Phase 2 (Voice Emotion Analysis)

### If Phase 1.5 Unsuccessful:
1. ✅ Keep Phase 1 model as baseline
2. ✅ Document findings (weighted loss limitations for this dataset)
3. ⏭️ Consider alternative approaches for Phase 2
   - Ensemble methods
   - Data augmentation for minority classes
   - Custom loss functions (focal loss)

---

## Technical Notes

### Why Transfer Learning?
- Starting from Phase 1 preserves ~40+ hours of training
- Takes advantage of learned low-level features
- Much faster than training from scratch

### Why Freeze Early Layers?
- Early CNN layers learn generic patterns (edges, textures)
- These are useful across all emotions
- Freezing prevents forgetting through catastrophic interference
- Only later layers (emotion-specific) need adjustment

### Why Weighted Loss?
- Standard CrossEntropyLoss treats all classes equally
- With imbalance, rare classes get drowned out
- Weighted loss makes rare samples matter more
- Mathematically: `loss = -weight[y] * log(p[y])`

### Why Lower Learning Rate?
- Phase 1: Discovered optimal decision boundaries (0.001 worked)
- Phase 1.5: Refining boundaries (need smaller steps)
- 10x lower LR → 10x smaller parameter updates
- Safer for transfer learning

---

## References

- **Weighted Loss:** https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- **Transfer Learning:** https://cs231n.github.io/transfer-learning/
- **Class Imbalance:** https://imbalanced-learn.org/
- **Fine-tuning Best Practices:** https://towardsdatascience.com/fine-tuning-pretrained-models-7e41558b2290

---

**Author:** Psychologist AI Team  
**Phase:** 1.5 (Fine-Tuning & Minority Class Improvement)  
**Status:** Ready for Fine-Tuning  
**Last Updated:** 2024-01-15

---

## Quick Start

```bash
# Run Phase 1.5 fine-tuning
python training/train_phase_1_5_finetune.py

# Check results
cat reports/phase15_evaluation_*.json | jq '.recommendation'

# View comparison plots
# Open: reports/phase15_comparison_*_comparison.png

# If good results, Phase 1.5 model is now production model
# If bad results, Phase 1 model remains unchanged
```
