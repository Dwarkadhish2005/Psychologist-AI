# PHASE 1.5: Quick Start Guide

## 📊 What is Phase 1.5?

Fine-tuning Phase 1 model to improve performance on minority emotion classes (disgust, fear) using:
1. **Weighted Loss** - Penalize minority class errors 16x more
2. **Frozen Early Layers** - Preserve learned features
3. **Lower Learning Rate** - Careful refinement (0.0001 vs 0.001)
4. **Short Training** - 15 epochs max, not 50

---

## 🚀 Run Phase 1.5

```bash
python training/train_phase_1_5_finetune.py
```

**Expected Duration:** 7-10 min (GPU) | 30-40 min (CPU)

---

## 📈 What to Expect

### Class Imbalance Before (Phase 1)
```
happy:    5,214 samples (25%)  ← Model optimizes for this
disgust:    316 samples (1.5%) ← Ignored!
fear:     2,961 samples (14%)  ← Underweighted
```

### Phase 1.5 Goal
```
Overall Accuracy:        +1-3% improvement
Disgust Recall:          +5-10% (critical!)
Fear Recall:             +2-4%
Happy Recall:            -1-2% (acceptable trade)
```

---

## 🎯 Decision Criteria

**KEEP Phase 1.5** if:
- ✅ Accuracy ≥ Phase 1 accuracy AND
- ✅ Minority class recall improved

**REVERT to Phase 1** if:
- ❌ Accuracy drops >1% AND
- ❌ Minority recall unchanged

---

## 📁 Output Files

After running, check:

1. **Evaluation Report**
   ```bash
   cat reports/phase15_evaluation_*.json | jq '.'
   ```
   Contains: accuracy comparison, per-class recall, recommendation

2. **Comparison Plots**
   ```bash
   # Open in image viewer:
   reports/phase15_comparison_*_comparison.png
   ```
   Shows: loss curves, accuracy curves, recall comparison

3. **Phase 1.5 Model**
   - If good results: `models/face_emotion/emotion_cnn_best.pth` (updated)
   - If bad results: `models/face_emotion/emotion_cnn_best.pth` (unchanged)

---

## 🔧 Customize (Optional)

Edit `training/train_phase_1_5_finetune.py`, Config class:

```python
class Config:
    LEARNING_RATE = 0.0001    # Try 0.00005 for slower learning
    NUM_EPOCHS = 15           # Try 10 for shorter training
    FREEZE_EARLY_LAYERS = True # Set False to train all layers
    PATIENCE = 5              # Try 7-8 for longer training
```

---

## ✅ Prerequisites Check

All should exist from Phase 1:
- ✅ `models/face_emotion/emotion_cnn_best.pth` (Phase 1 model)
- ✅ `data/face_emotion/train/` (20,749 images)
- ✅ `data/face_emotion/val/` (7,960 images)
- ✅ `data/face_emotion/test/` (7,178 images)
- ✅ All Python files in `training/`

---

## 📖 Detailed Info

See `PHASE_1_5_README.md` for:
- Full strategy explanation
- Hyperparameter justification
- Expected results details
- Troubleshooting guide
- Technical notes

---

## 💾 What Happens Automatically

1. **Load Phase 1 Model** → Transfer learning from Phase 1
2. **Calculate Weights** → Disgust gets 9.4x weight, happy gets 0.6x
3. **Fine-tune** → Train for max 15 epochs (stops early if no improvement)
4. **Evaluate Phase 1** → Get baseline metrics
5. **Evaluate Phase 1.5** → Get fine-tuned metrics
6. **Compare** → Show side-by-side results
7. **Decide** → Keep or revert (automatic)
8. **Save Results** → JSON report + plots
9. **Update Model** → If good, replaces Phase 1 model

---

## 🎨 Example Output

```
============================================================
PHASE 1.5: FINE-TUNING FOR MINORITY CLASS IMPROVEMENT
============================================================

Epoch 1/15
Training: 100%|████████| - loss: 1.234, acc: 42.15%
Validating: 100%|████████| - loss: 1.099, acc: 48.32%
  [Save] New best model (val_acc: 48.32%)

[... Epochs 2-12 ...]

Epoch 13/15
Training: 100%|████████| - loss: 0.645, acc: 68.91%
Validating: 100%|████████| - loss: 0.678, acc: 70.42%
  (No improvement, early stopping in 2 epochs)

============================================================
PHASE 1 vs PHASE 1.5 COMPARISON
============================================================

Overall Accuracy:
  Phase 1:    68.42%
  Phase 1.5:  70.15%
  Improvement: +1.73%

Per-Class Recall:
Class          Phase 1    Phase 1.5      Change
──────────────────────────────────────────────
angry            65.23%     66.14%      +0.91%
disgust          31.45%     39.62%      +8.17% ← SUCCESS!
fear             58.23%     61.85%      +3.62%
happy            82.15%     80.94%      -1.21%
neutral          72.31%     73.45%      +1.14%
sad              68.45%     70.12%      +1.67%
surprise         74.23%     75.31%      +1.08%

============================================================
DECISION
============================================================

[OK] Phase 1.5 improves or maintains performance!
>>> RECOMMENDATION: Use Phase 1.5 model as new baseline

[Save] Phase 1.5 model saved as new best model
```

---

## ⚠️ If Something Goes Wrong

**Accuracy drops significantly?**
→ Reduce `LEARNING_RATE` to 0.00005

**Minority classes not improving?**
→ Increase disgust weight in `calculate_class_weights()`

**Validation loss increases?**
→ Increase `PATIENCE` to 7

**GPU memory error?**
→ Reduce `BATCH_SIZE` to 16

---

## 📚 File Structure After Phase 1.5

```
Psychologist AI/
├── training/
│   ├── model.py
│   ├── preprocessing.py
│   ├── train_emotion_model.py
│   └── train_phase_1_5_finetune.py          ← NEW
├── models/face_emotion/
│   ├── emotion_cnn_best.pth                ← UPDATED (if good results)
│   ├── emotion_cnn_phase15_best.pth         ← NEW (backup)
│   └── config.json
├── reports/
│   ├── phase15_evaluation_<timestamp>.json  ← NEW
│   └── phase15_comparison_<timestamp>_comparison.png ← NEW
├── PHASE_1_README.md
├── PHASE_1_5_README.md                      ← NEW
└── PHASE_1_5_SETUP_COMPLETE.md              ← NEW (this file)
```

---

## 🎓 Why This Works

**The Problem:**
- Dataset has 16.5x more happy samples than disgust
- Standard training optimizes for majority class
- Model becomes useless for rare emotions

**The Solution:**
- Weighted loss: "Disgust errors cost 16.5x more"
- Frozen layers: "Keep what we learned, fine-tune emotions"
- Lower LR: "Gently nudge, don't redesign"
- Short training: "Refinement, not rediscovery"

**Result:**
- Better balance between all emotion classes
- Improved real-world performance on all emotions
- Still maintains overall accuracy

---

## 🔗 Related Files

- `PHASE_1_README.md` - Phase 1 details
- `PHASE_1_5_README.md` - Full Phase 1.5 strategy & docs
- `check_system.py` - Verify all files exist
- `requirements.txt` - Dependencies

---

## ✨ Summary

**Phase 1.5 is:**
- ✅ Conservative & safe (doesn't redesign model)
- ✅ Fast to run (15 epochs max)
- ✅ Automated (automatic decision to keep/revert)
- ✅ Well-documented (detailed logs & reports)
- ✅ Easy to customize (adjust Config class)

**Ready to fine-tune!**

```bash
python training/train_phase_1_5_finetune.py
```

---

**Status:** ✅ Ready  
**Time to Run:** 7-40 minutes (GPU-dependent)  
**Risk Level:** ✅ Low (Phase 1 backup preserved)  
**Expected Outcome:** +3-6% minority class improvement
