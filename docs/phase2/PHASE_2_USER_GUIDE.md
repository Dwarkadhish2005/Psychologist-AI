# 🎙️ Phase 2: Voice Emotion Recognition - User Guide

## Quick Start

### 1️⃣ Train Voice Emotion Models

```powershell
# Navigate to training directory
cd "C:\Dwarka\Machiene Learning\Psycologist AI"

# Train voice emotion and stress models
python training/voice/train_voice_emotion.py
```

**What happens:**
- Loads 4,240 audio samples (RAVDESS + TESS)
- Extracts 48-dimensional acoustic features
- Trains 2 models: Emotion (5 classes) + Stress (3 levels)
- Uses GPU acceleration (RTX 3050 6GB)
- Saves best models to `models/voice_emotion/`

**Expected training time:** ~30-60 minutes with GPU
**Expected accuracy:** 
- Emotion: ~60-70%
- Stress: ~70-80%

---

### 2️⃣ Real-Time Microphone Detection

```powershell
# Install audio library (if needed)
pip install sounddevice

# Run real-time detection
python inference/microphone_emotion_detection.py
```

**What happens:**
- Opens microphone stream
- Processes 3-second audio windows
- Extracts features in real-time
- Predicts emotion + stress level
- Shows live results

**Controls:**
- Speak into your microphone
- Press `Ctrl+C` to stop
- Shows emotion distribution at the end

---

## 📊 Phase 2 System Architecture

```
Audio Input (Microphone/File)
   ↓
Preprocessing (silence removal, normalization, noise reduction)
   ↓
Feature Extraction (48 features: pitch, energy, MFCCs, jitter, shimmer)
   ↓
┌─────────────────────┐     ┌─────────────────────┐
│ Emotion Model       │     │ Stress Detector     │
│ (5 classes)         │     │ (3 levels)          │
│ angry, fear, happy, │     │ low, medium, high   │
│ neutral, sad        │     │                     │
└─────────────────────┘     └─────────────────────┘
   ↓                            ↓
   Emotion + Confidence         Stress + Confidence
   ↓                            ↓
        Combined Output
        (for Phase 3 fusion)
```

---

## 🎯 Phase 2 Outputs

Phase 2 produces **parallel signals** to Phase 1 (face):

| Output | Values | Description |
|--------|--------|-------------|
| **Emotion** | angry, fear, happy, neutral, sad | What emotion is expressed in voice |
| **Stress Level** | low, medium, high | Stress/arousal level (separate from emotion) |
| **Emotion Confidence** | 0.0 - 1.0 | How confident the emotion prediction is |
| **Stress Confidence** | 0.0 - 1.0 | How confident the stress prediction is |

---

## 💡 Key Concepts

### Emotion vs Stress

**Important:** Stress ≠ Emotion!

You can be:
- 😊 Happy + 😰 Stressed
- 😐 Neutral + 😰 Stressed  
- 😢 Sad + 😌 Calm

This is why they're modeled separately.

### Feature Extraction

Voice emotion uses **acoustic features**, not raw audio:

**Emotional Features (what is felt):**
- Pitch (F0) and pitch variance
- Energy and intensity
- Speaking rate

**Stress Indicators (vocal strain):**
- Jitter (pitch instability)
- Shimmer (amplitude variation)
- Spectral flatness

**Spectral Features (voice timbre):**
- 13 MFCCs (Mel-frequency cepstral coefficients)
- Spectral centroid and rolloff

---

## 🔬 Phase 1 + Phase 2 Interaction

This is where the magic happens! 🪄

| Situation | Face | Voice | Insight |
|-----------|------|-------|---------|
| Masking emotion | 😊 happy | 😰 stressed | Forcing a smile |
| Suppressed anger | 😐 neutral | 😡 angry | Holding back anger |
| Emotional regulation | 😢 sad | 😐 neutral | Controlling emotions |
| High confidence | 😨 fear | 😨 fear | Agreement = real fear |

**Phase 3 (Fusion) will:**
- Detect agreement vs conflict
- Compute dominance (face vs voice)
- Generate multi-modal insights

---

## 📦 Dataset Information

**RAVDESS (Primary):**
- 1,440 audio samples
- 24 actors (12 male, 12 female)
- 8 emotions (mapped to 5)
- Professional recordings

**TESS (Secondary):**
- 2,800 audio samples  
- 2 speakers (older/younger female)
- 7 emotions (mapped to 5)
- Clear emotion separation

**Combined:**
- **4,240 total samples**
- **26 speakers**
- **Perfectly balanced:** 336 samples per emotion
- **Speaker-independent splits:** No leakage!

---

## 🎛️ Advanced Usage

### Test Individual Components

```powershell
# Test preprocessing
python training/voice/audio_preprocessing.py

# Test feature extraction
python training/voice/feature_extraction.py

# Test model architecture
python training/voice/voice_emotion_model.py

# Test label mapping
python training/voice/dataset_utils.py

# Check dataset status
python training/voice/download_datasets.py
```

### Custom Training Parameters

Edit `train_voice_emotion.py`:

```python
class Config:
    BATCH_SIZE = 32          # Batch size
    LEARNING_RATE = 0.001    # Learning rate
    NUM_EPOCHS = 50          # Max epochs
    EARLY_STOPPING_PATIENCE = 10  # Early stopping
    USE_CACHE = True         # Cache features (faster)
```

### Re-prepare Dataset

```python
from training.voice.download_datasets import prepare_combined_dataset

# Equal sampling (default)
train, val, test = prepare_combined_dataset(
    use_ravdess=True,
    use_tess=True,
    balance_strategy='equal'
)

# Weighted loss (keep all data)
train, val, test = prepare_combined_dataset(
    balance_strategy='weighted'
)
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'sounddevice'"

```powershell
pip install sounddevice
```

### Issue: "CUDA out of memory"

Reduce batch size in `train_voice_emotion.py`:

```python
BATCH_SIZE = 16  # Instead of 32
```

### Issue: "Models not found" during inference

Make sure you've trained first:

```powershell
python training/voice/train_voice_emotion.py
```

Models should be saved to:
- `models/voice_emotion/emotion_model_best.pth`
- `models/voice_emotion/stress_model_best.pth`

### Issue: Microphone not working

Test your microphone:

```powershell
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Set default input device in Windows Sound Settings.

### Issue: Training very slow

- **Enable GPU:** Check `python check_gpu.py`
- **Enable caching:** Set `USE_CACHE = True` in config
- **Reduce data:** Use smaller dataset initially

---

## 📈 Expected Performance

**Emotion Recognition:**
- Neutral: 70-80% (most common)
- Happy: 65-75%
- Sad: 60-70%
- Angry: 60-70%
- Fear: 55-65% (hardest)

**Stress Detection:**
- Low: 75-85%
- Medium: 70-80%
- High: 75-85%

**Why voice is harder than face:**
- Voice emotions overlap more
- Background noise affects features
- Speaker-dependent variations
- Temporal nature (emotion changes)

---

## ✅ Checklist: Is Phase 2 Working?

- [ ] Dataset prepared: 4,240 samples loaded
- [ ] Balanced training set: 336 per emotion
- [ ] Models trained: emotion + stress models saved
- [ ] Emotion accuracy: > 60%
- [ ] Stress accuracy: > 70%
- [ ] Microphone detection: Real-time emotion shown
- [ ] GPU acceleration: Training uses CUDA
- [ ] Feature caching: Second epoch faster than first

---

## 🚀 Next Steps: Phase 3 (Fusion)

Once Phase 2 is complete, Phase 3 will:

1. **Combine face + voice signals**
2. **Detect agreement vs conflict**
3. **Compute multi-modal confidence**
4. **Generate insights** (masking, suppression, etc.)
5. **Build personality profile** over time

---

## 📞 Quick Commands Reference

```powershell
# Train models
python training/voice/train_voice_emotion.py

# Real-time microphone
python inference/microphone_emotion_detection.py

# Check GPU
python check_gpu.py

# Test components
python training/voice/audio_preprocessing.py
python training/voice/feature_extraction.py
python training/voice/voice_emotion_model.py

# Re-prepare dataset
cd training/voice
python -c "from download_datasets import prepare_combined_dataset; prepare_combined_dataset()"
```

---

**Phase 2 Status:** ✅ READY TO TRAIN!

Dataset prepared, models designed, all systems go! 🎙️🚀
