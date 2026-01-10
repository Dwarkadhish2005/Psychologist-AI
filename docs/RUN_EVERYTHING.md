# 🚀 Complete System Guide: Face + Voice Emotion Recognition

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Phase 1: Face Emotion Recognition](#phase-1-face-emotion-recognition)
3. [Phase 2: Voice Emotion Recognition](#phase-2-voice-emotion-recognition)
4. [Running Everything Together](#running-everything-together)
5. [Quick Command Reference](#quick-command-reference)

---

## 🎯 System Overview

**Psychologist AI** is a multi-modal emotion recognition system that analyzes:
- **Phase 1:** Face emotions (7 classes)
- **Phase 2:** Voice emotions (5 classes) + Stress levels (3 levels)
- **Phase 3:** Multi-modal fusion (coming soon)

**Current Status:**
- ✅ Phase 1: Complete (trained models ready)
- ✅ Phase 2: Complete (dataset ready, ready to train)
- 🔄 Phase 3: Not started

---

## 🎭 Phase 1: Face Emotion Recognition

### What Phase 1 Does

Detects emotions from facial expressions:
- **7 emotions:** angry, disgust, fear, happy, neutral, sad, surprise
- **2 models:** 
  - Phase 1 (Main): 62.57% accuracy - general purpose
  - Phase 1.5 (Specialist): 61.26% disgust recall - minority expert
- **Dual-model strategy:** Smart routing for better accuracy

### Phase 1 Commands

#### 1. Test Webcam Detection (Single Model)

```powershell
python inference/webcam_emotion_detection.py
```

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot
- Press `p` to pause/resume

#### 2. Test Dual-Model Detection (Recommended)

```powershell
python inference/dual_model_emotion_detection.py
```

**Advantages:**
- Uses Phase 1 for initial prediction
- Consults Phase 1.5 when disgust/fear detected
- Better accuracy on minority classes

#### 3. Check Models

Models are located at:
- `models/face_emotion/emotion_cnn_best.pth` (Phase 1 - Main)
- `models/face_emotion/emotion_cnn_phase15_specialist.pth` (Phase 1.5 - Specialist)

---

## 🎙️ Phase 2: Voice Emotion Recognition

### What Phase 2 Does

Detects emotions and stress from voice:
- **5 emotions:** angry, fear, happy, neutral, sad
- **3 stress levels:** low, medium, high
- **Parallel signals:** Complements face emotion (not replacement)

### Phase 2 Commands

#### 1. Train Voice Models (First Time Only)

```powershell
python training/voice/train_voice_emotion.py
```

**What happens:**
- Loads 4,240 audio samples (RAVDESS + TESS)
- Extracts 48-dimensional features
- Trains emotion model (54,917 params)
- Trains stress detector (323 params)
- Uses GPU acceleration
- Saves models to `models/voice_emotion/`

**Training time:** ~30-60 minutes with GPU

**Expected results:**
- Emotion accuracy: ~60-70%
- Stress accuracy: ~70-80%

#### 2. Real-Time Microphone Detection

```powershell
# Install audio library first (if needed)
pip install sounddevice

# Run detection
python inference/microphone_emotion_detection.py
```

**What happens:**
- Opens microphone
- Processes 3-second audio windows
- Shows emotion + stress in real-time
- Press `Ctrl+C` to stop

---

## 🎬 Running Everything Together

### Scenario 1: Face Only (Webcam)

```powershell
# Simple single-model detection
python inference/webcam_emotion_detection.py

# OR dual-model (better accuracy)
python inference/dual_model_emotion_detection.py
```

### Scenario 2: Voice Only (Microphone)

```powershell
# Train first (if not done yet)
python training/voice/train_voice_emotion.py

# Then run detection
python inference/microphone_emotion_detection.py
```

### Scenario 3: Face + Voice Together (Manual)

**Terminal 1 (Face):**
```powershell
python inference/dual_model_emotion_detection.py
```

**Terminal 2 (Voice):**
```powershell
python inference/microphone_emotion_detection.py
```

**Watch both outputs:**
- Face window shows facial emotion
- Terminal shows voice emotion + stress
- Compare results for insights!

### Scenario 4: Complete System (Phase 3 - Coming Soon)

Phase 3 will combine face + voice into single output:
```powershell
# Coming soon:
python inference/multimodal_emotion_detection.py
```

**Will provide:**
- Combined emotion with higher confidence
- Agreement detection (face ≈ voice)
- Conflict detection (face ≠ voice)
- Insights (masking, suppression, etc.)

---

## 🔍 Quick Command Reference

### Check System Status

```powershell
# Check GPU
python check_gpu.py

# Check Python environment
python --version
pip list | findstr "torch librosa"

# List available models
dir models\face_emotion
dir models\voice_emotion
```

### Phase 1 (Face) Commands

```powershell
# Webcam detection (single model)
python inference/webcam_emotion_detection.py

# Webcam detection (dual model - better)
python inference/dual_model_emotion_detection.py

# Re-train Phase 1 (optional)
python training/train_emotion_model.py

# Re-train Phase 1.5 (optional - minority expert)
python training/train_phase_1_5_finetune.py
```

### Phase 2 (Voice) Commands

```powershell
# Train models (first time)
python training/voice/train_voice_emotion.py

# Microphone detection
python inference/microphone_emotion_detection.py

# Test individual components
python training/voice/audio_preprocessing.py
python training/voice/feature_extraction.py
python training/voice/voice_emotion_model.py
python training/voice/dataset_utils.py
```

### Utility Commands

```powershell
# Re-prepare voice dataset
cd training/voice
python -c "from download_datasets import prepare_combined_dataset; prepare_combined_dataset()"

# Check dataset status
python training/voice/download_datasets.py

# Install missing dependencies
pip install librosa noisereduce soundfile sounddevice
```

---

## 📊 System Capabilities

### Phase 1 (Face) Output

```python
{
    'emotion': 'happy',          # Detected emotion
    'confidence': 0.85,          # Confidence score
    'all_probabilities': [...],  # All class probabilities
    'model_used': 'Phase 1',     # Which model predicted
    'face_detected': True        # Whether face was found
}
```

**7 Emotions:** angry, disgust, fear, happy, neutral, sad, surprise

### Phase 2 (Voice) Output

```python
{
    'emotion': 'happy',              # Detected emotion
    'emotion_confidence': 0.75,      # Emotion confidence
    'stress_level': 'medium',        # Stress level
    'stress_confidence': 0.82,       # Stress confidence
    'overall_confidence': 0.785,     # Average confidence
    'emotion_probabilities': [...],  # Emotion probs
    'stress_probabilities': [...]    # Stress probs
}
```

**5 Emotions:** angry, fear, happy, neutral, sad  
**3 Stress Levels:** low, medium, high

### Phase 1 + Phase 2 Insights

| Face | Voice | Stress | Interpretation |
|------|-------|--------|----------------|
| 😊 happy | 😰 fear | high | **Masking emotion** - forcing smile |
| 😐 neutral | 😡 angry | high | **Suppressed anger** - holding back |
| 😢 sad | 😐 neutral | low | **Emotional regulation** - controlling emotions |
| 😨 fear | 😨 fear | high | **High confidence** - agreement = real fear |
| 😊 happy | 😊 happy | low | **Genuine happiness** - congruent signals |

---

## 🎓 Training Status

### Phase 1 Models (Already Trained)

- ✅ **emotion_cnn_best.pth** - Main model (62.57% accuracy)
- ✅ **emotion_cnn_phase15_specialist.pth** - Minority expert (+30% disgust!)
- **Training data:** FER2013 dataset (28,709 train images)
- **GPU:** Trained on RTX 3050 6GB

### Phase 2 Models (Ready to Train)

- 🔄 **emotion_model_best.pth** - Voice emotion (5 classes)
- 🔄 **stress_model_best.pth** - Stress detector (3 levels)
- **Training data:** RAVDESS + TESS (4,240 audio samples)
- **GPU:** Will use RTX 3050 6GB (~30-60 min)

**To train Phase 2:**
```powershell
python training/voice/train_voice_emotion.py
```

---

## 🔧 Troubleshooting

### GPU Not Working

```powershell
# Check CUDA availability
python check_gpu.py

# Should show:
# CUDA Available: True
# GPU: NVIDIA GeForce RTX 3050 6GB
```

If not working:
1. Reinstall PyTorch with CUDA: See `GPU_SETUP_GUIDE.md`
2. Check NVIDIA drivers: `nvidia-smi`

### Webcam Not Opening

1. Check camera permissions (Windows Settings → Privacy → Camera)
2. Close other apps using camera (Zoom, Teams, etc.)
3. Try different camera index in code: `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)`

### Microphone Not Working

```powershell
# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Set default in Windows Sound Settings
```

### Training Too Slow

**Phase 1 (Face):**
- Should use GPU automatically
- ~10-15 it/s with GPU
- ~5-6 it/s with CPU

**Phase 2 (Voice):**
- Enable feature caching: `USE_CACHE = True` (default)
- First epoch slow (extracting features)
- Subsequent epochs much faster (using cache)
- ~30-60 min total with GPU

### Models Not Found

**Phase 1:**
- Check: `models/face_emotion/emotion_cnn_best.pth`
- If missing: Download from releases or retrain

**Phase 2:**
- Must train first: `python training/voice/train_voice_emotion.py`
- Models saved to: `models/voice_emotion/`

---

## 📦 Project Structure

```
Psychologist AI/
├── data/
│   ├── fer2013/                      # Face emotion dataset
│   └── voice_emotion/                # Voice datasets
│       ├── RAVDESS/                  # 1,440 samples
│       ├── TESS/                     # 2,800 samples
│       └── dataset_splits.json       # Train/val/test splits
│
├── models/
│   ├── face_emotion/                 # Face models
│   │   ├── emotion_cnn_best.pth     # Phase 1 main
│   │   └── emotion_cnn_phase15_specialist.pth  # Phase 1.5
│   └── voice_emotion/                # Voice models
│       ├── emotion_model_best.pth   # Emotion (5 classes)
│       └── stress_model_best.pth    # Stress (3 levels)
│
├── training/
│   ├── train_emotion_model.py       # Phase 1 training
│   ├── train_phase_1_5_finetune.py  # Phase 1.5 training
│   └── voice/                        # Phase 2 training
│       ├── audio_preprocessing.py
│       ├── feature_extraction.py
│       ├── voice_emotion_model.py
│       ├── dataset_utils.py
│       ├── download_datasets.py
│       └── train_voice_emotion.py
│
├── inference/
│   ├── webcam_emotion_detection.py           # Phase 1 single model
│   ├── dual_model_emotion_detection.py       # Phase 1 dual model
│   └── microphone_emotion_detection.py       # Phase 2 microphone
│
└── Documentation/
    ├── PHASE_1_README.md             # Phase 1 guide
    ├── PHASE_1_5_README.md           # Phase 1.5 guide
    ├── PHASE_2_USER_GUIDE.md         # Phase 2 guide (this file)
    ├── GPU_SETUP_GUIDE.md            # GPU setup
    └── RUN_EVERYTHING.md             # This master guide
```

---

## ✅ System Checklist

### Phase 1 (Face)
- [x] FER2013 dataset downloaded
- [x] Phase 1 model trained (62.57% accuracy)
- [x] Phase 1.5 model trained (+30% disgust recall)
- [x] Webcam detection working
- [x] Dual-model detection working
- [x] GPU acceleration enabled

### Phase 2 (Voice)
- [x] RAVDESS dataset downloaded (1,440 samples)
- [x] TESS dataset downloaded (2,800 samples)
- [x] Dataset prepared (4,240 samples, balanced)
- [x] Speaker-independent splits created
- [x] Label mapping implemented
- [x] Feature extraction working (48 features)
- [x] Model architecture ready
- [ ] **→ TRAIN MODELS NOW!** ← Run: `python training/voice/train_voice_emotion.py`
- [ ] Microphone detection working

### System Requirements
- [x] Python 3.13 installed
- [x] PyTorch 2.6.0+cu124 (CUDA-enabled)
- [x] NVIDIA RTX 3050 6GB detected
- [x] All dependencies installed
- [x] GPU ready and working

---

## 🎯 Next Steps

### Immediate (Phase 2)

1. **Train Voice Models:**
   ```powershell
   python training/voice/train_voice_emotion.py
   ```
   ⏱️ Takes ~30-60 minutes

2. **Test Microphone Detection:**
   ```powershell
   python inference/microphone_emotion_detection.py
   ```

3. **Compare Face vs Voice:**
   - Run face detection in one terminal
   - Run voice detection in another terminal
   - Observe agreement and conflicts!

### Future (Phase 3)

1. **Multi-modal Fusion:**
   - Combine face + voice predictions
   - Weighted confidence scoring
   - Agreement/conflict detection

2. **Insight Generation:**
   - Detect masking emotion (smile + stress)
   - Detect suppressed emotions (neutral face + angry voice)
   - Emotional regulation patterns

3. **Personality Profiling:**
   - Track emotions over time
   - Build personality model
   - Detect behavioral patterns

---

## 📞 Support

### Documentation Files

- `PHASE_1_README.md` - Face emotion details
- `PHASE_1_5_README.md` - Minority class specialist
- `PHASE_2_USER_GUIDE.md` - Voice emotion details
- `GPU_SETUP_GUIDE.md` - GPU troubleshooting
- `RUN_EVERYTHING.md` - This master guide

### Quick Test Commands

```powershell
# Test everything is working
python check_gpu.py                                    # GPU
python training/voice/audio_preprocessing.py           # Audio
python training/voice/feature_extraction.py            # Features
python training/voice/voice_emotion_model.py           # Models
python inference/webcam_emotion_detection.py           # Webcam (press 'q')
```

---

**System Status:** ✅ Phase 1 Complete | 🔄 Phase 2 Ready to Train

**Next Action:** Run `python training/voice/train_voice_emotion.py` to complete Phase 2!

🚀 **Let's build a complete emotion recognition system!** 🎭🎙️
