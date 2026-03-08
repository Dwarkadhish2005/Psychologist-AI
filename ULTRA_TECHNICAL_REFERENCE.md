# PSYCHOLOGIST AI — ULTRA TECHNICAL REFERENCE
## Every Term, Method, Architecture, Layer, Function, Decision & Statistic

> **Version:** Complete — Updated March 8, 2026  
> **Created:** February 26, 2026  
> **Coverage:** All 5 Phases, All Layers, All Code Blocks, All Design Decisions  
> **Purpose:** Exhaustive reference for interviews, research, and deep implementation understanding  
> **Last Update:** Face model upgraded to EmotionCNNDeep (64.42% / F1=0.633); Voice model fixed (54% from collapsed 23%)

---

## TABLE OF CONTENTS

1. [Project Identity & Mission](#1-project-identity--mission)
2. [Complete Tech Stack](#2-complete-tech-stack)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Phase 1 & 1.5 — Face Emotion Detection](#4-phase-1--15--face-emotion-detection)
5. [Phase 2 — Voice Emotion & Stress Detection](#5-phase-2--voice-emotion--stress-detection)
6. [Phase 3 — Multi-Modal Fusion & Psychological Reasoning](#6-phase-3--multi-modal-fusion--psychological-reasoning)
7. [Phase 4 — Long-Term Cognitive Layer](#7-phase-4--long-term-cognitive-layer)
8. [Phase 5 — Personality Engine & Visualization](#8-phase-5--personality-engine--visualization)
9. [Data Pipeline & Datasets](#9-data-pipeline--datasets)
10. [Training Pipelines In Detail](#10-training-pipelines-in-detail)
11. [Inference System In Detail](#11-inference-system-in-detail)
12. [User Management System](#12-user-management-system)
13. [Memory System Architecture](#13-memory-system-architecture)
14. [Performance Statistics & Benchmarks](#14-performance-statistics--benchmarks)
15. [Design Decisions & Alternatives Rejected](#15-design-decisions--alternatives-rejected)
16. [Algorithms & Mathematical Formulations](#16-algorithms--mathematical-formulations)
17. [Data Structures & Enums](#17-data-structures--enums)
18. [File-by-File Breakdown](#18-file-by-file-breakdown)

---

## 1. PROJECT IDENTITY & MISSION

### What Is Psychologist AI?

A **production-ready, multi-modal AI system** that mimics the observational capabilities of a human therapist. It simultaneously analyzes:

- **Face** — What emotion does the person's face show?
- **Voice** — What emotion does their voice tone and prosody carry?
- **Stress** — What is the physiological stress level?

Then it fuses all three signals, detects contradictions (e.g., smiling face + stressed voice = emotional masking), infers a **mental state** from 15 possible states, assigns a **risk level** (4 levels), and builds a **long-term personality profile** using a 5-dimensional **Personality State Vector (PSV)** that updates across sessions.

### Core Mission

> Democratize mental health support by building an AI that:
> - Detects emotions through face, voice, and text
> - Maintains long-term memory of interactions and emotional patterns
> - Provides personalized therapeutic responses
> - Tracks psychological progress over time

### Design Philosophy

| Principle | Details |
|-----------|---------|
| **State ≠ Trait** | A momentary emotion is not a personality. The system distinguishes between short-lived states and long-term traits. |
| **Baseline > Absolute** | Personalized thresholds — a person‐specific "normal" is more meaningful than population-level norms |
| **No Diagnosis** | The system uses probabilistic behavioral descriptors, NOT clinical labels. It never says "you have depression." |
| **Explainable AI** | Every prediction comes with human-readable reasons |
| **Reliability > Sensitivity** | Conservative predictions are preferred; avoid false alarms |

---

## 2. COMPLETE TECH STACK

### Hardware & Environment

| Component | Specification |
|-----------|-------------|
| CUDA | 12.4 |
| Python | 3.12+ |
| OS | Windows (primary), Linux-compatible |
| GPU | NVIDIA (RTX recommended) |
| RAM | 8GB+ recommended |

### Deep Learning Frameworks

| Library | Version | Role | Why Chosen Over Alternatives |
|---------|---------|------|-------------------------------|
| **PyTorch** | 2.6.0 | Primary DL framework for all models | Over TensorFlow: Pythonic API, easier debugging, dynamic computation graph, better research community adoption. Over Keras standalone: more control. |
| **torchvision** | latest | Image transforms, pretrained models | Native PyTorch ecosystem |
| **torchaudio** | latest | Audio processing in PyTorch pipeline | Consistent PyTorch tensor operations |

### Computer Vision

| Library | Version | Role | Why Chosen |
|---------|---------|------|-----------|
| **OpenCV (cv2)** | ≥4.8.0 | Face detection (Haar Cascade), image I/O, real-time webcam | Industry standard for CV, extremely fast, Haar Cascade built-in |
| **Pillow (PIL)** | ≥10.0.0 | Image loading, conversion between formats | Best interoperability with torchvision transforms |
| **MediaPipe** | ≥0.10.0 | Pose/gesture detection | Google's production-ready landmark detection |

### Audio Processing

| Library | Version | Role | Why Chosen |
|---------|---------|------|-----------|
| **librosa** | ≥0.10.0 | All audio feature extraction (MFCC, pitch, energy, spectral) | The most comprehensive Python audio analysis library; scientific-grade feature extraction |
| **soundfile** | ≥0.12.0 | Reading/writing audio files in various formats | Easier than scipy.io.wavfile for multi-format support |

### NLP & Language Models

| Library | Version | Role | Why Chosen |
|---------|---------|------|-----------|
| **Transformers (HuggingFace)** | ≥4.30.0 | Pre-trained LMs for text sentiment | Unmatched model zoo; BERT, RoBERTa, DistilBERT out-of-the-box |
| **NLTK** | ≥3.8.0 | Text preprocessing, tokenization | Standard toolkit; lightweight |
| **spaCy** | ≥3.5.0 | Linguistic analysis, NER, POS tagging | Faster than NLTK for production; better for structured text pipelines |

### Data Science & ML Utilities

| Library | Role |
|---------|------|
| **NumPy ≥1.24.0** | All numerical arrays, matrix operations, statistics |
| **Pandas ≥2.0.0** | Data manipulation, CSV export/import, session logs |
| **scikit-learn ≥1.3.0** | `confusion_matrix`, `classification_report`, `train_test_split`, `StandardScaler`, encoders |
| **SciPy** | Signal processing (FFT, filters for audio) |

### Visualization

| Library | Role | Why |
|---------|------|-----|
| **Matplotlib ≥3.7.0** | Training curves, confusion matrices, PSV radar charts | Fully customizable; dominant scientific Python viz |
| **Seaborn ≥0.12.0** | Statistical plots, heatmaps | Higher-level API over matplotlib; beautiful defaults |
| **Plotly ≥5.15.0** | Interactive PSV dashboards, personality timelines | Only library for interactive browser-based charts in Python without JS |

### Development & Infrastructure

| Tool | Role |
|------|------|
| **pytest** | Comprehensive testing framework for all phases |
| **logging** | Structured diagnostic output throughout all modules |
| **pathlib** | OS-agnostic file path management (used everywhere instead of `os.path`) |
| **dataclasses** | Clean data structures without boilerplate (`@dataclass` decorator) |
| **collections.deque** | Fixed-length temporal memory buffer (O(1) append/pop from both ends) |
| **hashlib (SHA-256)** | User ID generation from name + timestamp |
| **json** | Serialization of all user memory, PSV, configs |
| **tqdm** | Progress bars in training loops |
| **TensorBoard ≥2.13.0** | Training monitoring (loss curves, weight histograms) |
| **wandb ≥0.15.0** | Experiment tracking (optional but listed) |

### Serving Infrastructure (Listed)

| Tool | Role |
|------|------|
| **Flask ≥2.3.0** | Lightweight HTTP API server |
| **FastAPI ≥0.100.0** | Async REST API for production |
| **uvicorn ≥0.23.0** | ASGI server for FastAPI |

---

## 3. SYSTEM ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│                     PSYCHOLOGIST AI SYSTEM                       │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  WEBCAM      │  │  MICROPHONE  │  │  TEXT INPUT (Future)  │  │
│  │  30 FPS      │  │  Real-time   │  │                       │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬────────────┘  │
│         │                 │                       │               │
│  ┌──────▼───────┐  ┌──────▼──────────────┐       │               │
│  │ PHASE 1      │  │ PHASE 2              │       │               │
│  │ Face Emotion │  │ Voice Emotion        │       │               │
│  │ CNNDeep      │  │ + Stress Detection   │       │               │
│  │ 7 classes    │  │ FC Net / 5 classes   │       │               │
│  │ ~6M params   │  │ + 3 stress levels    │       │               │
│  └──────┬───────┘  └──────┬──────────────┘       │               │
│         │                 │                       │               │
│  ┌──────▼─────────────────▼───────────────────────▼────────────┐ │
│  │                      PHASE 3                                  │ │
│  │             MULTI-MODAL FUSION ENGINE                         │ │
│  │  Layer 1: Signal Normalization (reliability weighting)        │ │
│  │  Layer 2: Temporal Reasoning (30-frame deque memory)          │ │
│  │  Layer 3: Fusion Logic (psychology-rule-based)                │ │
│  │  Layer 4: Psychological State Inference (15 states)           │ │
│  │  Output: PsychologicalState (dominant + hidden emotion,       │ │
│  │          mental state, risk level, confidence, stability)     │ │
│  └─────────────────────────┬─────────────────────────────────────┘ │
│                            │                                        │
│  ┌─────────────────────────▼─────────────────────────────────────┐ │
│  │                      PHASE 4                                   │ │
│  │             LONG-TERM COGNITIVE LAYER                          │ │
│  │  SessionMemory  →  LongTermMemory  →  PersonalityProfile       │ │
│  │  BaselineProfile  →  DeviationDetector                         │ │
│  └─────────────────────────┬─────────────────────────────────────┘ │
│                            │                                        │
│  ┌─────────────────────────▼─────────────────────────────────────┐ │
│  │                      PHASE 5                                   │ │
│  │             PERSONALITY ENGINE + VISUALIZATION                 │ │
│  │  PersonalityStateVector (PSV) — 5 traits                       │ │
│  │  Radar Charts, Trend Lines, Bar Charts                         │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Five-Phase Development Timeline

| Phase | Milestone | Status | Key Output |
|-------|-----------|--------|------------|
| 1 | Face Emotion Detection | ✅ Complete | 64.42% accuracy, 7 emotion classes, EmotionCNNDeep |
| 1.5 | Fine-tuning & Dual-Model Strategy | ✅ Complete | Minority class booster (specialist model) |
| 2 | Voice Emotion + Stress | ✅ Complete | ~54% voice accuracy, macro F1=0.508 |
| 3 | Multi-Modal Fusion | ✅ Complete | 9/10 scenario tests, 15 mental states |
| 4 | Cognitive Layer (Memory + Personalization) | ✅ Complete | Long-term user profiles |
| 5 | Personality Engine + Visualization | ✅ Complete | 5D PSV, radar charts |

---

## 4. PHASE 1 & 1.5 — FACE EMOTION DETECTION

### 4.1 Dataset: FER-2013

| Property | Value |
|----------|-------|
| **Name** | FER-2013 (Facial Expression Recognition 2013) |
| **Total Images** | ~35,887 |
| **Image Size** | 48×48 pixels, grayscale |
| **Classes** | 7: angry, disgust, fear, happy, sad, surprise, neutral |
| **Source** | Kaggle / ICML 2013 workshop |
| **Format** | CSV with pixel values OR folder structure |
| **Splits** | train / val / test |

**Why FER-2013?**
- The most widely benchmarked face emotion dataset → enables comparison with literature
- Free, well-curated, diverse in age/gender/lighting
- 48×48 grayscale format is computationally cheap → enables real-time inference

**Why not AffectNet or RAF-DB?**
- AffectNet has 1M+ images — excellent but requires more compute
- RAF-DB has real-world images but limited free access
- FER-2013 is sufficient for a production prototype

**Class Imbalance (known challenge):**
- `happy` and `neutral` are over-represented (~9K–8K images each)
- `disgust` is severely under-represented (~600 images)
- This is the motivation for Phase 1.5 (dual-model / fine-tuning strategy)

---

### 4.2 Model Architecture: `EmotionCNN` (Phase 1 — Baseline, now used only as specialist)

**Location:** `training/model.py`

```
Input: (batch=128, channels=1, H=48, W=48) — Grayscale images

Block 1: Conv2D(1→32, kernel=3×3, padding=1) → BatchNorm2D(32) → ReLU → MaxPool(2×2)
         Output: (32, 24, 24)

Block 2: Conv2D(32→64, kernel=3×3, padding=1) → BatchNorm2D(64) → ReLU → MaxPool(2×2)
         Output: (64, 12, 12)

Block 3: Conv2D(64→128, kernel=3×3, padding=1) → BatchNorm2D(128) → ReLU → MaxPool(2×2)
         Output: (128, 6, 6)

Flatten: 128 × 6 × 6 = 4608

FC1: Linear(4608 → 256) → ReLU
Dropout: p=0.5

FC2 (Output): Linear(256 → num_classes)
              Returns raw logits; applied Softmax at inference
```

**Total Parameters:** ~300,000 (~300K) — used as specialist model (`emotion_cnn_phase15_specialist.pth`)

**Key Design Decisions:**

| Decision | Reason | Alternative Rejected |
|----------|--------|---------------------|
| 3 conv blocks with doubling filters (32→64→128) | Standard pattern for progressively abstract feature detection | 4+ blocks would over-parameterize for 48×48 input |
| kernel_size=3×3 | Optimal balance of receptive field vs computation; factorizes large kernels | 5×5 or 7×7 would be too large for 48px images |
| padding=1 ("same" padding) | Preserves spatial dimensions after each conv | Zero padding would shrink feature maps prematurely |
| MaxPool(2×2) after each block | Reduces spatial size by half, increases translation invariance | AveragePool: MaxPool better for texture/edge detection |
| BatchNorm2D after each conv | Stabilizes training, acts as regularizer, enables higher LR | LayerNorm: BatchNorm designed for CNNs |
| Dropout(p=0.5) only in FC layers | Prevent overfitting in the densely connected part | Spatial dropout in conv layers (unnecessary at this scale) |
| Grayscale input (1 channel) | Emotion is in shape/texture, not color; reduces params by 3× | RGB input: adds 2 unnecessary channels |
| `num_classes` auto-detected | Flexible for different emotion sets | Hardcoded classes: limits reuse |

---

### 4.3 Main Model: `EmotionCNNDeep` (Active — `emotion_cnn_best.pth`)

Deeper VGG-style architecture adopted as the primary face model.

```
Block 1: Conv(1→64, k=3, pad=1) → Conv(64→64, k=3, pad=1) → BN → ReLU → MaxPool(2×2)
         Output: (64, 24, 24)
Block 2: Conv(64→128, k=3, pad=1) → Conv(128→128, k=3, pad=1) → BN → ReLU → MaxPool(2×2)
         Output: (128, 12, 12)
Block 3: Conv(128→256, k=3, pad=1) → Conv(256→256, k=3, pad=1) → BN → ReLU → MaxPool(2×2)
         Output: (256, 6, 6)

Flatten: 256 × 6 × 6 = 9216
FC1: Linear(9216 → 512) → ReLU → Dropout(0.5)
FC2: Linear(512 → 256) → ReLU → Dropout(0.3)
FC3 (Output): Linear(256 → num_classes)  ← raw logits

Total Params: 5,997,383 (~6M)
```

**Why two conv layers per block?** — VGG-style stacking of two 3×3 convolutions gives an effective 5×5 receptive field with fewer parameters and more non-linearities than a single 5×5 conv.

**Why switch from EmotionCNN to EmotionCNNDeep?**
- EmotionCNN plateaued at ~51% accuracy — insufficient capacity for 7-class FER2013
- EmotionCNNDeep's 4× wider filters (64→128→256 vs 32→64→128) capture richer facial features
- The larger FC head (9216→512→256) provides more discriminative power before the classifier
- Result: **+13% absolute accuracy gain** (51% → 64.42%)

---

### 4.4 Preprocessing Pipeline

**Location:** `training/preprocessing.py`

**Training Transforms (augmentation):**
```python
transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),               # Faces are left-right symmetric
    transforms.RandomRotation(degrees=15),                 # Head tilt variation (upgraded from 10°)
    transforms.RandomAffine(shear=5, scale=(0.85, 1.15)), # Position/scale variance (upgraded)
    transforms.ColorJitter(brightness=0.3, contrast=0.3), # Lighting variation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),           # Normalize to [-1, 1]
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))   # Random occlusion robustness
])
```

**Validation/Inference Transforms (NO augmentation):**
```python
transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

**Why these specific augmentations?**

| Augmentation | Justification | Why NOT Stronger? |
|-------------|---------------|--------------------|
| `RandomHorizontalFlip(0.5)` | Human faces are bilaterally symmetric; expands dataset effectively | Vertical flip would be unnatural |
| `RandomRotation(10°)` | Real-world head tilts; prevents angle-sensitivity | >15° would cause class ambiguity (upside-down face ≠ angry) |
| `RandomAffine(translate=0.1, scale=0.9-1.1)` | Camera position/distance variance | Larger crop/scale would alter critical facial features |
| `Normalize([0.5], [0.5])` | Centers pixel distribution; helps gradient flow | ImageNet normalization (different mean) not appropriate for grayscale |

**Key function `preprocess_face()`:**
- Converts BGR (OpenCV default) → Grayscale
- Resize to 48×48 using `cv2.INTER_AREA` (best for shrinking)
- Normalize to [0,1]: `pixel / 255.0`
- Add channel dimension: `np.expand_dims(..., axis=0)` → shape `(1, H, W)`

---

### 4.5 Face Detection: Haar Cascade

**Method:** `cv2.CascadeClassifier` with `haarcascade_frontalface_default.xml`

**Parameters:**
- `scaleFactor=1.3` — Image pyramid scaling factor (30% size reduction per level)
- `minNeighbors=5` — Quality threshold (how many overlapping detections needed)
- `minSize=(30, 30)` — Minimum face size to detect

**Why Haar Cascade?**
- Bundled with OpenCV, zero extra dependencies
- Very fast (runs on CPU at 30+ FPS)
- Sufficient for frontal face detection in controlled environments

**Why NOT MTCNN or Dlib?**
- MTCNN: More accurate but significantly slower for real-time
- Dlib HOG: CPU-heavy, worse with partial occlusions
- RetinaFace: Production-grade but requires additional setup

**`extract_face_roi()` function:**
- Adds 10px padding around detected face
- Clips to image boundaries (`max(0, x - padding)`)
- Reason: Includes chin/forehead context; reduces clipping at edges


### 4.6 Training Configuration & Hyperparameters

| Hyperparameter | Value | Justification |
|---------------|-------|---------------|
| `BATCH_SIZE` | 128 | Large batch for faster CPU/GPU throughput; stable gradient estimate |
| `NUM_EPOCHS` | 80 | Deeper model needs more warmup; early stopping handles cutoff |
| `LEARNING_RATE` | 0.001 | Adam default; empirically validated for CNN classification |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization; prevents overfitting |
| `PATIENCE` (early stopping) | 15 | Increased from 10; deeper model has longer warmup before convergence |
| `OPTIMIZER` | Adam | Adaptive learning rates; better than SGD for CNNs on FER |
| `LOSS` | WeightedCrossEntropyLoss | Class weights inversely proportional to frequency; disgust weight=9.38× |
| `SCHEDULER` | ReduceLROnPlateau(mode='max', patience=5) | Steps on macro F1 improvement; reduces LR by 2× on plateau |
| `SAVE CRITERION` | Best val macro F1 (not accuracy) | Macro F1 penalizes per-class failures equally; avoids happy-class bias |

**Why Adam over SGD?**
- Adam: Adaptive per-parameter learning rates, handles sparse gradients, converges faster
- SGD: Requires careful LR tuning, momentum configuration
- For emotion recognition: Adam consistently outperforms in fewer epochs

**Why WeightedCrossEntropyLoss?**
- Base: `CrossEntropyLoss = log-softmax + NLL loss` — the standard for multi-class
- Class weights computed via `sklearn.utils.compute_class_weight('balanced')` on training labels
- This forces the model to penalize disgust misclassification 9.38× more than happy misclassification
- `FocalLoss` was considered but class weights + macro F1 saving achieved the same goal more simply

**Why macro F1 as saving criterion (not accuracy)?**
- On imbalanced datasets, accuracy is dominated by majority classes (happy: 5214 samples)
- A model that gets 100% happy and 0% disgust gets 73% accuracy but terrible F1
- Macro F1 averages per-class F1 equally → forces the model to improve on minority classes too
- Result: disgust recall improved from ~10% (accuracy-optimized) to 65% (F1-optimized)

---

### 4.7 Phase 1.5: Dual-Model Strategy

**Problem:** `disgust` class has only ~600 images → severely underperforms

**Strategy:**
1. **General Model** — Trained on all 7 classes, handles majority
2. **Specialist Model** — Fine-tuned specifically to distinguish minority classes (disgust, fear)

**Technique — Fine-tuning (`train_phase_1_5_finetune.py`):**
- Freeze early conv layers (preserve low-level features)
- Only update FC layers and final conv block
- Use higher class weights for minority classes in loss function

**Class Weighting:**
```python
class_weights = total_samples / (num_classes * class_counts_per_class)
```
This upweights disgust from ~0.17 weight to ~1.5+, forcing the model to pay attention.

---

### 4.8 Training Metrics Achieved

| Metric | Value |
|--------|-------|
| Overall Test Accuracy | **64.42%** |
| Best Val Accuracy | **64.97%** |
| Best Macro F1 | **0.633** |
| Best Class F1 | Happy: **0.86** |
| Worst Class F1 | Fear: **0.41** |
| Disgust Recall | **65%** (was near 0% without class weights) |
| Surprise Recall | **84%** |
| Training Dataset | FER2013: 20,749 train / 7,960 val / 7,178 test |
| Architecture | EmotionCNNDeep (5,997,383 params) |
| Literature Benchmark | ~65-70% state-of-the-art on FER2013 |
| Trained (date) | 2026-03-08 |

**Class weights applied:**
| Class | Weight | Train Count |
|-------|--------|-------------|
| angry | 1.027 | 2,887 |
| disgust | 9.380 | 316 |
| fear | 1.001 | 2,961 |
| happy | 0.568 | 5,214 |
| neutral | 0.826 | 3,588 |
| sad | 0.849 | 3,491 |
| surprise | 1.293 | 2,292 |

**Why 64.42% is a strong result:**
- FER2013 is notoriously noisy (human labelers disagree ~65% of the time)
- Inter-human agreement on FER2013 is ~65% → model approaches human performance
- Previous `EmotionCNN` model achieved only ~51% — this is a **+13% absolute improvement**
- The system uses **fusion** in Phase 3, so single-modality perfection is unnecessary

---

## 5. PHASE 2 — VOICE EMOTION & STRESS DETECTION

### 5.1 Datasets Used

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

| Property | Value |
|----------|-------|
| **Full Name** | Ryerson Audio-Visual Database of Emotional Speech and Song |
| **Size** | 1440 files (24 actors × 60 trials) |
| **Emotions** | Neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| **Format** | WAV, 48000 Hz |
| **Coding** | 7th character of filename = emotion (01=neutral, 02=calm, ...) |
| **Location** | `data/voice_emotion/RAVDESS/` |

**TESS (Toronto Emotional Speech Set)**

| Property | Value |
|----------|-------|
| **Size** | ~2800 files (2 actresses × 200 target words × 7 emotions) |
| **Emotions** | Angry, disgust, fear, happy, pleasant surprise, ps, sad |
| **Format** | WAV |
| **Location** | `data/voice_emotion/TESS/` |

**Combined:** 3000+ samples used after filtering and mapping to 5 target classes.

**Why RAVDESS + TESS?**
- RAVDESS: High quality, acted speech, multiple actors (gender diversity)
- TESS: Complements RAVDESS with different speaking styles and actors
- Combination improves class balance and diversity over either alone

**Why not EmoDB or IEMOCAP?**
- EmoDB: Only German speech — not generalizable to English
- IEMOCAP: Excellent but requires institutional access; complex annotation format

---

### 5.2 Audio Feature Extraction (48 Dimensions Total)

**Location:** `training/voice/feature_extraction.py`

The feature vector is the **input to the voice model** — sound is converted to a 48-dimensional numerical fingerprint.

#### Feature Family 1: Pitch Features (5 dims)
```
pitch_mean     — Average fundamental frequency (F0) in Hz
pitch_std      — Pitch variability
pitch_range    — Max - Min pitch
pitch_median   — Median pitch (robust to outliers)
pitch_slope    — Upward/downward trend over utterance
```
**Extraction:** `librosa.piptrack()` — STFT algorithm for pitch tracking
- `fmin=75 Hz`, `fmax=500 Hz` — human voice range
- Per-frame: select strongest pitch (max magnitude)

**Why pitch matters:**
- High pitch → excitement, fear, surprise
- Low pitch → sadness, calmness
- High variability → emotional intensity, anxiety

#### Feature Family 2: Energy Features (5 dims)
```
rms_mean    — Average RMS energy
rms_std     — Energy variability
rms_max     — Peak intensity
zcr_mean    — Zero-crossing rate mean
zcr_std     — Zero-crossing rate variability
```
**Extraction:** `librosa.feature.rms()`, `librosa.feature.zero_crossing_rate()`

**Why energy matters:**
- High energy → anger, happiness
- Low energy → sadness, fatigue
- Zero-crossing rate → voice activity; unvoiced vs voiced sounds

#### Feature Family 3: Spectral Features (32 dims)
```
mfcc_mean[1..13]              — 13 MFCC coefficient means
mfcc_std[1..13]               — 13 MFCC coefficient standard deviations
spectral_centroid_mean/std    — Brightness of voice
spectral_rolloff_mean/std     — High-frequency spread
spectral_flatness_mean/std    — Noisiness (tonal vs noisy)
```
**Extraction:** `librosa.feature.mfcc(n_mfcc=13)`

**MFCC (Mel-Frequency Cepstral Coefficients):**
- Standard voice features since 1980s → industry standard for ASR and voice emotion
- Convert waveform → STFT (spectrogram) → Mel filterbank (perceptual scale) → Log → DCT
- First 13 coefficients capture ~99% of voice timbral information
- Both mean and std extracted → static AND dynamic information

**Why 13 MFCCs?**
- Below 13: lose important spectral detail
- Above 20: diminishing returns + overfitting risk
- 13 is the standard in HTK, Kaldi and most voice processing literature

#### Feature Family 4: Temporal/Prosody Features (3 dims)
```
speaking_rate   — Speech onsets per second
num_onsets      — Total number of speech onset events
pause_ratio     — % of time silent
```

#### Feature Family 5: Stress-Specific Features (3 dims)
```
jitter          — Cycle-to-cycle pitch irregularity
shimmer         — Cycle-to-cycle amplitude irregularity
pitch_variance  — Erratic vs smooth pitch
```
**Why these 3 for stress?**
- `jitter` and `shimmer` are clinical biomarkers for vocal stress
- Stress causes microtremor in vocal folds → measurable jitter/shimmer
- `spectral_flatness` is excluded here — already captured in Family 3 as `spectral_flatness_mean/std`
- `pitch_variance` captures vocal instability under stress

---

### 5.3 Voice Emotion Model: `VoiceEmotionModel`

**Location:** `training/voice/voice_emotion_model.py`

```
Input: Feature vector (48 dims, float32)

FC Layer 1: Linear(48 → 256) → BatchNorm1D(256) → ReLU → Dropout(0.3)
FC Layer 2: Linear(256 → 128) → BatchNorm1D(128) → ReLU → Dropout(0.3)
FC Layer 3: Linear(128 → 64) → BatchNorm1D(64) → ReLU → Dropout(0.3)

Classifier: Linear(64 → 5)  ← 5 emotion classes
```

**Classes:** `['angry', 'fear', 'happy', 'neutral', 'sad']`

**Why FC layers instead of CNN/LSTM?**
- Audio features are already pre-extracted (48 numbers) — no spatial/temporal structure to exploit
- CNN needs 2D spatial data (spectrograms)
- LSTM needs sequential data (raw audio frames)
- Pre-extracted features → FC network is optimal, fast, and interpretable
- CNN on pre-extracted features would be meaningless (1D vector has no spatial relationships)

**BatchNorm1D:** Normalizes within a FC layer — stabilizes training when input feature scales vary wildly (pitch in 100-500 Hz range, MFCCs in -20 to +20 range)

---

### 5.4 Stress Detector: `StressDetector`

**Separate model from emotion detection** — Critical design decision

```
Input: Stress features (4 dims: jitter, shimmer, flatness, pitch_var)

FC: Linear(4 → 32) → BatchNorm1D → ReLU → Dropout(0.2)
Output: Linear(32 → 3)  ← 3 stress levels: [low, medium, high]
```

**Why a separate stress model?**
- Stress ≠ Emotion: You can be **happy + stressed** (excited) or **neutral + stressed** (worried)
- Conflating stress with emotion loses important clinical information
- Stress has dedicated physiological biomarkers (jitter, shimmer) separate from emotional content

**Why this model has `modality reliability = 0.9` in Phase 3?**
- Physiological markers (jitter, shimmer) are harder to consciously control
- Voice emotion can be performed/acted; stress response often cannot
- Clinical literature supports acoustic stress markers as highly reliable

---

### 5.5 Training for Voice: `train_voice_emotion_balanced.py`

**Full training pipeline:**
- **StandardScaler** — fitted on all 1,680 training samples; saved as `feature_scaler.pkl`; applied at ALL inference points
- `WeightedRandomSampler` — enforces equal class representation per mini-batch regardless of dataset imbalance
- `compute_class_weight('balanced')` — weighted cross-entropy loss as a second layer of imbalance correction
- **Noise augmentation** — Gaussian noise (std=0.1) added to training features to improve generalization
- `ReduceLROnPlateau(mode='max', patience=8, factor=0.5)` — scheduler steps on macro F1, not val loss
- **Saves on macro F1** — avoids happy-class bias in checkpoint criterion
- `lr=0.0003`, `weight_decay=5e-4`, `patience=20` (early stopping)
- Model architecture: hidden layers `[256, 128, 64]`, Dropout=0.5

**Critical preprocessing alignment (fixed):**
- Training uses `preprocess_audio(filepath, sr=16000)` — noise reduction + RMS normalize + preemphasis
- All inference paths (test, integrated inference, run_verify) now use identical pipeline
- This fixed a critical SR mismatch: original test loaded at sr=22050 causing completely wrong feature distributions
- `feature_scaler.pkl` is **required** at inference — raw 48-dim features have wildly different scales (pitch=100-500Hz, MFCCs=-20 to +20)

**Why `train_voice_emotion_balanced.py` over `train_voice_emotion.py`?**
- Original script had no StandardScaler, no WeightedRandomSampler, and `CosineAnnealingWarmRestarts` that reset LR just as training converged
- Balanced training prevents happy class (abundant) from dominating
- Results: Collapsed model (predicting 'sad' for 83% of inputs, 23% accuracy) → functional 5-class model

---

### 5.6 Performance Statistics

| Metric | Value |
|--------|-------|
| Overall Val Accuracy | **~54%** |
| Best Val Macro F1 | **0.508** |
| Training samples | 1,680 (336/class × 5 classes, perfectly balanced) |
| Val samples | 1,520 |
| Dataset | RAVDESS + TESS, mapped to 5 classes |
| Scaler | StandardScaler fitted on 1,680 training samples |

**Previous collapsed state (fixed):**
- Before fixes: 23% accuracy, predicting 'sad' for 83% of all inputs
- Root cause: SR mismatch (test used 22050 Hz, training used 16000 Hz) + no StandardScaler applied at inference
- After fixes: functional 5-class prediction, ~54% accuracy, macro F1=0.508

**Why ~54% is the honest ceiling for this setup:**
- Voice emotion is inherently harder than face emotion
- Same sentence in different emotional states can sound very similar
- RAVDESS/TESS are acted emotions — real emotion is more subtle
- Dataset ceiling: only 336 samples/class limits generalization significantly
- State-of-the-art on RAVDESS with engineered features is ~60-70%; our FC net on 48-dim features + small dataset lands at ~54%
- Adding raw spectrogram CNN or wav2vec embeddings would significantly improve this

---

## 6. PHASE 3 — MULTI-MODAL FUSION & PSYCHOLOGICAL REASONING

**Location:** `inference/phase3_multimodal_fusion.py`
**Lines:** 838

This is the most architecturally sophisticated phase. It implements a **4-layer fusion engine** that mirrors how a human psychologist integrates multiple cues.

### 6.1 Modality Reliability Weights

```python
class Modality(Enum):
    FACE  = ("face",  0.5)   # Medium — people mask facial emotions
    VOICE = ("voice", 0.7)   # High — tone is harder to control
    STRESS = ("stress", 0.9) # Very high — physiological signal
```

**Why these specific values?**

| Modality | Score | Reasoning |
|----------|-------|-----------|
| Face = 0.5 | Medium | People consciously control facial expressions (politeness, culture); high masking rate especially in professional contexts |
| Voice = 0.7 | High | Tone, pitch, pace are subconscious; harder to fake over extended conversation |
| Stress = 0.9 | Very high | Jitter/shimmer are physiological; cannot be consciously suppressed |

**Literature basis:** Paul Ekman's research on micro-expressions supports face < voice in emotional authenticity. Clinical voice stress analysis (used in aviation/military) validates stress marker reliability.

---

### 6.2 Layer 1: Signal Normalization (`SignalNormalizer`)

**Purpose:** Convert raw model outputs into comparable `NormalizedSignal` objects.

**`NormalizedSignal` dataclass:**
```python
modality       : Modality
emotion        : str
confidence     : float     # Raw model confidence
reliability    : float     # Modality base score (0.5/0.7/0.9)
signal_quality : float     # Dynamic quality based on detection conditions
timestamp      : float

# Derived property:
weighted_confidence = confidence × reliability × signal_quality
```

**Signal quality computation:**
- Face: `quality = 0.0` if face not detected; `0.3` if confidence < 0.3 threshold; else `min(1.0, confidence × 1.5)`
- Voice: `quality = min(1.0, confidence × audio_quality)`
- Stress: `quality = max(0.5, confidence)` — always at least 0.5 because stress signal is reliable even at lower confidence

**Emotion Mapping (cross-modal normalization):**
```python
EMOTION_MAPPING = {
    'disgust': 'angry',     # Disgust → anger mapping (similar physiological response)
    'surprise': 'neutral',  # Surprise is transient; treated as neutral for sustained analysis
}
```
**Why?** Face model has 7 classes, voice has 5. Disgust has no voice equivalent → mapped to closest neighbor. Surprise is a transient state → mapped to neutral to avoid false alarms.

---

### 6.3 Layer 2: Temporal Reasoning (`TemporalMemory`)

**Purpose:** Detect patterns over time, not just in a single frame.

**Core Structure:** `collections.deque(maxlen=30)` — 30-frame sliding window (≈1 second at 30fps)

**Why `deque` over `list`?**
- `deque.appendleft()` and `deque.popleft()` are O(1) vs O(n) for list
- `maxlen` automatically discards oldest elements
- Perfect for a fixed-length sliding window

**Temporal patterns detected:**

| Pattern | Detection Logic | Significance |
|---------|----------------|--------------|
| `emotion_switch` | Emotion changes within 5 frames | Emotional instability |
| `stress_persistence` | Stress stays high for 15+ frames | Chronic stress, not transient |
| `masking_event` | Face=happy, Voice/Stress=negative | Person hiding emotions |
| `instability` | >3 different emotions in 10 frames | Emotional dysregulation |
| `flat_affect` | Low confidence/expression for 20+ frames | Emotional flatness (depression marker) |

**Why 30 frames?**
- 30fps × 30 frames = 1 second of history
- Long enough to detect real patterns; short enough to respond to changes quickly
- Longer window (100 frames) would miss rapid emotional shifts

---

### 6.4 Layer 3: Fusion Logic (Rule-Based Psychology Engine)

**Approach:** Psychology-informed rule system, NOT a learned model

**Why rules instead of another neural network?**
- Interpretability: Rules produce human-readable explanations
- Data scarcity: Not enough labeled "fused emotion + ground-truth mental state" data to train
- Safety: Rules can be reviewed, audited, and corrected by psychologists
- Consistency: Same input always → same output (no stochastic neural variance)

**Core fusion rules (examples):**

| Condition | Output Mental State | Reasoning |
|-----------|--------------------|----|
| Face=happy, Stress=high | `HAPPY_UNDER_STRESS` | Classic emotional masking — smiling while internally stressed |
| Face=neutral, Voice=sad, Stress=high | `EMOTIONALLY_MASKED` | Suppressing sadness with neutral expression |
| All 3 signals = calm | `CALM` | Unanimous agreement |
| All 3 signals = stressed/negative | `OVERWHELMED` | Multi-modal distress confirmation |
| Rapid emotional switches + high stress | `EMOTIONALLY_UNSTABLE` | Pattern-based inference |
| Low expression across all modalities | `EMOTIONALLY_FLAT` | Potential depression/dissociation marker |

**Conflict resolution logic:**
```
High-reliability signals win over low-reliability signals
STRESS (0.9) > VOICE (0.7) > FACE (0.5)
```

---

### 6.5 Layer 4: Psychological State Inference

**Output Structure: `PsychologicalState`**

```python
@dataclass
class PsychologicalState:
    dominant_emotion   : str              # Primary emotion label
    hidden_emotion     : Optional[str]    # Masked emotion (if detected)
    mental_state       : MentalState      # One of 15 states (enum)
    confidence         : float            # 0-1
    explanations       : List[str]        # Human-readable reasoning
    risk_level         : RiskLevel        # LOW/MODERATE/HIGH/CRITICAL
    stability_score    : float            # 0-1 (0=unstable)
    temporal_patterns  : List[TemporalPattern]
    raw_signals        : Dict[str, NormalizedSignal]
    timestamp          : float
```

### 6.6 The 15 Mental States (`MentalState` Enum)

| State | Description | Trigger Conditions |
|-------|-------------|-------------------|
| `CALM` | Relaxed, no stress | All signals positive/neutral, low stress |
| `JOYFUL` | Active positive state | High face happy, voice happy, low stress |
| `HAPPY_UNDER_STRESS` | Smiling but internally stressed | Face=happy + stress=high |
| `ANXIOUS` | Worried, anticipatory | Voice=fear, stress=medium, rapid transitions |
| `STRESSED` | General stress state | Stress=medium/high without overwhelm |
| `OVERWHELMED` | Acute stress overload | All signals negative + stress=high |
| `CONFUSED` | Mixed/conflicting signals | No consistent pattern across modalities |
| `EMOTIONALLY_MASKED` | Hiding true emotions | Neutral face + negative voice/stress |
| `EMOTIONALLY_FLAT` | Reduced emotional expression | Low confidence/expression across all |
| `EMOTIONALLY_UNSTABLE` | Rapid emotional fluctuations | High volatility in temporal window |
| `DEPRESSED` | Sustained low/negative state | Persistent sadness + low energy patterns |
| `ANGRY` | Anger confirmed multi-modally | Face=angry + voice=angry |
| `FEARFUL` | Fear confirmed multi-modally | Face=fear + voice=fear |
| `MIXED_EMOTIONS` | Contradictory but clear signals | e.g., face=happy + voice=angry |
| `UNKNOWN` | Insufficient signal quality | Low confidence all modalities |

### 6.7 Risk Levels (`RiskLevel` Enum)

| Level | Description | Trigger |
|-------|-------------|---------|
| `LOW` | Normal range | No concerning patterns |
| `MODERATE` | Attention warranted | Persistent stress, some masking |
| `HIGH` | Requires support | Overwhelmed + emotional instability |
| `CRITICAL` | Intervention needed | Combination of extreme indicators |

### 6.8 Test Results

| Metric | Value |
|--------|-------|
| Scenario tests passed | **9/10** |
| Test scenarios | Synthetic (all mental states exercised) |
| Test location | `tests/test_phase3_final.py` |

---

## 7. PHASE 4 — LONG-TERM COGNITIVE LAYER

**Location:** `inference/phase4_cognitive_layer.py`
**Lines:** 2,724 (largest file)

### 7.1 Architecture: 6 Modules

```
Phase 3 (PsychologicalState)
         ↓
Module 1: SessionMemory      — Track states within one session (30-90 min)
         ↓
Module 2: LongTermMemory     — Aggregate sessions over days/weeks
         ↓
Module 3: PersonalityProfile — Infer stable traits from long-term patterns
         ↓
Module 4: BaselineProfile    — Establish personal "normal"
         ↓
Module 5: DeviationDetector  — Detect unusual behavior vs personal baseline
         ↓
Output: UserPsychologicalProfile
```

### 7.2 Module 1: `SessionMemory`

**Purpose:** Short-term tracking within a single session (one sitting, 30-90 minutes)

**Internal storage:**
```python
states: List[Tuple[float, PsychologicalState]]  # (timestamp, state) pairs
start_time: float
_last_mental_state: Optional[MentalState]
_state_switch_count: int
```

**`add_state(PsychologicalState)` method:**
- Appends to `states` list with current timestamp
- Tracks state transitions: increments `_state_switch_count` when mental state changes

**`calculate_metrics()` → `SessionMetrics` dataclass:**

This is the core computation method. It extracts:

| Metric Group | Fields |
|-------------|--------|
| Temporal | `session_start`, `session_duration`, `total_frames` |
| Mental States | `dominant_mental_states`, `mental_state_switches` |
| Confidence | `avg_confidence`, `avg_stability`, `confidence_variance`, `stability_variance` |
| Stress | `stress_duration_ratio`, `high_stress_duration_ratio`, `avg_stress_intensity` |
| Masking | `masking_frequency`, `total_masking_events`, `masking_duration_ratio` |
| Risk | `avg_risk_level`, `high_risk_duration_ratio`, `risk_escalations` |
| Emotional Polarity | `positive_emotion_ratio`, `negative_emotion_ratio`, `neutral_emotion_ratio` |

**Stress intensity mapping:**
```python
StressIntensity = {
    STRESSED: 0.5,
    OVERWHELMED: 1.0,
    ANXIOUS: 0.6,
    EMOTIONALLY_UNSTABLE: 0.8
}
```

**`is_active(timeout_minutes=5.0)`:**
- Returns `True` if activity within last 5 minutes
- Enables automatic session boundary detection

---

### 7.3 Module 2: `LongTermMemory`

**Purpose:** Cross-session storage and analysis (days to months)

**Storage structure:**
```
data/user_memory/{user_id}_longterm_memory.json
data/user_memory/archive/
```

**Data hierarchy:**
```
LongTermMemory
├── daily_profiles: Dict[date_str → DailyProfile]
└── weekly_aggregates: Dict[week_start → WeeklyAggregate]
```

**`DailyProfile` dataclass** summarizes all sessions from one day:
- Aggregated stress/masking/risk ratios
- Mental state distribution
- Confidence/stability trends
- Emotional polarity ratios

**`WeeklyAggregate` dataclass** summarizes 7 days:
- Weekly averages of all metrics
- Trend slopes (coefficient from linear regression): `stress_trend`, `masking_trend`, `risk_trend`, `stability_trend`
- Positive slope = metric worsening; negative = improving

**Persistence:**
- `load()` — Loads from JSON on initialization
- `save()` — Serializes with `@dataclass → asdict()` then `json.dump`
- `max_days_stored=90` — Prunes old data automatically
- `archive_dir` — Old data moved here, not deleted

**Why JSON over SQL/SQLite?**
- No database dependency
- Human-readable for debugging
- Perfect for hierarchical profile data
- Future: Can migrate to SQLite if scale demands it

---

### 7.4 Module 3: `PersonalityProfile`

**Infers stable personality traits from long-term patterns**

**Key insight:** `STATE ≠ TRAIT`
- Session state: "anxious today" (temporary)
- Personality trait: "tends toward anxiety" (requires weeks of data)

**Trait inference requires minimum N sessions** (configurable, typically 5+) to become statistically meaningful.

---

### 7.5 Module 4: `BaselineProfile`

**Purpose:** Establish "what's normal for THIS user"

**Why this matters:**
- A naturally stoic person showing slight sadness is a deviation
- A naturally expressive person showing the same sadness is normal
- Population averages are misleading for individual mental health

**Baseline established from first N sessions** (typically 3-5 sessions):
- Baseline stress level, masking frequency, emotional volatility
- Used by `DeviationDetector`

---

### 7.6 Module 5: `DeviationDetector`

**Purpose:** Anomaly detection vs personal baseline

**Detection method:** Z-score comparison
```
z_score = (current_value - baseline_mean) / baseline_std
```
- `|z| > 2.0` → notable deviation
- `|z| > 3.0` → significant deviation (flag for clinical attention)

**Why Z-score over absolute thresholds?**
- Personalized: 50% stress ratio is alarming for person A (baseline=10%), normal for person B (baseline=45%)
- Self-calibrating: Adapts as user provides more data

---

## 8. PHASE 5 — PERSONALITY ENGINE & VISUALIZATION

**Location:** `inference/phase5_personality_engine.py` + `inference/phase5_visualization.py`

### 8.1 Personality State Vector (PSV)

**The core innovation of Phase 5** — A numerical fingerprint of long-term behavioral tendencies.

```python
@dataclass
class PersonalityStateVector:
    emotional_stability : float = 0.5   # 0=unstable, 1=consistent
    stress_sensitivity  : float = 0.5   # 0=resilient, 1=highly reactive
    recovery_speed      : float = 0.5   # 0=slow, 1=fast
    positivity_bias     : float = 0.5   # 0=negative-leaning, 1=positive-leaning
    volatility          : float = 0.5   # 0=stable, 1=highly dynamic
    
    # Metadata
    last_updated               : str
    total_sessions_processed   : int
    confidence                 : float   # 0-1 based on data quantity
    
    # History (last 10 updates per trait)
    emotional_stability_history  : List[float]
    stress_sensitivity_history   : List[float]
    recovery_speed_history       : List[float]
    positivity_bias_history      : List[float]
    volatility_history           : List[float]
```

All trait values normalized to **[0, 1]** — not Big Five raw scores, but behavioral frequency ratios.

### 8.2 Trait Formulas

| Trait | Formula |
|-------|---------|
| `emotional_stability` | `1 - variance(emotional_states_per_session)` |
| `stress_sensitivity` | `avg(stress_increase_rate) + stress_reaction_probability` |
| `recovery_speed` | `1 / avg(time_to_return_to_baseline)` |
| `positivity_bias` | `positive_time_ratio / total_observation_time` |
| `volatility` | `total_emotional_transitions / total_observation_time` |

### 8.3 Update Strategy: Exponential Decay Weighting

**Why exponential decay?**
- Recent sessions matter more than sessions from 3 months ago
- Prevents permanent "labeling" from one bad week

```python
weight = exp(-decay_rate × sessions_ago)
new_trait = old_trait × (1 - learning_rate) + new_observation × learning_rate
```
- `learning_rate`: Small value (e.g., 0.1) — PSV changes slowly
- Consequence: Need 10+ sessions before traits reach 80% confidence

**Why slow updates?**
- Prevents a single panic attack from permanently labeling someone as "stress-sensitive"
- Mirrors clinical psychology practice: one observation ≠ pattern

### 8.4 Trend Analysis (`get_trait_trends()`)

Uses **linear regression** on last 5 updates:
```python
slope = np.polyfit(x=range(len(history)), y=history[-5:], deg=1)[0]

if slope > 0.02:   → "increasing"
elif slope < -0.02: → "decreasing"
else:               → "stable"
```

### 8.5 Confidence Levels

| Confidence | Sessions Required | Label |
|-----------|------------------|-------|
| < 0.2 | 0-2 sessions | very_low |
| 0.2-0.4 | 3-5 sessions | low |
| 0.4-0.6 | 6-10 sessions | moderate |
| 0.6-0.8 | 11-20 sessions | high |
| > 0.8 | 20+ sessions | very_high |

### 8.6 Behavioral Descriptors (Ethical AI)

`get_behavioral_descriptor()` generates text like:
> "demonstrates consistent emotional patterns, with notable stress resilience, and demonstrates quick emotional recovery"

**Key ethical constraints:**
- ❌ Never says "you have anxiety disorder"
- ❌ Never uses DSM-5/ICD-11 diagnostic labels
- ❌ Never makes absolute claims
- ✅ Uses probabilistic language: "tends to," "shows patterns of"
- ✅ Contextual: "in observed sessions"

### 8.7 Visualization: `phase5_visualization.py`

#### PSV Radar Chart (`create_psv_radar_chart`)
- **Purpose:** Show all 5 traits simultaneously on a polar chart
- **Library:** `matplotlib` with `projection='polar'`
- **Shape:** Pentagon (5 traits, equidistant angles)
- Angles: `np.linspace(0, 2π, 5, endpoint=False)`
- Polygon closed by appending first value to end
- `set_theta_offset(π/2)`: Start from top (12 o'clock position)
- `set_theta_direction(-1)`: Clockwise (standard psychology convention)
- Saved at DPI=300 for print quality

#### PSV Trend Line Chart (`create_psv_trend_chart`)
- Shows all 5 traits evolving over update sessions
- Each trait = different colored line
- X-axis = update number; Y-axis = trait value [0,1]
- Enables "is stress sensitivity increasing over time?" question

#### PSV Bar Chart (`create_psv_bar_chart`)
- Horizontal bars for current trait values
- Optional trend arrows (↑↓→) from `get_trait_trends()`

**Why Matplotlib over Plotly for PSV charts?**
- Matplotlib: Saves to file (PNG, PDF) for reports
- Plotly: Listed for interactive dashboards
- Both are available; default uses matplotlib for offline use

---

## 9. DATA PIPELINE & DATASETS

### 9.1 Face Data Structure

```
data/face_emotion/
├── train/
│   ├── angry/    (images)
│   ├── disgusted/
│   ├── fearful/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
├── val/
└── test/

data/emotion_face/   (secondary dataset, same structure)
```

### 9.2 Voice Data Structure

```
data/voice_emotion/
├── RAVDESS/          (WAV files, coded filenames)
├── TESS/             (WAV files, emotion-named folders)
├── feature_cache/    (Pre-computed 48-dim vectors)
├── dataset_metadata.json
└── dataset_splits.json
```

**Feature caching:**
- For 3000+ audio files, re-extracting features per training run wastes hours
- `feature_cache/` stores pre-computed 48-dim numpy arrays
- Keyed by filename; invalidated if feature extraction code changes

### 9.3 User Data Structure

```
data/user_memory/
├── users.json                      ← User registry
├── {name}_{id}_longterm_memory.json ← Per-user Phase 4 data
├── {name}_{id}_psv.json            ← Per-user Phase 5 PSV
└── archive/                        ← Old data preserved

data/demo_memory/    ← Demo users (alice, bob) for testing
data/test_memory/    ← Test users for pytest
```

**User ID format:** `{sanitized_name}_{sha256_hash_12chars}`
Example: `alice_90806d704641`
- Hash ensures uniqueness (two "Alice" users get different IDs)
- SHA-256 is cryptographically secure for ID generation

---

## 10. TRAINING PIPELINES IN DETAIL

### 10.1 Face Training (`train_emotion_model.py`)

**End-to-end training loop:**
```
1. Instantiate Config (hyperparameters)
2. Create EmotionDataset (auto-detects classes from folder names)
3. Create DataLoaders (train=128 batch/8 workers, val/test=4 workers, persistent_workers=True)
4. Instantiate EmotionCNNDeep (num_classes auto-detected)
5. optimizer = Adam(lr=0.001, weight_decay=1e-4)
6. criterion = WeightedCrossEntropyLoss (weights from compute_class_weight('balanced'))
7. scheduler = ReduceLROnPlateau(mode='max', factor=0.5, patience=5)  # steps on macro F1
8. For epoch in range(80):
   a. train_one_epoch() → loss, acc
   b. validate() → val_loss, val_acc
   c. Compute val_macro_f1 via full val pass (sklearn f1_score, average='macro')
   d. scheduler.step(val_macro_f1)
   e. If val_macro_f1 improved: save model checkpoint → emotion_cnn_best.pth
   f. If no improvement for 15 epochs: early stop
9. evaluate_model() → test acc, confusion matrix, classification report
10. Save: model weights, config.json (includes architecture + best_macro_f1 + test_acc), plots
```

**Config.json saved fields (used by all consumers):**
```json
{
  "architecture": "EmotionCNNDeep",
  "num_classes": 7,
  "class_names": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
  "best_val_acc": 64.97,
  "best_macro_f1": 0.6330,
  "test_acc": 64.42,
  "timestamp": "20260308_165826"
}
```

**Dynamic model loading (all consumers):**
All downstream consumers (`test_face_emotion.py`, `integrated_psychologist_ai.py`, `run_verify.py`) read `architecture` from `config.json` and instantiate the correct class dynamically:
```python
_ARCH_MAP = {'EmotionCNNDeep': EmotionCNNDeep, 'EmotionCNN': EmotionCNN}
ModelClass = _ARCH_MAP.get(config['architecture'], EmotionCNN)
model = ModelClass(num_classes=..., input_size=48)
```

**`train_one_epoch()` inner loop:**
```python
for batch in dataloader:
    optimizer.zero_grad()      # Clear gradients
    outputs = model(inputs)    # Forward pass
    loss = criterion(outputs, labels)  # Compute loss
    loss.backward()            # Backpropagation (auto-grad)
    optimizer.step()           # Update weights
```

**Model checkpointing:**
- `torch.save(model.state_dict(), path)` — saves only weights, not architecture
- `torch.load()` + `model.load_state_dict()` — restores
- Best val **macro F1** checkpoint saved (not last epoch, not val accuracy)

**Training history saved to JSON:**
```json
{
  "train_loss": [...],
  "train_acc": [...],
  "val_loss": [...],
  "val_acc": [...],
  "val_macro_f1": [...],
  "best_epoch": 47,
  "best_val_acc": 64.97,
  "best_macro_f1": 0.6330
}
```

### 10.2 Voice Training (`train_voice_emotion_balanced.py`)

**Key differences from face training:**
1. Input = pre-extracted 48-dim feature vectors, not raw images
2. Uses `WeightedRandomSampler` for class balancing
3. Trains both `VoiceEmotionModel` AND `StressDetector` separately

**WeightedRandomSampler:**
```python
class_counts = Counter(labels)
weights = 1.0 / class_counts[label] for each sample
sampler = WeightedRandomSampler(weights, num_samples=len(train_set))
```
Each class appears proportionally equal regardless of dataset imbalance.

**StandardScaler (critical for voice):**
```python
scaler = StandardScaler()
scaler.fit(all_train_features)          # Fit on 1680 training samples
joblib.dump(scaler, 'feature_scaler.pkl')
# At inference:
features = scaler.transform(raw_features.reshape(1, -1)).flatten()
```
Without the scaler, pitch features (100-500 Hz range) and MFCCs (-20 to +20 range) have incompatible scales — the model never sees normalized inputs at inference.

**Audio preprocessing pipeline (applied identically at train + inference):**
```python
def preprocess_audio(filepath, sr=16000):
    y, _ = librosa.load(filepath, sr=16000)
    y = nr.reduce_noise(y=y, sr=16000)   # Noise reduction
    y = y / (np.sqrt(np.mean(y**2)) + 1e-8)  # RMS normalize
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])  # Preemphasis
    return y
```

---

## 11. INFERENCE SYSTEM IN DETAIL

### 11.1 `integrated_psychologist_ai.py` — The Complete Real-Time System

Orchestrates all phases simultaneously:

```
Thread 1 (main): Webcam capture → face detection → Phase 1 CNN → NormalizedSignal
Thread 2 (audio): Microphone capture → feature extraction → Phase 2 FC Net → NormalizedSignal
Fusion (main): Phase 3 engine → PsychologicalState → Phase 4 memory → Phase 5 PSV
Display: OpenCV window with overlays
```

### 11.2 `webcam_emotion_detection.py`

**Core loop:**
```python
cap = cv2.VideoCapture(0)   # Open webcam
while True:
    ret, frame = cap.read()
    faces = detect_faces_haar(frame)
    for face in faces:
        roi = extract_face_roi(frame, face)
        tensor = preprocess_face_torch(roi)
        with torch.no_grad():
            probs = model.predict(tensor.to(device))
        emotion = class_names[probs.argmax()]
        cv2.putText(frame, emotion, ...)  # Overlay
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) == ord('q'): break
cap.release()
```

**`torch.no_grad()` during inference:**
- Disables gradient computation — halves memory usage
- ~2× faster than with gradients
- Critical for real-time performance at 30 FPS

### 11.3 `microphone_emotion_detection.py`

```
pyaudio capture (2-second chunks) → librosa processing →
48-dim feature vector → VoiceEmotionModel → emotion + confidence
                                          → StressDetector → stress level
```

### 11.4 `dual_model_emotion_detection.py` (Phase 1.5)

Runs both general and specialist models:
```
Input frame
   ↓
Model A (general 7-class): [0.3, 0.6, 0.05, 0.02, 0.01, 0.01, 0.01]
Model B (minority specialist): [0.4 disgust, 0.3 fear, ...]

Ensemble: Weighted average, with higher weight for specialist on minority classes
Final: Most confident combined prediction
```

---

## 12. USER MANAGEMENT SYSTEM

**Location:** `inference/phase4_user_manager.py`

### 12.1 `UserProfile` Dataclass

```python
user_id       : str   # e.g., "alice_90806d704641"
name          : str   # Display name
created_at    : str   # ISO datetime
last_active   : str   # ISO datetime
total_sessions: int
```

### 12.2 `UserManager` Class

| Method | Purpose |
|--------|---------|
| `register_user(name)` | Creates SHA-256 based ID, saves to `users.json` |
| `_generate_user_id(name, timestamp)` | `sha256(f"{name}_{timestamp}").hexdigest()[:12]` |
| `get_user(user_id)` | Lookup by ID |
| `list_users()` | Sorted by `last_active` (most recent first) |
| `update_last_active(user_id)` | Called on each session start |
| `increment_session_count(user_id)` | Called on session end |
| `delete_user(user_id, delete_data=True)` | Removes from registry + optionally deletes files |

**Persistence:** `_load_users()` on init, `_save_users()` after every mutation.

### 12.3 `UserSelector` — Interactive CLI

```
=== USER SELECTION ===
1. alice (last active: 2026-01-10)
2. bob (last active: 2026-01-08)
3. Create new user
0. Exit
```

---

## 13. MEMORY SYSTEM ARCHITECTURE

### Data Flow Through Time

```
Frame t=0: PsychologicalState
Frame t=1: PsychologicalState
...
Frame t=N: PsychologicalState
              ↓ SessionMemory.calculate_metrics()
          SessionMetrics (one per session, ~30-90 min)
              ↓ LongTermMemory.add_session()
          DailyProfile (one per day)
              ↓ (after 7 days)
          WeeklyAggregate
              ↓ (after 5+ sessions)
          PersonalityProfile (inferred traits)
              ↓
          PersonalityStateVector (PSV) — 5-dim float vector
```

### Memory Files Per User

```
{user_id}_longterm_memory.json:
{
  "daily_profiles": {
    "2026-01-10": { ...DailyProfile... },
    "2026-01-11": { ...DailyProfile... }
  },
  "weekly_aggregates": {
    "2026-01-06": { ...WeeklyAggregate... }
  }
}

{user_id}_psv.json:
{
  "emotional_stability": 0.72,
  "stress_sensitivity": 0.45,
  "recovery_speed": 0.61,
  "positivity_bias": 0.58,
  "volatility": 0.34,
  "confidence": 0.65,
  "total_sessions_processed": 12,
  "emotional_stability_history": [0.5, 0.55, 0.6, 0.68, 0.72],
  ...
}
```

---

## 14. PERFORMANCE STATISTICS & BENCHMARKS

### Accuracy Summary

| Model/System | Accuracy | Notes |
|-------------|----------|-------|
| Face Emotion (EmotionCNNDeep) | **64.42%** | FER2013; human agreement ≈ 65%; +13% vs previous model |
| Face Macro F1 | **0.633** | Saved on F1 criterion |
| Face Best Class F1 | **0.86** (Happy) | |
| Face Worst Class F1 | **0.41** (Fear) | |
| Face Disgust Recall | **65%** | Class weight 9.38× |
| Face Surprise Recall | **84%** | |
| Voice Emotion (VoiceEmotionModel) | **~54% val** | RAVDESS + TESS, post-fix |
| Voice Macro F1 | **0.508** | |
| Phase 3 Scenario Tests | **9/10 passed** | Synthetic testing scenarios |
| Stress Detection | High reliability (rule-aided) | Jitter/shimmer biomarkers |

### Inference Latency (Estimated)

| Component | Latency |
|-----------|---------|
| Face detection (Haar) | ~5ms (CPU) |
| Face emotion (EmotionCNN, GPU) | ~2ms |
| Feature extraction (librosa) | ~50ms per 2-sec chunk |
| Voice emotion (FC Net, CPU) | <1ms |
| Phase 3 fusion | <1ms |
| Total frame-to-prediction | ~60-80ms |

### Model Sizes

| Model | Parameters | File Size | Status |
|-------|-----------|-----------|--------|
| EmotionCNNDeep (face main) | 5,997,383 | 22.9 MB | ✅ Active |
| EmotionCNN (face specialist) | ~300K | 4.9 MB | ✅ Active (minority class booster) |
| VoiceEmotionModel [256,128,64] | ~70K | 226 KB | ✅ Active |
| StressDetector | ~2K | 5 KB | ✅ Active |
| feature_scaler.pkl | — | 1.7 KB | ✅ Required at inference |

---

## 15. DESIGN DECISIONS & ALTERNATIVES REJECTED

### Why PyTorch over TensorFlow?

| Criterion | PyTorch | TensorFlow | Decision |
|-----------|---------|-----------|---------|
| Debugging | Easy (Pythonic, eager execution) | Harder (graph mode legacy) | PyTorch |
| Research adoption | 70%+ of research papers | Decreasing | PyTorch |
| Dynamic graphs | ✅ Native | ✅ In TF2, but older code | PyTorch |
| Deployment | TorchScript, ONNX | TFLite, SavedModel | Tie |

### Why Rule-Based Fusion over Learned Fusion?

| Criterion | Rule-Based (chosen) | Learned Neural Fusion | Decision |
|-----------|--------------------|-----------------------|---------|
| Interpretability | ✅ Full explanations | ❌ Black box | Rule-based |
| Data requirements | 0 | 10K+ labeled multimodal samples | Rule-based |
| Modification by psychologist | ✅ Simple rule edits | ❌ Requires retraining | Rule-based |
| Consistency | ✅ Deterministic | ❌ Stochastic | Rule-based |
| Performance ceiling | Limited by rules | Potentially higher | Learned (future) |

### Why deque over list for temporal memory?

- `list.pop(0)` = O(n) — shifts all elements
- `deque.popleft()` = O(1) — doubly-linked list
- At 30fps, executing 30 O(n) operations per second would degrade performance

### Why not ResNet/EfficientNet for face emotion?

| Criterion | Custom EmotionCNN | ResNet-50 |
|-----------|-----------------|-----------|
| Inference speed | ~2ms | ~10ms |
| Parameters | 300K | 25M |
| FER2013 accuracy | 62.57% | ~65-68% |
| Real-time suitability | ✅ 30 FPS capable | ❌ Struggles on CPU |

For a 48×48 grayscale image, ResNet's full depth adds ~5% accuracy but 80× more parameters — not worth it for real-time use.

### Why 48×48 image resolution?

- FER2013 standard (dataset images are 48×48)
- Sufficient for emotion-relevant facial features (eyes, mouth, brow)
- Higher resolution doesn't help when original data is 48×48
- Upsizing to 224×224 for ResNet would be artificial padding, not real information

### Why MFCC over raw spectrogram for voice?

| Feature Type | Dims | Information | Requires GPU |
|-------------|------|-------------|-------------|
| Raw waveform | 44100/sec | Too high-dimensional | CNN + LSTM |
| Spectrogram | 128×T | Spatial | 2D CNN |
| MFCC | 13 | Perceptual, compressed | FC Net only |

MFCC is the standard for voice since 1980 — it compresses audio into the dimensions most relevant to human perception (mimicking the cochlear frequency response).

---

## 16. ALGORITHMS & MATHEMATICAL FORMULATIONS

### MFCC Extraction Pipeline

```
1. Pre-emphasis filter:  s'[n] = s[n] - 0.97 × s[n-1]
   (boosts high frequencies; compensates for vocal tract dampening)

2. Framing: 25ms frames, 10ms hop (standard STFT parameters)

3. Windowing: Hamming window to reduce spectral leakage
   w[n] = 0.54 - 0.46 cos(2πn/N)

4. FFT: Compute power spectrum

5. Mel filterbank: 26 triangular filters on mel scale
   Mel = 2595 × log10(1 + f/700)
   (perceptual frequency scale; humans hear logarithmically)

6. Log: Apply log to mel energies (perceptual loudness)

7. DCT: Discrete Cosine Transform → decorrelates coefficients
   Final output: 13 MFCC coefficients per frame
```

### Modality-Weighted Confidence

```
weighted_confidence = confidence × reliability × signal_quality

Where:
  confidence      = model softmax probability [0,1]
  reliability     = FACE=0.5, VOICE=0.7, STRESS=0.9
  signal_quality  = dynamic quality factor [0,1]
```

### PSV Exponential Decay Update

```
PSV_new[trait] = PSV_old[trait] × (1 - α) + new_observation × α

Where α = learning_rate (e.g., 0.1)

With exponential weighting of past sessions:
weight[session_i] = exp(-λ × sessions_since_i)
normalized_weight = weight[i] / sum(all_weights)
```

### Z-Score Deviation Detection

```
z = (x - μ_baseline) / σ_baseline

Thresholds:
  |z| < 1.0  → Normal variation
  |z| > 2.0  → Notable deviation
  |z| > 3.0  → Significant deviation (alert)
```

### Mental State Stability Score

```
stability_score = 1 - (emotional_variance / max_possible_variance)

Where emotional_variance = var(mental_state_encodings_over_window)
max_possible_variance = max possible state change
```

### Linear Trend Slope for PSV Trends

```python
x = np.arange(len(history[-5:]))
y = history[-5:]
slope = np.polyfit(x, y, deg=1)[0]   # First-degree polynomial coefficient

if slope > 0.02:   → "increasing"
elif slope < -0.02: → "decreasing"
else:               → "stable"
```

---

## 17. DATA STRUCTURES & ENUMS

### Complete Enum List

```python
# Phase 3
class Modality(Enum):    FACE, VOICE, STRESS
class RiskLevel(Enum):   LOW, MODERATE, HIGH, CRITICAL
class MentalState(Enum): CALM, STRESSED, HAPPY_UNDER_STRESS, EMOTIONALLY_MASKED,
                         ANXIOUS, OVERWHELMED, EMOTIONALLY_FLAT, JOYFUL, FEARFUL,
                         ANGRY_STRESSED, SAD_DEPRESSED, CONFUSED, STABLE_NEGATIVE,
                         STABLE_POSITIVE, EMOTIONALLY_UNSTABLE

# Phase 4 (implicit in logic)
# Risk numeric mapping: LOW=0, MODERATE=1, HIGH=2, CRITICAL=3
```

### Complete Dataclass List

| Class | Phase | Key Fields | Purpose |
|-------|-------|-----------|---------|
| `NormalizedSignal` | 3 | modality, emotion, confidence, reliability, signal_quality, timestamp | Layer 1 output |
| `TemporalPattern` | 3 | pattern_type, duration, intensity, description | Layer 2 output |
| `PsychologicalState` | 3 | dominant_emotion, hidden_emotion, mental_state, confidence, explanations, risk_level, stability_score | Layer 4 output |
| `SessionMetrics` | 4 | 20+ aggregated fields | Session summary |
| `DailyProfile` | 4 | date, sessions, stress/masking/risk metrics | Day summary |
| `WeeklyAggregate` | 4 | week dates, averages, trend slopes | Week summary |
| `PersonalityStateVector` | 5 | 5 trait floats + histories + confidence | PSV fingerprint |
| `UserProfile` | 4 | user_id, name, timestamps, session count | User registry entry |

---

## 18. FILE-BY-FILE BREAKDOWN

### inference/

| File | Lines | Purpose |
|------|-------|---------|
| `phase3_multimodal_fusion.py` | 838 | Core fusion engine |
| `phase4_cognitive_layer.py` | 2724 | Memory + personality profiling |
| `phase5_personality_engine.py` | 790 | PSV computation |
| `phase5_visualization.py` | 497 | Charts and reports |
| `phase4_user_manager.py` | 345 | User CRUD |
| `integrated_psychologist_ai.py` | large | Full system orchestration |
| `webcam_emotion_detection.py` | medium | Face real-time |
| `microphone_emotion_detection.py` | medium | Voice real-time |
| `dual_model_emotion_detection.py` | medium | Phase 1.5 ensemble |
| `phase3_demo.py` | medium | Demo scenarios |
| `demo_phase4_integration.py` | medium | Phase 4 demo |
| `demo_phase4_enhancements.py` | medium | Phase 4 features demo |

### training/

| File | Lines | Purpose |
|------|-------|---------|
| `model.py` | 338 | EmotionCNN + EmotionCNNDeep |
| `preprocessing.py` | 303 | Face transforms, face detection utils |
| `train_emotion_model.py` | 475 | Complete face training pipeline |
| `train_phase_1_5_finetune.py` | — | Fine-tuning for minority classes |
| `split_dataset.py` | — | Dataset splitting utilities |
| `training_template.py` | — | Reusable training boilerplate |
| `voice/feature_extraction.py` | 563 | 48-dim audio feature pipeline |
| `voice/voice_emotion_model.py` | 380 | VoiceEmotionModel + StressDetector |
| `voice/train_voice_emotion_balanced.py` | — | Balanced voice training |
| `voice/audio_preprocessing.py` | — | Audio normalization |
| `voice/dataset_utils.py` | — | RAVDESS/TESS loading |

### tests/ & diagnostics/

| File | Purpose |
|------|---------|
| `tests/test_phase3_final.py` | 10 synthetic scenario tests |
| `test_phase5.py` | Phase 5 unit tests |
| `diagnostics/check_voice_model.py` | Voice model validation |
| `diagnostics/test_happy_audio.py` | Happy emotion audio diagnostics |

### models/

```
models/
├── face_emotion/
│   ├── face_model.pth        ← EmotionCNN weights
│   └── model_config.json     ← Architecture params
└── voice_emotion/
    ├── voice_emotion_model.pth
    ├── stress_detector.pth
    └── model_config.json
```

---

## QUICK CHEAT SHEET

```
FACE:  FER2013 → 48×48 grayscale → Haar detect → EmotionCNN(300K) → 7 classes → 62.57%
VOICE: RAVDESS+TESS → librosa → 48-dim features → VoiceFC(70K) → 5 classes → 44% (+80% happy)
STRESS: librosa jitter/shimmer → StressFC(2K) → 3 levels (low/medium/high)

FUSION: NormalizedSignal(reliability) → deque(30) → psychology rules → PsychologicalState
OUTPUT: 15 MentalStates × 4 RiskLevels × confidence × explanations × stability

MEMORY: PsychologicalState → SessionMetrics → DailyProfile → WeeklyAggregate → PSV
PSV: [emotional_stability, stress_sensitivity, recovery_speed, positivity_bias, volatility]

KEY NUMBERS:
  Face accuracy: 62.57%   Voice accuracy: 44%   Happy: 80%   Scenario pass: 9/10
  Face params: 300K       Voice params: 70K      Stress params: 2K
  PSV dims: 5             Mental states: 15      Risk levels: 4
  Face classes: 7         Voice classes: 5       Stress levels: 3
  Feature vector: 48-dim  Temporal window: 30    Max memory: 90 days
```

---

*End of ULTRA_TECHNICAL_REFERENCE.md*
*All data sourced directly from project source code and documentation.*
