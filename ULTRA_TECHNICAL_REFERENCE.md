# PSYCHOLOGIST AI — ULTRA TECHNICAL REFERENCE
## Every Term, Method, Architecture, Layer, Function, Decision & Statistic

> **Version:** Complete — Updated March 31, 2026  
> **Created:** February 26, 2026  
> **Coverage:** All 5 Phases, All Layers, All Code Blocks, All Design Decisions  
> **Purpose:** Exhaustive reference for interviews, research, and deep implementation understanding  
> **Last Update:** Synced with live source code — fixed MentalState enum, model filenames, PSV learning rate, forward pass order, test files, and cheat sheet metrics

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
Block 1: Conv(1→64, k=3, pad=1) → ReLU → Conv(64→64, k=3, pad=1) → ReLU → BN→ MaxPool(2×2)
         Output: (64, 24, 24)
Block 2: Conv(64→128, k=3, pad=1) → ReLU → Conv(128→128, k=3, pad=1) → ReLU → BN → MaxPool(2×2)
         Output: (128, 12, 12)
Block 3: Conv(128→256, k=3, pad=1) → ReLU → Conv(256→256, k=3, pad=1) → ReLU → BN → MaxPool(2×2)
         Output: (256, 6, 6)

Flatten: 256 × 6 × 6 = 9216
FC1: Linear(9216 → 512) → ReLU → Dropout(0.5)
FC2: Linear(512 → 256) → ReLU → Dropout(0.3)
FC3 (Output): Linear(256 → num_classes)  ← raw logits

Total Params: 5,997,383 (~6M)
```

**Forward pass order (important):** Each conv block applies ReLU immediately after each Conv2d, then BatchNorm ONCE after both convolutions, then MaxPool. This differs from the common Conv→BN→ReLU pattern — here BatchNorm is applied after both ReLUs in the block, just before pooling.

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
    transforms.RandomHorizontalFlip(p=0.5),                          # Faces are left-right symmetric
    transforms.RandomRotation(degrees=15),                            # Head tilt variation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),         # Position variance
                            scale=(0.85, 1.15), shear=5),            # Scale + shear variance
    transforms.ColorJitter(brightness=0.3, contrast=0.3),            # Lighting variation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),                      # Normalize to [-1, 1]
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3))  # Random occlusion
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

**Parameters (live inference — `integrated_psychologist_ai.py`):**
- `scaleFactor=1.1` — Image pyramid scaling factor (10% size reduction per level; finer pyramid steps → more accurate detection at a minor speed cost)
- `minNeighbors=5` — Quality threshold (how many overlapping detections needed)
- `minSize=(30, 30)` — Minimum face size to detect

> **Note:** The `detect_faces_haar()` utility function in `preprocessing.py` defaults to `scaleFactor=1.3` (the original value). The live inference system (`integrated_psychologist_ai.py`, line 140) was updated to `scaleFactor=1.1` for better detection accuracy. These two values coexist in the codebase — `1.3` is the utility default, `1.1` is what actually runs in production.

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

#### Feature Family 5: Stress-Specific Features (4 dims)
```
jitter             — Cycle-to-cycle pitch period irregularity (%)
shimmer            — Cycle-to-cycle amplitude irregularity (%)
spectral_flatness  — Noisiness of voice (tonal vs noisy; 0=pure tone, 1=white noise)
pitch_variance     — Std-dev of pitch contour (vocal instability under stress)
```
**Why these 4 for stress?**
- `jitter` and `shimmer` are clinical biomarkers for vocal stress — microtremor in vocal folds under stress
- `spectral_flatness` appears here as a stress indicator (voice becomes less tonal when stressed) even though its mean/std are also in Family 3; here it feeds directly into `StressDetector` as a raw per-file scalar
- `pitch_variance` captures vocal instability — erratic pitch is a reliable marker of cognitive load
- Together these 4 features form the `stress_features` input vector to `StressDetector`

**`extract_stress_score()` — Rule-based fallback:**
```python
stress_score = 0.3 * jitter_norm + 0.3 * shimmer_norm + 0.2 * flatness_norm + 0.2 * pitch_var_norm
# low if < 0.3,  medium if < 0.6,  high otherwise
```
This rule-based score can be used when the neural `StressDetector` model is not available.

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
Input: Stress features (4 dims: jitter, shimmer, spectral_flatness, pitch_variance)

FC1: Linear(4 → 32) → BatchNorm1D(32) → ReLU → Dropout(0.2)
Output: Linear(32 → 3)  ← 3 stress levels: [low, medium, high]

Predict method:
  stress_idx, confidence, probabilities = stress_detector.predict_stress(features)
  stress_level = ['low', 'medium', 'high'][stress_idx]
```

**`VoiceEmotionSystem` — Combined wrapper class:**
```python
class VoiceEmotionSystem(nn.Module):
    # Wraps both VoiceEmotionModel + StressDetector into one system
    # predict() returns dict with: emotion, emotion_confidence,
    #   stress_level, stress_confidence, overall_confidence
    def __init__(self, feature_dim=48, num_emotions=5, num_stress_levels=3)
```
This class orchestrates both models, computing `overall_confidence = (emotion_conf + stress_conf) / 2.0`.

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

| State | Enum Value | Description | Trigger Conditions |
|-------|-----------|-------------|-------------------|
| `CALM` | `"calm"` | Relaxed, no stress | All signals neutral/positive, stress=low |
| `JOYFUL` | `"joyful"` | Active positive state | Face=happy, voice=happy, stress=low, high confidence |
| `HAPPY_UNDER_STRESS` | `"happy_under_stress"` | Smiling but internally stressed | dominant=happy + stress=high |
| `ANXIOUS` | `"anxious"` | Worried, fear-tinted | dominant=fear + stress=high/medium |
| `STRESSED` | `"stressed"` | General stress state | stress=high, no pattern match |
| `OVERWHELMED` | `"overwhelmed"` | Acute stress overload | stress=high + stress_persistence pattern |
| `CONFUSED` | `"confused"` | Mixed signals, no pattern | neutral emotion + stress=medium |
| `EMOTIONALLY_MASKED` | `"emotionally_masked"` | Hiding true emotions | temporal masking pattern detected |
| `EMOTIONALLY_FLAT` | `"emotionally_flat"` | Reduced emotional expression | neutral + stress=low + face & voice confidence < 0.5 |
| `EMOTIONALLY_UNSTABLE` | `"emotionally_unstable"` | Rapid emotional fluctuations | emotional_instability temporal pattern |
| `ANGRY_STRESSED` | `"angry_stressed"` | Anger + stress combined | dominant=angry + stress=high |
| `SAD_DEPRESSED` | `"sad_depressed"` | Sadness without high stress | dominant=sad + stress=low |
| `STABLE_NEGATIVE` | `"stable_negative"` | Persistent low state, not acute | negative emotion + medium stress |
| `STABLE_POSITIVE` | `"stable_positive"` | Consistent positive state | dominant=happy + stress=low, lower confidence |
| `FEARFUL` | `"fearful"` | Fear without high stress | dominant=fear + stress<=low |

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
# Temporal weights for N daily profiles:
ages = np.arange(N)                        # 0 = most recent, N-1 = oldest
weights = np.exp(-decay_lambda * ages)     # decay_lambda = 0.1
weights = weights / weights.sum()          # normalize to sum=1

# PSV slow-update rule (applied once per update_psv() call):
PSV_new[trait] = (1 - η) * PSV_old[trait] + η * new_observation
```
- `η = learning_rate = 0.03` (default) — PSV changes very slowly
- `decay_lambda = 0.1` — exponential recency bias
- `min_sessions_required = 3` — PSV does not update until at least 3 sessions exist
- **Confidence formula:** `confidence = min(1.0, total_sessions_processed / 50.0)` → full confidence after 50 sessions
- History capped at last 10 updates per trait to avoid unbounded growth

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

| Model | Parameters | File | Size | Status |
|-------|-----------|------|------|--------|
| EmotionCNNDeep (face main) | 5,997,383 | `emotion_cnn_best.pth` | 22.9 MB | ✅ Active |
| EmotionCNN (face specialist) | ~300K | `emotion_cnn_phase15_specialist.pth` | 4.87 MB | ✅ Active (minority class booster) |
| VoiceEmotionModel [256,128,64] | ~70K | `emotion_model_best_balanced.pth` | 226 KB | ✅ Active |
| StressDetector [4→32→3] | ~1.2K | `stress_model_best.pth` | 5 KB | ✅ Active |
| feature_scaler.pkl | — | `feature_scaler.pkl` | 1.7 KB | ✅ Required at inference |

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
| `phase3_multimodal_fusion.py` | 838 | Core fusion engine (4-layer architecture) |
| `phase4_cognitive_layer.py` | 2,342 | Memory + personality profiling (largest file) |
| `phase5_personality_engine.py` | 790 | PSV computation + PersonalityEngine |
| `phase5_visualization.py` | 497 | Radar, trend, bar charts + personality report |
| `phase4_user_manager.py` | 345 | UserManager + UserSelector + UserProfile |
| `integrated_psychologist_ai.py` | 36,680 bytes | Full real-time system orchestrating all phases |

### training/

| File | Lines | Purpose |
|------|-------|---------|
| `model.py` | 338 | EmotionCNN + EmotionCNNDeep + model utilities |
| `preprocessing.py` | 311 | Face transforms, face detection, ROI extraction |
| `train_emotion_model.py` | 513 | Complete face training pipeline (EmotionCNNDeep) |
| `train_phase_1_5_finetune.py` | — | Fine-tuning for minority classes (phase 1.5) |
| `split_dataset.py` | — | Dataset splitting utilities |
| `voice/feature_extraction.py` | 563 | 48-dim audio feature pipeline + stress score |
| `voice/voice_emotion_model.py` | 380 | VoiceEmotionModel + StressDetector + VoiceEmotionSystem |
| `voice/train_voice_emotion_balanced.py` | — | Balanced voice training with StandardScaler |
| `voice/audio_preprocessing.py` | — | preprocess_audio() with noise reduction + preemphasis |
| `voice/dataset_utils.py` | — | RAVDESS/TESS loading and label mapping |
| `voice/download_datasets.py` | — | Dataset download utilities |

### tests/

| File | Purpose |
|------|---------|
| `tests/test_face_emotion.py` | Face emotion model unit tests |
| `tests/test_voice_emotion.py` | Voice emotion model unit tests |
| `tests/run_all_tests.py` | Test runner for all test suites |
| `tests/utils.py` | Shared test utilities |

### models/

```
models/
├── face_emotion/
│   ├── emotion_cnn_best.pth              ← EmotionCNNDeep weights (22.9 MB)
│   ├── emotion_cnn_phase15_specialist.pth ← EmotionCNN specialist (4.87 MB)
│   ├── labels.json                        ← {0: "angry", ..., 6: "surprise"}
│   └── config.json                        ← architecture, metrics, timestamp
└── voice_emotion/
    ├── emotion_model_best_balanced.pth    ← VoiceEmotionModel weights (226 KB)
    ├── stress_model_best.pth              ← StressDetector weights (5 KB)
    ├── feature_scaler.pkl                 ← StandardScaler fitted on 1680 train samples (1.7 KB)
    ├── labels.json                        ← {0: "angry", ..., 4: "sad"}
    └── config.json                        ← sr=16000, window=3s, feature_type
```

---

## QUICK CHEAT SHEET

```
FACE:  FER2013 → 48×48 grayscale → Haar detect → EmotionCNNDeep(6M) → 7 classes → 64.42% / F1=0.633
VOICE: RAVDESS+TESS → preprocess_audio(sr=16000) → 48-dim features → VoiceFC(70K) → 5 classes → ~54%
STRESS: jitter/shimmer/flatness/pitch_var (4-dim) → StressFC(1.2K) → 3 levels (low/medium/high)

FUSION: NormalizedSignal(reliability) → deque(30) → rule-based psychology → PsychologicalState
OUTPUT: 15 MentalStates × 4 RiskLevels × confidence × explanations × stability_score

MEMORY: PsychologicalState → SessionMetrics → DailyProfile → WeeklyAggregate → PSV
PSV: [emotional_stability, stress_sensitivity, recovery_speed, positivity_bias, volatility]
PSV UPDATE: η=0.03, decay_λ=0.1, confidence grows at sessions/50, history capped at 10 updates

KEY NUMBERS:
  Face accuracy: 64.42%    Face MacroF1: 0.633    Disgust recall: 65%
  Voice accuracy: ~54%     Voice MacroF1: 0.508   Scenario tests: 9/10
  Face params: ~6M         Voice params: ~70K     Stress params: ~1.2K
  PSV dims: 5              Mental states: 15      Risk levels: 4
  Face classes: 7          Voice classes: 5       Stress levels: 3
  Feature vector: 48-dim   Temporal window: 30    Max memory: 90 days
  Min sessions for PSV: 3  Full PSV confidence: 50 sessions

MODEL FILES:
  emotion_cnn_best.pth              (22.9 MB — EmotionCNNDeep)
  emotion_cnn_phase15_specialist.pth (4.87 MB — EmotionCNN specialist)
  emotion_model_best_balanced.pth   (226 KB — VoiceEmotionModel)
  stress_model_best.pth             (5 KB — StressDetector)
  feature_scaler.pkl                (1.7 KB — REQUIRED at voice inference)
```

---

## 19. TECHNICAL INTERVIEW QUESTIONS & ANSWERS

> Organized by topic. Each answer is grounded in actual project implementation.

---

### 🧠 Section A — System Architecture & Design

**Q1. Walk me through the overall architecture of Psychologist AI.**
> The system has 5 phases. Phase 1 detects facial emotion (7 classes) using a custom VGG-style CNN trained on FER2013. Phase 2 detects voice emotion (5 classes) and stress (3 levels) via a fully-connected network on 48-dim hand-crafted audio features. Phase 3 fuses all three signals using a 4-layer rule-based engine that outputs a `PsychologicalState` from 15 possible mental states. Phase 4 aggregates states into `SessionMemory → DailyProfile → WeeklyAggregate` and persists them per user as JSON. Phase 5 maintains a 5-dimensional `PersonalityStateVector` (PSV) that updates slowly via exponential decay weighting to infer long-term behavioral traits.

**Q2. Why did you choose a rule-based fusion engine instead of training a neural fusion layer?**
> Three reasons: (1) **Data scarcity** — there is no large labeled dataset of simultaneously recorded face + voice + ground-truth mental states. (2) **Interpretability** — rules produce human-readable explanations like "face shows happy but stress is high → masked emotion," which a neural network cannot. (3) **Safety** — in a mental health context, rules can be audited and corrected by a psychologist; a black-box model cannot. The consistency of deterministic rules is also important: same input always produces same output.

**Q3. What is the `STATE ≠ TRAIT` principle and why does it matter?**
> A momentary emotion (state) — e.g., feeling anxious today — is not the same as a personality trait. The system deliberately separates them: Phase 3 measures state per frame, Phase 4 accumulates states across a session, and Phase 5 only infers traits (PSV) after 3+ sessions with a very slow learning rate (η=0.03). This prevents a single bad day from permanently labeling someone as "stress-sensitive" — a critical ethical constraint.

**Q4. Why does the system use `Baseline > Absolute` for personalization?**
> Population averages are meaningless for individual mental health. A stoic person at 30% stress ratio IS stressed; an expressive person at 30% is calm. The `BaselineProfile` and `DeviationDetector` (Z-score) calibrate thresholds per user, making the system genuinely personalized rather than population-normalized.

**Q5. How do you handle the case when a face is not detected?**
> In `SignalNormalizer.normalize_face_signal()`, if `face_detected=False`, `signal_quality` is set to `0.0`. Since `weighted_confidence = confidence × reliability × signal_quality`, a zero quality collapses the face signal's contribution. The fusion engine then falls back to voice and stress signals (which have higher reliability weights of 0.7 and 0.9 respectively). The system degrades gracefully rather than crashing or guessing.

---

### 🖼️ Section B — Phase 1: Face Emotion (CNN)

**Q6. Why EmotionCNNDeep over ResNet-50 for face emotion?**
> ResNet-50 has 25M parameters and achieves ~65-68% on FER2013 — only ~3-4% better than our EmotionCNNDeep (64.42%) with 6M params. But ResNet struggles at real-time on CPU (~10ms vs ~2ms), and the input is only 48×48 grayscale — ResNet's depth is designed for 224×224 RGB images. For a 48×48 input, heavy downsampling destroys spatial information before the deeper blocks contribute anything. Our custom architecture is purpose-built for the task.

**Q7. Explain the EmotionCNNDeep forward pass order — specifically the ReLU/BatchNorm placement.**
> Unlike the canonical Conv→BN→ReLU pattern, EmotionCNNDeep applies ReLU **immediately after each Conv2d** (two per block), then BatchNorm **once** at the end of the block (after both ReLUs), before MaxPool. So: `Conv→ReLU→Conv→ReLU→BN→MaxPool`. This means non-linearity is introduced before normalization, which preserves gradient flow through the double-conv block while still gaining the stabilization benefits of BatchNorm before the spatial reduction.

**Q8. Why use macro F1 as the training checkpoint criterion instead of validation accuracy?**
> FER2013 is heavily imbalanced (`happy`=5214, `disgust`=316). Accuracy is dominated by majority classes — a model predicting 100% happy gets 73% accuracy but near-0% disgust recall. Macro F1 averages per-class F1 scores equally, so poor disgust recall directly tanks the metric and forces the model to improve on minority classes. This is why we saved on `val_macro_f1 > best_macro_f1` in the training loop.

**Q9. How did you handle class imbalance in FER2013?**
> Three-layer approach: (1) `compute_class_weight('balanced')` from sklearn computes weights inversely proportional to class frequency — disgust gets weight 9.38×. (2) These weights are passed to `nn.CrossEntropyLoss(weight=class_weight_tensor)` so minority class errors are penalized more. (3) Phase 1.5 fine-tunes a specialist model on minority classes with frozen early layers. Result: disgust recall improved from ~10% to 65%.

**Q10. What does `RandomErasing` augmentation achieve?**
> It randomly masks out a rectangle (2-15% of image area, aspect ratio 0.3-3.3) after ToTensor. This forces the model to not over-rely on any single facial region — e.g., always detecting "happy" purely from a wide mouth. It simulates partial occlusion (hand over face, hair, scarves) and makes the model more robust to real-world inference conditions. Applied at p=0.3 so only 30% of training images are affected.

**Q11. Why grayscale and not RGB for face emotion?**
> Emotion expression is encoded in **shape and texture** of facial muscles, not in color. Converting to grayscale reduces input channels from 3 to 1, cutting the first conv layer's parameters by 3×. FER2013 images are already grayscale (annotated that way), so using RGB would mean duplicating the same channel three times — adding parameters without any new information.

**Q12. What is `cv2.INTER_AREA` and why use it for resizing?**
> `INTER_AREA` uses pixel area relation for resampling — it's effectively a weighted average of the original pixels that map to the target pixel. This is the best interpolation method when **shrinking** images (downsampling), as it avoids aliasing by averaging rather than picking nearest neighbors or bicubic extrapolation. Since FER2013 images may be larger and we resize to 48×48, `INTER_AREA` produces the cleanest downsamples.

---

### 🎙️ Section C — Phase 2: Voice Emotion & Stress

**Q13. Why 48 features specifically for voice emotion?**
> The 48 dimensions are: 5 pitch features (mean, std, range, median, slope) + 5 energy features (rms_mean, rms_std, rms_max, zcr_mean, zcr_std) + 3 temporal features (speaking_rate, num_onsets, pause_ratio) + 6 spectral scalar features (centroid mean/std, rolloff mean/std, flatness mean/std) + 3 stress scalar features (jitter, shimmer, pitch_variance) = **22 scalars** + **13 MFCC means + 13 MFCC stds = 26** → total **48**.

**Q14. Why fully-connected layers for voice emotion rather than a CNN or LSTM?**
> The input is already a pre-extracted 48-dimensional feature vector — a flat array of numbers. CNN needs 2D spatial structure (spectrogram), LSTM needs sequential raw frames. Since we've already collapsed the audio into a summary vector via librosa, there's no spatial or temporal structure remaining for CNN/LSTM to exploit. FC layers are optimal, fast, and interpretable for this input type.

**Q15. What is `jitter` and `shimmer` and why are they clinical stress biomarkers?**
> **Jitter** is cycle-to-cycle variation in the fundamental period (1/F0) of vocal fold vibration — measured as `mean(|T[i] - T[i-1]|) / mean(T)` × 100%. **Shimmer** is cycle-to-cycle variation in amplitude — measured as `mean(|A[i] - A[i-1]|) / mean(A)` × 100%. Under stress, microtremors in the laryngeal muscles cause irregular vocal fold vibration, increasing both. They're used clinically in Parkinson's diagnosis and aviation safety (pilot stress detection). Normal jitter < 1%, shimmer < 5%; elevated values indicate physiological arousal.

**Q16. What was the root cause of the voice model collapsing to 23% accuracy predicting only 'sad'?**
> Two bugs: (1) **Sample rate mismatch** — training loaded audio at `sr=16000` via `librosa.load()`, but the test script loaded at default `sr=22050`. Since MFCC extraction is SR-dependent (mel filterbank frequencies scale with SR), features extracted at 22050 had completely different distributions than the scaler was fit on. (2) **Missing StandardScaler at inference** — pitch features are in 100-500 Hz range while MFCCs are in -20 to +20 range; without normalization, the model receives out-of-distribution inputs. Fix: identical `preprocess_audio(sr=16000)` + `scaler.transform()` at all inference points.

**Q17. Why use `WeightedRandomSampler` AND weighted loss simultaneously?**
> They serve different purposes. `WeightedRandomSampler` ensures each **mini-batch** has equal class representation by oversampling minority classes during data loading — it changes *what the model sees per batch*. `Weighted CrossEntropyLoss` changes *how much each error counts* in the gradient. Together they provide balanced gradient signal from two directions: balanced sampling + balanced loss terms. Either alone is often insufficient for extreme imbalance.

**Q18. Why StandardScaler specifically (not MinMaxScaler or RobustScaler)?**
> Voice features have very different scales: pitch mean ~200 Hz, MFCCs ~-5 to +15, RMS ~0.001-0.1. `StandardScaler` (zero mean, unit variance) is ideal because: (1) FC+BatchNorm layers assume roughly zero-centered inputs, (2) it's robust to the Gaussian-like distributions of these features, (3) it doesn't clip outliers (MinMaxScaler would compress the pitch range into [0-1] while MFCC values near the edges get distorted). Critical: the scaler is **fit only on training data** and applied identically to val, test, and real-time inference.

---

### 🔀 Section D — Phase 3: Multi-Modal Fusion

**Q19. Explain modality reliability weights and their justification.**
> `FACE=0.5, VOICE=0.7, STRESS=0.9`. Face gets the lowest weight because people **consciously control** facial expressions (professional politeness, cultural suppression). Voice tone and pitch are largely **subconscious** — harder to sustain fake emotion over extended conversation. Stress markers (jitter/shimmer) are **physiological** — governed by the autonomic nervous system and nearly impossible to consciously suppress. Paul Ekman's work on micro-expressions supports face < voice; aviation/military stress detection literature validates acoustic stress markers.

**Q20. How is `weighted_confidence` computed and what does it represent?**
> `weighted_confidence = confidence × reliability × signal_quality` where `confidence` is the model's softmax probability [0,1], `reliability` is the modality base score (0.5/0.7/0.9), and `signal_quality` is a dynamic factor based on detection conditions (e.g., face not detected → 0.0, high confidence face → min(1.0, confidence × 1.5)). This three-way product ensures that a high-confidence prediction from an unreliable modality (face, 0.5) can be outweighed by a moderate-confidence prediction from a reliable modality (stress, 0.9).

**Q21. Why use `collections.deque(maxlen=30)` for temporal memory instead of a list?**
> `list.pop(0)` is O(n) — it shifts all n elements left when removing from the front. `deque.popleft()` is O(1) — it's a doubly-linked list, so head removal is pointer manipulation only. At 30 FPS with continuous processing, this matters: 30 O(n) pops/second for n=30 is 900 operations vs 30 O(1) operations. The `maxlen` parameter also automatically discards oldest elements on `append()`, eliminating the need for manual bounds checking.

**Q22. Walk through the confidence penalty calculation.**
> Base confidence is a weighted sum: `face × 0.25 + voice × 0.35 + stress × 0.40` (weights proportional to reliability). Three penalties are subtracted: (1) **Conflict penalty** = -0.15 if a `hidden_emotion` was detected (modality disagreement); (2) **Instability penalty** = -0.10 × pattern.intensity if `emotional_instability` pattern is present; (3) **Quality penalty** = -(1 - avg_signal_quality) × 0.15. Final = `max(0.0, min(1.0, base - all_penalties))`. This ensures confident predictions only when signals agree and are high quality.

**Q23. How does masking detection work?**
> `TemporalMemory.detect_masking()` looks at the last 5 frames: if `face_neutral_ratio > 0.6` (face is neutral 60%+ of the time) AND `voice_emotional_ratio > 0.6` (voice is non-neutral 60%+ of the time), masking is detected. The dominant voice emotion is reported. This logic mirrors the clinical concept of emotional suppression — the face presents a composed neutral front while the voice betrays an underlying emotional state the person is trying to hide.

**Q24. What are the 5 emotion mapping rules and why?**
> `FACE→VOICE` normalization: `'disgust'→'angry'` (disgust has no voice equivalent; physiological response overlaps anger — raised heart rate, muscle tension) and `'surprise'→'neutral'` (surprise is extremely transient, lasting ~0.5s; sustained analysis would categorize it as a random spike rather than an enduring emotional state). This allows cross-modal comparison between the 7-class face model and 5-class voice model.

---

### 🧬 Section E — Phase 4: Long-Term Cognitive Layer

**Q25. How does `calculate_metrics()` build `SessionMetrics`?**
> It iterates over all `(timestamp, PsychologicalState)` pairs stored during a session and computes: mental state distribution (Counter → top 5), stress duration ratio (count of stress/overwhelmed/anxious/unstable states / total), high-stress ratio (overwhelmed + unstable only), masking frequency (masking_events / session_minutes), risk level distribution (numeric 0-3 mapping), emotional polarity (positive/negative/neutral state ratios), and confidence/stability variance. All values are stored in a `SessionMetrics` dataclass for persistence.

**Q26. Explain incremental averaging in `_update_daily_profile()`.**
> When session N arrives, existing daily profile averages were computed from N-1 previous sessions. The formula `new_avg = (old_avg × n + new_value) / (n + 1)` updates the running average in O(1) without storing all raw session data. This is the **online/streaming mean update** formula. It's numerically stable and memory-efficient — the daily profile file stays fixed size regardless of how many sessions occur that day.

**Q27. How does the WeeklyAggregate compute trend slopes?**
> For each metric (stress, masking, risk, stability), it collects the daily values for that week and runs `np.polyfit(x, y, deg=1)[0]` — linear regression returning the slope coefficient. Positive slope = metric increasing over the week (e.g., stress trending up). Negative = improving. This is the same 1D linear regression used for PSV trend analysis. For n<2 days, slope defaults to 0.

**Q28. Why JSON over SQLite for user memory storage?**
> (1) **Zero dependencies** — no database engine needed, `json` is stdlib. (2) **Human-readable** — directly inspectable for debugging corrupt user profiles. (3) **Hierarchical** — the nested `{date → DailyProfile → sessions}` structure maps naturally to JSON; SQL would require multiple tables with foreign keys. (4) **Portable** — JSON files can be emailed/backed up trivially. Trade-off: no query language, no concurrent write safety. For future scale (>1000 users), SQLite migration is noted.

**Q29. How does `DeviationDetector` know what's "unusual" for a specific user?**
> Z-score: `z = (current_value - baseline_mean) / baseline_std`. The baseline is computed from the first 3-5 sessions of a new user. `|z| > 2.0` = notable deviation; `|z| > 3.0` = significant. This is personalized: if a user normally has 40% stress ratio (they're naturally expressive/high-energy), their z-score for 45% stress is near 0 — normal for them. But for a user whose baseline is 10% stress, 45% gives z ≈ 3.5 — red flag. Absolute thresholds would incorrectly flag the first person.

---

### 🎭 Section F — Phase 5: Personality Engine

**Q30. What is the Personality State Vector (PSV) and what do each of the 5 traits measure?**
> PSV is a 5-dimensional float vector `[0,1]` representing long-term behavioral tendencies:
> - `emotional_stability`: 1 - variance(emotion states) — high = consistent emotions
> - `stress_sensitivity`: weighted avg(stress_ratio + high_stress_ratio) — high = reactive to stressors
> - `recovery_speed`: proxy via avg_stability - escalation_penalty — high = returns quickly to baseline
> - `positivity_bias`: positive_time / total_time — high = positive emotional orientation
> - `volatility`: state_switches_per_minute / 5.0 — high = frequent emotional transitions

**Q31. Why learning rate η=0.03 specifically?**
> With η=0.03, the PSV update rule `PSV_new = 0.97 × PSV_old + 0.03 × new_observation` means a single session can shift any trait by at most 3%. Even a maximally extreme session (all 1s or 0s) changes a trait from 0.5 to 0.515 or 0.485 — imperceptible. After 10 sessions of consistent extreme behavior, the trait moves to ~0.74. This slow convergence prevents mood-state contamination of the personality model and requires sustained patterns to shift the PSV meaningfully.

**Q32. How does confidence grow and what does it mean practically?**
> `confidence = min(1.0, total_sessions_processed / 50.0)`. After 3 sessions: 0.06 (very_low). After 10: 0.20 (low–moderate boundary). After 25: 0.50 (moderate). After 50: 1.0 (very_high). Practically: at very_low confidence, the behavioral descriptor adds "(Limited data - preliminary assessment)" and PSV traits are close to their 0.5 prior. At high confidence, trait values are statistically meaningful aggregates. The confidence label prevents over-interpretation of early PSV readings.

**Q33. Why exponential decay weights for older daily profiles?**
> `weight = exp(-0.1 × age_in_days)`. A profile from 20 days ago gets weight `exp(-2) ≈ 0.135` relative to today's weight of 1.0. This means: (1) Recent behavior dominates the PSV — last week matters more than last month. (2) Old trauma or aberrant behavior doesn't permanently define the profile. (3) If a person genuinely changes (e.g., starts therapy, reduces work stress), the PSV adapts over several weeks rather than requiring erasure of history. It's mathematically equivalent to a weighted moving average with no hard cutoff.

---

### ⚙️ Section G — Training & ML Engineering

**Q34. What is `ReduceLROnPlateau` and why step it on macro F1?**
> `ReduceLROnPlateau(mode='max', patience=5, factor=0.5)` halves the learning rate when the monitored metric doesn't improve for 5 epochs. We monitor `val_macro_f1` (not val_loss) because: the loss is distorted by class weights (high disgust weight inflates absolute loss even when learning), while macro F1 directly measures what we care about — per-class performance. Stepping on F1 ensures LR reduction happens when we're actually plateauing in classification quality, not in weighted loss.

**Q35. What is `persistent_workers=True` in DataLoader and why use it?**
> With `num_workers > 0`, PyTorch spawns worker processes to load data in parallel. By default, workers are killed after each epoch and respawned next epoch — this startup overhead can be significant. `persistent_workers=True` keeps workers alive across epochs, eliminating respawn cost. At `num_workers=8` for train and `num_workers=4` for val/test, this is a non-trivial speedup especially on Windows where process spawning is slower than Linux fork.

**Q36. Why `torch.save(model.state_dict())` instead of `torch.save(model)`?**
> Saving `state_dict()` (a dict of {layer_name: tensor}) is the recommended PyTorch practice because: (1) **Portability** — it doesn't pickle the model class definition, so loading works even if the code is refactored. (2) **Architecture flexibility** — can be loaded into a differently-initialized model (e.g., different dropout rate). (3) **Smaller files** — no Python bytecode, just tensor data. Loading requires re-instantiating the model class then calling `model.load_state_dict(torch.load(...))`.

**Q37. Explain the dynamic architecture loading pattern and why it matters.**
> `config.json` stores `"architecture": "EmotionCNNDeep"`. All consumers read this field and use `_ARCH_MAP = {'EmotionCNNDeep': EmotionCNNDeep, 'EmotionCNN': EmotionCNN}` to instantiate the correct class. This means if Phase 1.5 trains a specialist using `EmotionCNN`, its config says `EmotionCNN`, and the inference code loads it correctly without hardcoding. The system can handle a mix of architectures across checkpoints without code changes.

**Q38. What is preemphasis filtering and why apply it?**
> Preemphasis: `y[n] = y[n] - 0.97 × y[n-1]` — a first-order high-pass FIR filter that amplifies high frequencies by ~20 dB/decade. It compensates for the natural roll-off of the vocal tract (which acts as a low-pass filter), making the spectral envelope more flat and improving MFCCs' sensitivity to fricatives and consonants. Without preemphasis, voiced sounds (vowels) dominate the spectrum and MFCC features are biased toward their characteristics. Standard in HTK, Kaldi, and clinical voice analysis pipelines.

---

### 📐 Section H — Mathematics & Algorithms

**Q39. Derive the MFCC extraction pipeline from raw audio to coefficients.**
> (1) Pre-emphasis: `s'[n] = s[n] - 0.97s[n-1]` — boost high freq. (2) Frame into 25ms windows with 10ms hop. (3) Apply Hamming window: `w[n] = 0.54 - 0.46cos(2πn/N)` — reduces spectral leakage at frame edges. (4) FFT + power spectrum: `P[k] = |FFT(w·s)|²`. (5) Apply 26 mel-scale triangular filters: `Mel(f) = 2595×log10(1 + f/700)` — mimics cochlear frequency resolution (logarithmic). (6) Log of filter energies (Weber-Fechner perceptual loudness). (7) DCT: decorrelates the filterbank energies, producing 13 compact, uncorrelated coefficients.

**Q40. How does `np.polyfit(x, y, deg=1)[0]` give trend direction?**
> `np.polyfit(x, y, 1)` fits a degree-1 polynomial `y = ax + b` using least-squares regression, returning `[a, b]`. `[0]` extracts the slope `a`. Positive slope = y increasing over time = metric worsening (for stress/risk) or improving (for stability). The threshold `|slope| < 0.02` = "stable" is empirically set to filter out numerical noise from small float changes. Slope is computed on the last 5 history values, giving a short-term trend rather than all-time trend.

**Q41. Explain Z-score deviation detection mathematically.**
> Z-score: `z = (x - μ) / σ` where μ = baseline mean, σ = baseline std. It measures how many standard deviations the current observation is from the user's personal average. `|z| < 1.0`: normal variation (68% of a Gaussian falls here). `|z| > 2.0`: notable (outside 95% CI). `|z| > 3.0`: significant (outside 99.7% CI). Since behavioral metrics are not perfectly Gaussian, these thresholds are approximate but provide a principled, personalized, self-calibrating signal.

**Q42. Why does the stability score start at 1.0 and subtract penalties?**
> Starting at maximum stability (1.0) and subtracting represents the assumption of stability as default — deviation from stability is the signal, not the presence of it. Penalty 1: `switch_penalty = min(0.5, (face_switches + voice_switches) × 0.05)` — capped at 0.5. Penalty 2: `-0.3 × instability_pattern.intensity`. Bonus: +0.1 if dominant emotion > 60% in both face and voice (consensus stability). Final is `max(0.0, min(1.0, result))`.

---

### 🔧 Section I — System Engineering & Production

**Q43. How does the system handle multiple concurrent users?**
> Each user has a separate file namespace: `{user_id}_longterm_memory.json` and `{user_id}_psv.json`. The `UserManager` uses a single `users.json` registry. All file I/O is synchronous (`json.dump`/`json.load`) — race conditions would occur with concurrent writes. Current design is single-user-at-a-time. For multi-user production, file locking (fcntl/portalocker) or migration to SQLite (which has proper ACID transactions) would be needed.

**Q44. How does session boundary detection work without explicit user input?**
> `SessionMemory.is_active(timeout_minutes=5.0)` checks `time.time() - last_frame_timestamp < 5 × 60`. If the user walks away and no new frames arrive for 5 minutes, the next call returns `False`, signaling the integrated system to finalize the session, compute `SessionMetrics`, and call `LongTermMemory.add_session()`. This is automatic activity-based boundary detection — the user never needs to press "end session."

**Q45. Why `pathlib.Path` everywhere instead of `os.path`?**
> `pathlib.Path` is object-oriented (method chaining: `path / 'subdir' / 'file.json'`), returns `Path` objects that work natively with `open()`, `exists()`, `mkdir(parents=True, exist_ok=True)`, etc. It's OS-agnostic (handles Windows `\` vs Unix `/` automatically). `os.path` returns strings, requiring cast back to str for many operations. `pathlib` is the modern Python 3.4+ standard; `os.path` is legacy.

**Q46. What does `@dataclass` give you over a plain class?**
> `@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` from field annotations, eliminating boilerplate. `field(default_factory=list)` creates a fresh list per-instance (critical — a mutable default `[]` in a regular class would be shared across all instances, a classic Python bug). `asdict()` from `dataclasses` recursively converts the whole structure to a JSON-serializable dict, enabling one-line persistence: `json.dump(asdict(obj), f)`.

**Q47. What is `hashlib.sha256` used for in user ID generation?**
> `user_id = sha256(f"{name}_{timestamp}".encode()).hexdigest()[:12]`. This 12-char hex prefix from a SHA-256 hash provides: (1) **Uniqueness** — two users named "Alice" registered at different timestamps get different IDs. (2) **No collision risk** — SHA-256 is collision-resistant; 12 hex chars = 48 bits of entropy = 2^48 possible IDs. (3) **Security** — the hash is one-way; you can't recover name/timestamp from the ID. The sanitized name is prepended for human readability (`alice_90806d704641`).

---

### 💬 Section J — Behavioral & Conceptual Questions

**Q48. What would you do to improve voice emotion accuracy beyond 54%?**
> Three concrete approaches: (1) **Wav2Vec 2.0 embeddings** — replace hand-crafted 48-dim features with 768-dim contextual embeddings from Facebook's self-supervised model; state-of-the-art on IEMOCAP reaches 80%+. (2) **Larger + more diverse dataset** — IEMOCAP (12h acted+spontaneous), MSP-IMPROV, or cross-corpus training; current 336 samples/class is the main bottleneck. (3) **Raw mel-spectrogram CNN** — 2D CNN on spectrograms (80×time frames) captures both temporal and frequency patterns that hand-crafted features miss.

**Q49. What ethical constraints did you deliberately build into the system?**
> (1) **No clinical diagnoses** — only behavioral descriptors using probabilistic language ("shows patterns of," not "has"). (2) **No DSM-5/ICD-11 labels** — never outputs "depression" or "anxiety disorder." (3) **Slow PSV updates** — one bad session cannot permanently label someone. (4) **Confidence levels** — PSV confidence below 0.4 adds a "preliminary assessment" warning. (5) **Risk levels, not alerts** — the system flags HIGH risk but doesn't contact anyone; it surfaces data to the human user. (6) **No biometric data stored** — only aggregated metrics, never raw video/audio.

**Q50. If a new modality (e.g., text, keystroke dynamics) needs to be added, how would you extend Phase 3?**
> The architecture is modality-agnostic at the fusion layer. Steps: (1) Add a new `Modality` enum value with a reliability score. (2) Implement `normalize_text_signal()` in `SignalNormalizer` that returns a `NormalizedSignal` with the new modality. (3) Add a history deque in `TemporalMemory` for the new signal. (4) Update `FusionEngine.fuse_signals()` to include the new signal's `weighted_confidence` in the conflict and agreement logic. (5) Update `ConfidenceCalculator` base confidence weights. No other changes needed — Phase 4/5 consume `PsychologicalState` objects and are modality-agnostic.

**Q51. How does the face model load dynamically from config, and why is this important?**
> All consumers read `models/face_emotion/config.json`, extract `"architecture"`, and use `_ARCH_MAP = {'EmotionCNNDeep': EmotionCNNDeep, 'EmotionCNN': EmotionCNN}` to instantiate. This means: if a researcher trains a new `EmotionCNNV3` architecture, adds it to the map, and saves it with `"architecture": "EmotionCNNV3"` in config.json, **zero other code changes** are needed — `integrated_psychologist_ai.py`, `run_verify.py`, and test scripts all pick it up automatically. This is the open/closed principle: open for extension, closed for modification.

**Q52. What happens if the PSV file is corrupted or missing?**
> `PersonalityEngine._load_psv()` wraps JSON loading in a `try/except Exception`. If loading fails for any reason (file missing, JSON parse error, schema mismatch), it falls back to `return PersonalityStateVector()` — a fresh PSV with all traits at 0.5 (neutral prior) and zero sessions. This graceful degradation ensures the system never crashes due to a corrupt user profile; it simply starts fresh. A warning is printed: `"⚠️ Warning: Could not load PSV, creating new one."`.

---

### 🏗️ Section K — Quick-Fire Technical Questions

| Question | Answer |
|----------|--------|
| What are the 3 temporal patterns detected? | `stress_persistence`, `emotional_masking`, `emotional_instability` |
| What is the window size of `TemporalMemory`? | 30 frames (deque maxlen=30) |
| Face model input shape? | `(batch, 1, 48, 48)` — grayscale |
| Voice model input shape? | `(batch, 48)` — flat feature vector |
| What optimizer for face training? | Adam(lr=0.001, weight_decay=1e-4) |
| What scheduler for face training? | ReduceLROnPlateau(mode='max', patience=5, factor=0.5) |
| Early stopping patience? | 15 epochs (face); 20 epochs (voice) |
| Maximum sessions per day (LongTermMemory)? | 10 (configurable) |
| Maximum days stored in memory? | 90 days, older data archived |
| What is `signal_quality` minimum for stress? | 0.5 (always at least 0.5: `max(0.5, confidence)`) |
| How many sessions before PSV updates? | min_sessions_required = 3 |
| Full PSV confidence after how many sessions? | 50 sessions |
| Training augmentation applied at inference? | Never — `get_val_transforms()` only at inference |
| User ID format? | `{sanitized_name}_{sha256[:12]}` e.g. `alice_90806d704641` |
| Conflict penalty in confidence calculation? | -0.15 when hidden_emotion is not None |
| Stability score starts at what value? | 1.0 (decremented by penalties) |
| Voice SR for training AND inference? | 16000 Hz (critical: must match) |
| Which phase has the largest source file? | Phase 4 (`phase4_cognitive_layer.py`, 2342 lines) |
| How many face classes? | 7: angry, disgust, fear, happy, neutral, sad, surprise |
| How many voice classes? | 5: angry, fear, happy, neutral, sad |
| How many stress levels? | 3: low, medium, high |
| PSV traits history: last N kept? | Last 10 updates per trait |
| What does disgust map to in voice space? | `'angry'` (EMOTION_MAPPING in SignalNormalizer) |
| What does surprise map to in voice space? | `'neutral'` |

---

*End of ULTRA_TECHNICAL_REFERENCE.md*
*All data sourced directly from project source code and documentation.*
*Last sync: March 31, 2026*
