# 🧠 PSYCHOLOGIST AI - PHASE 0: FOUNDATION

## 0.1 Project Vision Lock 🔒

**Your project is ONE system, not 8 random projects.**

```
PSYCHOLOGIST_AI/
│
├── data/              # All datasets (face, voice, text, etc.)
├── models/            # Saved model weights (.pth, .h5)
├── training/          # Training scripts for each phase
├── inference/         # Real-time prediction scripts
├── fusion/            # Multi-modal integration logic
├── memory/            # Conversation history & user profiles
├── ui/                # Web/Desktop interface
└── assets/reports/    # Training logs, metrics, visualizations
```

**Every future phase plugs into this structure. No exceptions.**

---

## 0.2 Environment Setup (Clean AF) 🔧

### Tools You MUST Install

- ✅ **Python 3.10 or 3.11** (avoid 3.12 for library compatibility)
- ✅ **VS Code** (with Python extension)
- ✅ **Git + GitHub** (version control)
- ✅ **Webcam + Mic** (hardware check for inference)

### Python Libraries (Core)

Install with: `pip install -r requirements.txt`

**Core Libraries:**
- `numpy` - numerical computing
- `pandas` - data manipulation
- `matplotlib` - visualization
- `opencv-python` - computer vision
- `scikit-learn` - ML utilities
- `torch torchvision torchaudio` - **PyTorch** (our DL framework)
- `librosa` - audio processing (Phase 2)
- `mediapipe` - pose/gesture (Phase 3)

### 👉 Deep Learning Framework Choice

**We chose: PyTorch** 🔥

**Why PyTorch?**
- Flexibility for research & experimentation
- Better debugging (Pythonic code)
- Industry standard for NLP/CV research
- Easier to understand what's happening under the hood

---

## 0.3 Machine Learning Fundamentals (Training POV)

Before deep learning, understand **training logic**:

### Core Concepts You MUST Know

| Concept | What It Means |
|---------|---------------|
| **Features (X)** | Input data (images, audio, text) |
| **Labels (y)** | Ground truth (emotions, classes) |
| **Train/Val/Test Split** | 70% train, 15% validation, 15% test |
| **Overfitting** | Model memorizes training data (bad) |
| **Underfitting** | Model too simple to learn patterns (bad) |
| **Accuracy** | % of correct predictions |
| **Precision** | How many predicted positives are actually positive |
| **Recall** | How many actual positives did we catch |
| **F1 Score** | Balance between precision and recall |
| **Confusion Matrix** | Shows where your model gets confused |

### 💡 Rule of Thumb

**If you don't know WHY accuracy is bad sometimes — you're not ready yet.**

Example: 
- Dataset: 95% healthy patients, 5% sick
- Model predicts everyone is healthy → 95% accuracy
- But it's useless! (0% recall for sick patients)

---

## 0.4 Deep Learning Core (Non-Negotiable) 🧠

### Concepts You MUST Understand

| Concept | Explanation |
|---------|-------------|
| **Neural Networks** | Function approximators that learn patterns |
| **Loss Function** | Measures how wrong your predictions are |
| **Optimizer** | Updates weights to reduce loss (Adam, SGD) |
| **Backpropagation** | How gradients flow backward to update weights |
| **Epochs** | Number of times model sees entire dataset |
| **Batch Size** | Number of samples per weight update |
| **Validation** | Check performance on unseen data during training |

### 🧠 Meme Version

- **Loss** = ego damage
- **Optimizer** = gym trainer
- **Backprop** = character development arc
- **Overfitting** = studying only past year papers
- **Regularization** = touching grass

---

## 0.5 Dataset Mindset (Where Pros Differ) 📊

**You said you want to train yourself — W mindset 👑**

### Dataset Rules

For **every phase**, you will:

1. ✅ Collect OR download raw data
2. ✅ Clean it (remove corrupted files)
3. ✅ Label it (organize by classes)
4. ✅ Split it (train/val/test)
5. ✅ Augment it (flip, rotate, noise)
6. ✅ Train on it

**No shortcuts.**

### Folder Example

```
data/
└── emotion_face/
    ├── train/
    │   ├── happy/
    │   ├── sad/
    │   ├── angry/
    │   ├── neutral/
    │   ├── fear/
    │   └── surprise/
    ├── val/
    │   ├── happy/
    │   ├── sad/
    │   └── ...
    └── test/
        ├── happy/
        ├── sad/
        └── ...
```

This structure repeats for **voice, text, pose**, etc.

---

## 0.6 Training Pipeline (Universal for ALL Phases) 🔁

Every model you train will follow this ritual:

### Training Loop Flow

```
Load data
    ↓
Preprocess (normalize, resize)
    ↓
Build model (CNN, RNN, Transformer)
    ↓
Choose loss + optimizer
    ↓
Train (epochs)
    ↓
Validate (check val_loss)
    ↓
Evaluate (test accuracy, confusion matrix)
    ↓
Save model (weights + config)
```

**If you master this ONCE, you can train any model in existence.**

### Example Training Code Structure

```python
# 1. Load data
train_loader, val_loader, test_loader = load_data()

# 2. Build model
model = EmotionCNN(num_classes=7)

# 3. Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss = validate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
# 5. Evaluate
test_accuracy = evaluate(model, test_loader)

# 6. Save
torch.save(model.state_dict(), 'models/emotion_cnn.pth')
```

---

## 0.7 Model Saving & Reuse (Future-You Will Thank You) 💾

You will **ALWAYS** save:

1. ✅ **Model weights** (`model.pth`)
2. ✅ **Label mappings** (`label_map.json`)
3. ✅ **Config** (input size, classes, preprocessing)

### Why?

**Training ≠ Inference**

- Training = gym 🏋️
- Inference = match day ⚽

When you deploy, you only load weights and predict. No training data needed.

### Example Save Format

```
models/
├── emotion_face_cnn.pth          # Model weights
├── emotion_face_config.json      # Input size, classes
└── emotion_face_labels.json      # {0: 'happy', 1: 'sad', ...}
```

---

## 0.8 Git + Experiment Tracking 📝

### Minimum Git Discipline

- ✅ One folder per phase
- ✅ Meaningful commits
- ✅ README per phase

### Example Commit Messages

```bash
feat: trained CNN for facial emotion (72% val acc)
fix: corrected data augmentation pipeline
docs: added Phase 1 README with results
data: added FER2013 dataset preprocessing script
```

**Sexy AND professional.**

### Git Workflow

```bash
git add .
git commit -m "feat: completed Phase 0 setup"
git push origin main
```

---

## 0.9 Phase 0 Final Deliverables ✅

Before moving to **Phase 1**, you MUST have:

- ✅ Working Python environment
- ✅ Chosen DL framework (PyTorch)
- ✅ Clear project folder structure
- ✅ Understood training loop
- ✅ Trained at least ONE simple model (any dataset)

### 🎮 Boss-Level Checkpoint

**Challenge:** Train a simple CNN on MNIST (digit recognition) to prove you understand the training loop.

Success criteria:
- 95%+ test accuracy
- Model saved as `models/mnist_cnn.pth`
- Training script in `training/mnist_train.py`

**Once this is done, you're ready for Phase 1.**

---

## 📚 Resources to Master Phase 0

### PyTorch Basics
- [Official PyTorch 60-min Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [PyTorch Image Classification Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

### ML Fundamentals
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

### Datasets for Practice
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)

---

## 🚀 Next Phase Preview

**Phase 1: Facial Emotion Recognition**
- Dataset: FER2013 or AffectNet
- Model: CNN (ResNet-18 or EfficientNet)
- Output: Real-time webcam emotion detection

**Stay locked in. 🔒**
