# 🧠 Psychologist AI

**A Multi-Modal AI System for Emotion Recognition & Mental Health Support**

---

## 🎯 Project Vision

This is **ONE integrated system** that combines:

- 😊 **Facial Emotion Recognition** (Computer Vision)
- 🗣️ **Voice Emotion Analysis** (Audio Processing)
- 📝 **Text Sentiment Analysis** (NLP)
- 🤲 **Gesture & Posture Detection** (Pose Estimation)
- 🧬 **Multi-Modal Fusion** (Integration)
- 💬 **Conversational AI** (Therapy Bot)
- 🧠 **Memory & Context** (User Profiles)
- 🌐 **Web Interface** (Deployment)

---

## 📂 Project Structure

```
PSYCHOLOGIST_AI/
│
├── data/              # All datasets (face, voice, text, etc.)
│   ├── emotion_face/
│   ├── emotion_voice/
│   └── emotion_text/
│
├── models/            # Saved model weights (.pth, .h5)
│
├── training/          # Training scripts for each phase
│   └── training_template.py
│
├── inference/         # Real-time prediction scripts
│
├── fusion/            # Multi-modal integration logic
│
├── memory/            # Conversation history & user profiles
│
├── ui/                # Web/Desktop interface
│
├── reports/           # Training logs, metrics, visualizations
│
├── requirements.txt   # Python dependencies
├── .gitignore        # Git ignore rules
├── README.md         # This file
└── PHASE_0_README.md # Phase 0 setup guide
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Dwarkadhish2005/Psychologist-AI.git
cd Psychologist-AI
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed ✓')"
```

---

## 📋 Development Phases

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 0** | Foundation & Setup | ✅ **COMPLETE** |
| **Phase 1** | Facial Emotion Recognition | � **IN PROGRESS** |
| **Phase 2** | Voice Emotion Analysis | 📅 Planned |
| **Phase 3** | Text Sentiment Analysis | 📅 Planned |
| **Phase 4** | Gesture & Posture Detection | 📅 Planned |
| **Phase 5** | Multi-Modal Fusion | 📅 Planned |
| **Phase 6** | Conversational AI | 📅 Planned |
| **Phase 7** | Memory & Personalization | 📅 Planned |
| **Phase 8** | Deployment & UI | 📅 Planned |

---

## 🛠️ Tech Stack

### Deep Learning Framework
- **PyTorch** 🔥 (Chosen for flexibility & research capabilities)

### Core Libraries
- `numpy`, `pandas` - Data manipulation
- `opencv-python` - Computer vision
- `librosa` - Audio processing
- `transformers` - NLP models
- `mediapipe` - Pose estimation
- `matplotlib`, `seaborn` - Visualization

### Deployment
- `Flask` / `FastAPI` - Web API
- `Streamlit` - Interactive UI

---

## 📚 Documentation

- [Phase 0: Foundation Setup](PHASE_0_README.md) - **Complete** ✅
- [Phase 1: Facial Emotion Recognition](PHASE_1_README.md) - **In Progress** 🚀
- Phase 2: Voice Emotion Analysis - Coming soon
- Phase 3: Text Sentiment Analysis - Coming soon

---

## 🎓 Learning Resources

### PyTorch
- [Official PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch 60-min Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

### Computer Vision
- [Stanford CS231n](http://cs231n.stanford.edu/)
- [PyImageSearch](https://pyimagesearch.com/)

### NLP
- [Hugging Face Course](https://huggingface.co/course)
- [Fast.ai NLP](https://www.fast.ai/)

---

## 🤝 Contributing

This is a personal learning project, but suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/awesome-feature`)
3. Commit changes (`git commit -m 'feat: add awesome feature'`)
4. Push to branch (`git push origin feature/awesome-feature`)
5. Open a Pull Request

---

## 📝 Git Workflow

```bash
# Stage changes
git add .

# Commit with meaningful message
git commit -m "feat: completed Phase 0 setup"

# Push to GitHub
git push origin main
```

### Commit Message Convention

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `data:` - Dataset changes
- `model:` - Model architecture changes
- `train:` - Training improvements

---

## 📊 Current Progress

### Phase 0 Deliverables ✅

- ✅ Project structure created
- ✅ Python environment configured
- ✅ PyTorch installed & verified
- ✅ Training pipeline template ready
- ✅ Git repository initialized
- ✅ Documentation complete

### Phase 1 Progress 🚀

- ✅ Dataset structure created (5 emotion classes)
- ✅ Preprocessing pipeline implemented
- ✅ EmotionCNN model architecture ready
- ✅ Training script complete
- ✅ Real-time webcam inference script ready
- ⏳ **Next: Download dataset & train model**

**Current Phase: Facial Emotion Recognition** 🎭

---

## 📧 Contact

**Dwarkadhish**  
GitHub: [@Dwarkadhish2005](https://github.com/Dwarkadhish2005)

---

## 📜 License

This project is for educational purposes.

---

**"Building AI that understands emotions, one phase at a time."** 🧠✨
