# 🚀 Quick Reference Card

**Psychologist AI - Organized Project Structure**

---

## 📂 Directory Overview

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| **assets/** | Visualizations & images | 3 PNG files |
| **config/** | Configuration files | (Ready for future) |
| **data/** | Training datasets | 46K+ files |
| **diagnostics/** | Testing tools | 2 diagnostic scripts |
| **docs/** | All documentation | 25 documents (organized) |
| **inference/** | Real-time systems | 8 detection engines |
| **models/** | Trained models | 12 model files |
| **scripts/** | Utility scripts | 2 utility scripts |
| **tests/** | Test suites | 1 comprehensive test |
| **training/** | Training pipelines | 22 training files |

---

## 📖 Documentation Structure

```
docs/
├── DOCUMENTATION_INDEX.md      # 📚 Start here for navigation
├── RUN_EVERYTHING.md           # All commands
├── models_guide.md             # Model architecture
│
├── setup/                      # Installation & GPU (3 files)
│   ├── GPU_SETUP_GUIDE.md
│   ├── GPU_SETUP_SUCCESS.md
│   └── PHASE_0_README.md
│
├── phase1/                     # Face Emotion (13 files)
│   ├── PHASE_1_5_START_HERE.md ⭐ Start here
│   ├── PHASE_1_5_QUICKSTART.md
│   ├── DUAL_MODEL_STRATEGY.md
│   └── ...
│
├── phase2/                     # Voice Emotion (2 files)
│   ├── PHASE_2_USER_GUIDE.md
│   └── VOICE_MODEL_DIAGNOSIS.md
│
└── phase3/                     # Multi-Modal Fusion (4 files)
    ├── README_PHASE3.md
    ├── PHASE3_QUICK_START.md
    ├── PHASE3_DOCUMENTATION.md
    └── PHASE3_COMPLETION_SUMMARY.md
```

---

## 🚀 Common Commands

### **System Operations**
```bash
# Check GPU
python scripts/check_gpu.py

# Check system requirements
python scripts/check_system.py

# Install dependencies
pip install -r requirements.txt
```

### **Running Inference**
```bash
# Complete integrated system (ALL 3 PHASES)
python inference/integrated_psychologist_ai.py

# Face emotion only
python inference/dual_model_emotion_detection.py
python inference/webcam_emotion_detection.py

# Voice emotion only
python inference/microphone_emotion_detection.py

# Phase 3 demo with visualizations
python inference/phase3_demo.py
```

### **Training**
```bash
# Train face emotion model
python training/train_emotion_model.py

# Fine-tune Phase 1.5
python training/train_phase_1_5_finetune.py

# Train voice emotion (balanced - BEST)
python training/voice/train_voice_emotion_balanced.py
```

### **Testing**
```bash
# Comprehensive Phase 3 test (10 scenarios)
python tests/test_phase3_final.py

# Voice model diagnostics
python diagnostics/check_voice_model.py

# Audio testing
python diagnostics/test_happy_audio.py
```

---

## 📁 Key Files

### **Documentation**
| File | Purpose |
|------|---------|
| [README.md](../README.md) | Main project overview |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Complete architecture (200+ lines) |
| [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) | Navigation hub |
| [ORGANIZATION_COMPLETE.md](ORGANIZATION_COMPLETE.md) | Reorganization summary |

### **Inference (Production)**
| File | Purpose |
|------|---------|
| [inference/integrated_psychologist_ai.py](inference/integrated_psychologist_ai.py) | Complete system (all 3 phases) |
| [inference/phase3_multimodal_fusion.py](inference/phase3_multimodal_fusion.py) | Fusion engine (890 lines) |
| [inference/phase3_demo.py](inference/phase3_demo.py) | Demo & visualizations |

### **Training**
| File | Purpose |
|------|---------|
| [training/train_emotion_model.py](training/train_emotion_model.py) | Face emotion training |
| [training/voice/train_voice_emotion_balanced.py](training/voice/train_voice_emotion_balanced.py) | Voice emotion (BEST) |

### **Testing**
| File | Purpose |
|------|---------|
| [tests/test_phase3_final.py](tests/test_phase3_final.py) | Comprehensive tests (10 scenarios) |
| [diagnostics/check_voice_model.py](diagnostics/check_voice_model.py) | Voice validation |

---

## 🎯 Quick Start Paths

### **New User**
1. Read [README.md](README.md)
2. Read [docs/phase3/PHASE3_QUICK_START.md](docs/phase3/PHASE3_QUICK_START.md)
3. Run `python inference/integrated_psychologist_ai.py`

### **Understanding Architecture**
1. Read [docs/PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
2. Browse [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md)
3. Check phase-specific docs

### **Training Models**
1. Read [docs/phase1/PHASE_1_5_START_HERE.md](docs/phase1/PHASE_1_5_START_HERE.md) (face)
2. Read [docs/phase2/PHASE_2_USER_GUIDE.md](docs/phase2/PHASE_2_USER_GUIDE.md) (voice)
3. Run training scripts

### **Testing System**
1. Run `python tests/test_phase3_final.py`
2. Run `python diagnostics/check_voice_model.py`
3. Check test outputs

---

## 📊 System Capabilities

| Phase | Capability | Performance |
|-------|-----------|-------------|
| **Phase 1** | Face Emotion (7 emotions) | 62.57% accuracy |
| **Phase 2** | Voice Emotion + Stress | 44% overall, 80% happy |
| **Phase 3** | Multi-Modal Fusion (15 states) | 9/10 scenarios pass |

**Status:** ✅ All phases complete and production-ready

---

## 🏗️ Phase 3 Architecture

```
4-Layer System:
├── Layer 1: Signal Normalization (reliability weighting)
├── Layer 2: Temporal Reasoning (30-frame memory)
├── Layer 3: Fusion Logic (psychology rules)
└── Layer 4: Psychological Reasoning (mental states)

15 Mental States:
• CALM, JOYFUL, HAPPY_UNDER_STRESS
• ANXIOUS, STRESSED, OVERWHELMED, CONFUSED
• EMOTIONALLY_MASKED, EMOTIONALLY_FLAT, EMOTIONALLY_UNSTABLE
• DEPRESSED, ANGRY, FEARFUL, MIXED_EMOTIONS, UNKNOWN

4 Risk Levels: LOW, MODERATE, HIGH, CRITICAL
```

---

## 🔗 Navigation Links

### **Main Documentation**
- [README.md](README.md) - Project overview
- [docs/PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Architecture
- [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md) - All docs

### **Phase Guides**
- [docs/phase1/PHASE_1_5_START_HERE.md](docs/phase1/PHASE_1_5_START_HERE.md) - Face
- [docs/phase2/PHASE_2_USER_GUIDE.md](docs/phase2/PHASE_2_USER_GUIDE.md) - Voice
- [docs/phase3/PHASE3_QUICK_START.md](docs/phase3/PHASE3_QUICK_START.md) - Fusion

### **Technical Details**
- [docs/phase1/DUAL_MODEL_STRATEGY.md](docs/phase1/DUAL_MODEL_STRATEGY.md) - Architecture
- [docs/phase3/PHASE3_DOCUMENTATION.md](docs/phase3/PHASE3_DOCUMENTATION.md) - Complete docs

---

## ✅ Organization Benefits

✓ **Clear Structure** - Everything in its place  
✓ **Easy Navigation** - Documentation index  
✓ **Professional Layout** - Follows best practices  
✓ **Phase Separation** - Organized by development phase  
✓ **Quick Access** - Common operations easy to find  
✓ **Production Ready** - Clean, deployable structure  

---

**Last Updated:** January 10, 2026
