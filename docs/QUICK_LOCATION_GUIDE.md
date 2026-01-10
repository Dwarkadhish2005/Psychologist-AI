# 🗂️ Quick File Location Reference

**Last Updated:** January 10, 2026  
**After file structure reorganization**

---

## 📍 Where to Find Everything

### **Documentation** 📚
```
All docs now in: docs/
├── README.md                    → (still in root - main entry point)
├── docs/PROJECT_STRUCTURE.md    → Architecture guide
├── docs/QUICK_REFERENCE.md      → Command reference
├── docs/ORGANIZATION_COMPLETE.md → Old organization summary
├── docs/FILE_STRUCTURE_ORGANIZED.md → This reorganization summary
└── docs/RUN_EVERYTHING.md       → All commands
```

### **Phase Documentation** 📖
```
├── docs/setup/          → GPU setup, Phase 0
├── docs/phase1/         → Face emotion (13 files)
├── docs/phase2/         → Voice emotion (2 files)
├── docs/phase3/         → Multi-modal fusion (5 files)
│   └── PHASE3_DETAILED_BREAKDOWN.md (moved here from root)
└── docs/phase4/         → Cognitive layer (5 files)
```

### **Training Reports** 📊
```
Old: reports/
New: assets/reports/
├── confusion_matrix_20260108_235349.png
├── training_plot_20260108_235349.png
└── training_history_20260108_235349.json
```

### **Inference Scripts** 🚀
```
Location: inference/
├── integrated_psychologist_ai.py        → Main system (all phases)
├── phase3_multimodal_fusion.py         → Phase 3 fusion (890 lines)
├── phase3_demo.py                      → Phase 3 demo
├── phase4_cognitive_layer.py           → Phase 4 cognitive (2500+ lines)
├── phase4_user_manager.py              → User management (345 lines)
├── demo_phase4_integration.py          → Phase 4 scenarios
├── demo_phase4_enhancements.py         → Phase 4.1/4.2 features
├── webcam_emotion_detection.py         → Face-only
├── microphone_emotion_detection.py     → Voice-only
└── dual_model_emotion_detection.py     → Dual face models
```

### **Trained Models** 🤖
```
Location: models/
├── face_emotion/
│   ├── emotion_cnn_best.pth                  → Phase 1 baseline
│   ├── emotion_cnn_phase15_specialist.pth    → Happy specialist
│   ├── emotion_cnn_phase15_best.pth          → Currently active
│   ├── labels.json
│   └── config.json
└── voice_emotion/
    ├── emotion_model_best.pth                → Original baseline
    ├── emotion_model_best_improved.pth       → Improved version
    ├── emotion_model_best_balanced.pth       → Currently active ✅
    ├── stress_model_best.pth                 → Stress detector
    ├── labels.json
    └── config.json
```

### **User Data** 👤
```
Location: data/user_memory/
├── users.json                          → User registry
├── archive/                            → Archived sessions
└── {user_id}_longterm_memory.json      → Per-user memory (created after session)
```

### **Training Scripts** 🎓
```
Location: training/
├── train_emotion_model.py              → Face training
├── train_phase_1_5_finetune.py         → Face fine-tuning
└── voice/
    ├── train_voice_emotion.py              → Original voice training
    ├── train_voice_emotion_improved.py     → Improved training
    └── train_voice_emotion_balanced.pth    → Best training (use this)
```

### **Utility Scripts** 🔧
```
Location: scripts/
├── check_gpu.py                        → Check GPU availability
└── check_system.py                     → Check system requirements
```

---

## 🔍 Quick Find Guide

### **"Where is the main README?"**
→ Root directory: `README.md` (unchanged)

### **"Where is PROJECT_STRUCTURE.md?"**
→ **MOVED:** `docs/PROJECT_STRUCTURE.md`

### **"Where are training reports?"**
→ **MOVED:** `assets/reports/` (was `reports/`)

### **"Where is QUICK_REFERENCE.md?"**
→ **MOVED:** `docs/QUICK_REFERENCE.md`

### **"Where is PHASE3_DETAILED_BREAKDOWN.md?"**
→ **MOVED:** `docs/phase3/PHASE3_DETAILED_BREAKDOWN.md`

### **"Where is the user manager code?"**
→ `inference/phase4_user_manager.py`

### **"Where are empty directories?"**
→ **DELETED:** `fusion/`, `memory/`, `ui/`, `config/` (were empty)

### **"Where is the main system script?"**
→ `inference/integrated_psychologist_ai.py` (unchanged)

---

## 📁 Root Directory Now Contains

```
Psychologist AI/
├── .gitignore           ← Git ignore rules
├── README.md            ← Main project overview
└── requirements.txt     ← Python dependencies

Only 3 files! Clean and organized! ✅
```

---

## 🎯 Quick Commands

### **Run Main System**
```bash
python inference/integrated_psychologist_ai.py
```

### **Read Architecture**
```bash
# Windows
type docs\PROJECT_STRUCTURE.md

# Linux/Mac
cat docs/PROJECT_STRUCTURE.md
```

### **List All Documentation**
```bash
# Windows
Get-ChildItem docs -Recurse -Filter *.md

# Linux/Mac
find docs -name "*.md"
```

### **View Training Reports**
```bash
# Windows
explorer assets\reports

# Linux/Mac
open assets/reports
```

---

## 📊 File Counts

| Category | Location | Files |
|----------|----------|-------|
| Documentation | `docs/` | 28 files |
| Inference Scripts | `inference/` | 10 files |
| Trained Models | `models/` | 11 files |
| Training Scripts | `training/` | 13 files |
| Utility Scripts | `scripts/` | 2 files |
| Tests | `tests/` | 1 file |
| Reports | `assets/reports/` | 3 files |
| Visualizations | `assets/visualizations/` | 3 files |
| **Root Files** | `.` | **3 files** ✅ |

---

## 🚫 What Was Deleted

**Empty directories only:**
- ❌ `fusion/` - Empty, unused
- ❌ `memory/` - Empty, replaced by Phase 4
- ❌ `ui/` - Empty, unused
- ❌ `config/` - Empty, unused

**No code, models, or data was deleted!** ✅

---

## ✅ Verification

All systems operational:
```bash
python -c "from inference.integrated_psychologist_ai import IntegratedPsychologistAI; print('✅ All working')"
```

Expected output:
```
✅ All working
```

---

**Need more details?** See [docs/FILE_STRUCTURE_ORGANIZED.md](FILE_STRUCTURE_ORGANIZED.md) for complete reorganization report.
