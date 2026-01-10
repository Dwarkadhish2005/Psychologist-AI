# GPU Setup Guide for Phase 1.5

**Current Status:** PyTorch 2.7.1+cpu (CPU-only)  
**Target:** PyTorch with CUDA support for GPU acceleration

---

## 🎯 Quick Setup (3 Steps)

### Step 1: Uninstall CPU PyTorch
```powershell
pip uninstall torch torchvision torchaudio -y
```

### Step 2: Install CUDA PyTorch
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify GPU
```powershell
python check_gpu.py
```

---

## 📊 Expected Speed Improvement

### Phase 1.5 Training Time:

**Current (CPU):**
- Epoch 1: 2 min 5 sec
- Total (9 epochs): ~20 minutes
- Speed: 5-6 it/s

**After GPU:**
- Epoch 1: ~6-10 seconds
- Total (9 epochs): ~2-3 minutes
- Speed: 15-20 it/s
- **Improvement: 10-20x faster!**

---

## 🔍 What You Have Now

### Current PyTorch:
```
PyTorch: 2.7.1+cpu
CUDA Available: False
```

This is CPU-only version. Works but SLOW for training.

### After Installation:
```
PyTorch: 2.7.1+cu121
CUDA Available: True
GPU: [Your NVIDIA GPU Name]
```

---

## 💻 System Requirements

### To use GPU, you need:
1. ✅ **NVIDIA GPU** (GeForce, RTX, etc.)
2. ✅ **NVIDIA Drivers** (latest version)
3. ✅ **CUDA Toolkit** (auto-installed with PyTorch)

### Check if you have NVIDIA GPU:
```powershell
nvidia-smi
```

If this command works → You have NVIDIA GPU!  
If error → You might have Intel/AMD GPU (CPU-only)

---

## 🚀 Installation Commands

Copy-paste these commands one by one:

### PowerShell:
```powershell
# Step 1: Uninstall CPU version
pip uninstall torch torchvision torchaudio -y

# Step 2: Install GPU version (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 3: Verify installation
python check_gpu.py
```

---

## ✅ Verification

After installation, run:
```powershell
python check_gpu.py
```

**Expected output (if GPU available):**
```
============================================================
PYTORCH GPU/CUDA SETUP CHECK
============================================================

PyTorch Version: 2.7.1+cu121
CUDA Available: True
CUDA Version: 12.1
Number of GPUs: 1

GPU 0: NVIDIA GeForce RTX 3060
  Memory Allocated: 0.00 GB
  Memory Cached: 0.00 GB
  Total Memory: 12.00 GB

------------------------------------------------------------
Testing GPU Tensor Operations...
------------------------------------------------------------
✓ Matrix multiplication successful on cuda:0

✅ GPU is READY for PyTorch!
```

**If CPU-only (no NVIDIA GPU):**
```
CUDA Available: False

❌ CUDA not available
```

---

## 🎯 After GPU Setup

Once GPU is ready, re-run Phase 1.5:
```powershell
python training/train_phase_1_5_finetune.py
```

You should see:
```
Device: cuda  ← Instead of "cpu"!
```

Training will be **10-20x faster!**

---

## 🔧 Troubleshooting

### Issue 1: "nvidia-smi not found"
**Cause:** No NVIDIA GPU or drivers not installed  
**Solution:** 
- Check if you have NVIDIA GPU in Device Manager
- Install latest NVIDIA drivers from nvidia.com

### Issue 2: GPU detected but PyTorch still uses CPU
**Cause:** Installed CPU version by mistake  
**Solution:**
```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue 3: CUDA version mismatch
**Cause:** CUDA version incompatibility  
**Solution:** Try different CUDA version:
```powershell
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## 📊 Current Results (CPU-only)

Your Phase 1.5 training on CPU achieved:
```
Phase 1 (Main):
  - Overall: 62.57%
  - Disgust: 30.63%
  - Fear: 32.62%

Phase 1.5 (Specialist):
  - Overall: 62.43% (-0.14%)
  - Disgust: 61.26% (+30.63%!) 🎯
  - Fear: 34.96% (+2.34%)

Result: ✅ Phase 1.5 is excellent specialist for disgust!
```

**This is SUCCESS!** Phase 1.5 specialized in minority classes.

---

## ✅ Summary

**What You Have:**
- ✅ Phase 1 (Main): emotion_cnn_best.pth
- ✅ Phase 1.5 (Specialist): emotion_cnn_phase15_specialist.pth
- ✅ Dual-model strategy: Use both intelligently
- ✅ Training successful: +30% disgust improvement!

**What You Need:**
- 🟡 GPU PyTorch: For faster training (optional but recommended)

**Next Steps:**
1. Install GPU PyTorch (commands above)
2. Verify GPU: `python check_gpu.py`
3. Optional: Re-run Phase 1.5 with GPU (10x faster)
4. Test dual-model: `python inference/dual_model_emotion_detection.py`

---

## 🎉 You're Ready!

Both models are trained and ready to use:
- **Phase 1:** General emotion (62.57%)
- **Phase 1.5:** Disgust expert (61.26% disgust recall!)

GPU setup is optional but highly recommended for future training.

---

*Run these commands to get GPU support:*
```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python check_gpu.py
```
