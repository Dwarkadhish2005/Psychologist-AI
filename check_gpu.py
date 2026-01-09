"""
GPU Detection and PyTorch CUDA Setup Check
==========================================
Verify if CUDA-enabled PyTorch is installed and GPU is available.
"""

import torch
import sys

print("=" * 60)
print("PYTORCH GPU/CUDA SETUP CHECK")
print("=" * 60)

# Check PyTorch version
print(f"\nPyTorch Version: {torch.__version__}")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Test tensor operations
    print("\n" + "-" * 60)
    print("Testing GPU Tensor Operations...")
    print("-" * 60)
    
    device = torch.device("cuda")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.mm(x, y)
    
    print(f"✓ Matrix multiplication successful on {device}")
    print(f"  Result shape: {z.shape}")
    print(f"  Result device: {z.device}")
    
    print("\n GPU is READY for PyTorch!")
    
else:
    print("\n CUDA not available")
    print("\nTo enable GPU:")
    print("1. Check if you have an NVIDIA GPU")
    print("2. Install CUDA-enabled PyTorch:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\n3. If you don't have NVIDIA GPU, training will use CPU (slower)")

print("\n" + "=" * 60)
print("DEVICE RECOMMENDATION FOR TRAINING")
print("=" * 60)

if cuda_available:
    print("device = torch.device('cuda')")
    print(" Use GPU for training (10-20x faster)")
else:
    print("device = torch.device('cpu')")
    print(" Training will be slower on CPU")

print("=" * 60)
