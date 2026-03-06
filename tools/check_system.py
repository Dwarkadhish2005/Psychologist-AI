"""
Comprehensive System Check for Psychologist AI - Phase 1
=========================================================
Verify all components are ready for training.
"""

import os
import sys

def check_directories():
    """Check all required directories exist"""
    print("=" * 60)
    print("CHECKING DIRECTORIES")
    print("=" * 60)
    
    required_dirs = [
        'data/face_emotion/train',
        'data/face_emotion/val',
        'data/face_emotion/test',
        'models/face_emotion',
        'training',
        'inference',
        'reports'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        status = "[OK]" if exists else "[MISSING]"
        print(f"{status} {dir_path}")
        if not exists:
            all_ok = False
    
    return all_ok

def check_dataset():
    """Check dataset has images"""
    print("\n" + "=" * 60)
    print("CHECKING DATASET")
    print("=" * 60)
    
    splits = ['train', 'val', 'test']
    total_images = 0
    
    for split in splits:
        split_path = f'data/face_emotion/{split}'
        if not os.path.exists(split_path):
            print(f"[MISSING] {split_path}")
            continue
        
        classes = [d for d in os.listdir(split_path) 
                  if os.path.isdir(os.path.join(split_path, d))]
        
        split_count = 0
        for class_name in sorted(classes):
            class_path = os.path.join(split_path, class_name)
            images = [f for f in os.listdir(class_path) 
                     if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            count = len(images)
            split_count += count
            print(f"  {split}/{class_name:10s}: {count:5d} images")
        
        total_images += split_count
        print(f"  {split.upper()} TOTAL: {split_count} images\n")
    
    print(f"TOTAL DATASET: {total_images} images")
    return total_images > 0

def check_python_files():
    """Check all required Python files exist"""
    print("\n" + "=" * 60)
    print("CHECKING PYTHON FILES")
    print("=" * 60)
    
    required_files = [
        'training/model.py',
        'training/preprocessing.py',
        'training/train_emotion_model.py',
        'training/split_dataset.py',
        'inference/webcam_emotion_detection.py'
    ]
    
    all_ok = True
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "[OK]" if exists else "[MISSING]"
        print(f"{status} {file_path}")
        if not exists:
            all_ok = False
    
    return all_ok

def check_imports():
    """Check critical imports work"""
    print("\n" + "=" * 60)
    print("CHECKING PYTHON IMPORTS")
    print("=" * 60)
    
    imports_to_check = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    all_ok = True
    for module, name in imports_to_check:
        try:
            __import__(module)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[MISSING] {name} - Run: pip install {module}")
            all_ok = False
    
    return all_ok

def check_model():
    """Check model can be instantiated"""
    print("\n" + "=" * 60)
    print("CHECKING MODEL")
    print("=" * 60)
    
    try:
        sys.path.append('training')
        from model import EmotionCNN
        
        # Test with 7 classes
        model = EmotionCNN(num_classes=7, input_size=48)
        params = sum(p.numel() for p in model.parameters())
        
        print(f"[OK] EmotionCNN can be instantiated")
        print(f"     Parameters: {params:,}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create model: {e}")
        return False

def check_preprocessing():
    """Check preprocessing works"""
    print("\n" + "=" * 60)
    print("CHECKING PREPROCESSING")
    print("=" * 60)
    
    try:
        sys.path.append('training')
        from preprocessing import preprocess_face
        import numpy as np
        
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = preprocess_face(dummy_img, target_size=(48, 48))
        
        if result.shape == (1, 48, 48):
            print(f"[OK] Preprocessing works correctly")
            print(f"     Output shape: {result.shape}")
            return True
        else:
            print(f"[ERROR] Unexpected shape: {result.shape}")
            return False
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return False

def main():
    """Run all checks"""
    print("\n")
    print("=" * 60)
    print("PSYCHOLOGIST AI - PHASE 1 SYSTEM CHECK")
    print("=" * 60)
    
    checks = [
        ("Directories", check_directories),
        ("Dataset", check_dataset),
        ("Python Files", check_python_files),
        ("Python Imports", check_imports),
        ("Model", check_model),
        ("Preprocessing", check_preprocessing)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n[ERROR] {name} check failed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n[OK] All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Run: python training/train_emotion_model.py")
        print("  2. Wait for training to complete (~30-50 epochs)")
        print("  3. Run: python inference/webcam_emotion_detection.py")
    else:
        print("\n[FAIL] Some checks failed. Please fix issues before training.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
