"""
Move 10-15% of training images to validation set
=================================================
This script moves (not copies) images from train to val folders
to create a proper train/val split.

Author: Psychologist AI Team
"""

import os
import shutil
import random
from pathlib import Path

# Configuration
DATA_DIR = 'data/face_emotion'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
VAL_SPLIT = 0.15  # 15% for validation

def move_images_to_val(class_name):
    """
    Move 15% of images from train to val for a specific class.
    
    Args:
        class_name: Name of emotion class (e.g., 'angry', 'happy')
    """
    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    val_class_dir = os.path.join(VAL_DIR, class_name)
    
    # Create val class directory if it doesn't exist
    os.makedirs(val_class_dir, exist_ok=True)
    
    # Get all images in train directory
    train_images = [f for f in os.listdir(train_class_dir) 
                    if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    
    if len(train_images) == 0:
        print(f"  ⚠️  {class_name}: No images found in train folder")
        return
    
    # Calculate number of images to move
    num_val = int(len(train_images) * VAL_SPLIT)
    
    if num_val == 0:
        num_val = 1  # Move at least 1 image
    
    # Randomly select images to move
    random.shuffle(train_images)
    images_to_move = train_images[:num_val]
    
    # Move images
    moved_count = 0
    for img_name in images_to_move:
        src = os.path.join(train_class_dir, img_name)
        dst = os.path.join(val_class_dir, img_name)
        
        try:
            shutil.move(src, dst)
            moved_count += 1
        except Exception as e:
            print(f"    Error moving {img_name}: {e}")
    
    # Final counts
    train_remaining = len(os.listdir(train_class_dir))
    val_total = len([f for f in os.listdir(val_class_dir) 
                     if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
    
    print(f"  ✓ {class_name:10s}: Moved {moved_count} images | "
          f"Train: {train_remaining} | Val: {val_total}")

def main():
    """Main function to move images for all classes"""
    
    print("=" * 60)
    print("MOVING IMAGES FROM TRAIN TO VALIDATION")
    print("=" * 60)
    print(f"Validation split: {VAL_SPLIT * 100:.0f}%")
    print("=" * 60)
    
    # Get all classes from train directory
    classes = [d for d in os.listdir(TRAIN_DIR) 
               if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    
    print(f"\nFound {len(classes)} classes: {', '.join(classes)}")
    print("\nMoving images...\n")
    
    # Move images for each class
    for class_name in sorted(classes):
        move_images_to_val(class_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)
    
    total_train = 0
    total_val = 0
    
    for class_name in sorted(classes):
        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        val_class_dir = os.path.join(VAL_DIR, class_name)
        
        train_count = len([f for f in os.listdir(train_class_dir) 
                          if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
        
        if os.path.exists(val_class_dir):
            val_count = len([f for f in os.listdir(val_class_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
        else:
            val_count = 0
        
        total_train += train_count
        total_val += val_count
        
        print(f"{class_name:10s}: Train={train_count:5d} | Val={val_count:4d}")
    
    print("-" * 60)
    print(f"{'TOTAL':10s}: Train={total_train:5d} | Val={total_val:4d}")
    print("=" * 60)
    
    print("\n✅ Dataset split complete!")
    print(f"Train: {total_train} images ({100 * total_train / (total_train + total_val):.1f}%)")
    print(f"Val:   {total_val} images ({100 * total_val / (total_train + total_val):.1f}%)")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Confirm before running
    print("\n⚠️  WARNING: This will MOVE images from train to val folders.")
    print("This operation cannot be easily undone.")
    response = input("\nProceed? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        main()
    else:
        print("Operation cancelled.")
