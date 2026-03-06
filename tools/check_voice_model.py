"""
Voice Model Diagnostic Script
==============================
Check if the model is configured correctly for all 5 emotions including happy.
"""

import torch
import sys
import json
import glob
import os
from pathlib import Path
from collections import Counter

# Add training modules to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'training' / 'voice'))

from voice_emotion_model import VoiceEmotionModel, StressDetector
from dataset_utils import TARGET_EMOTIONS, TARGET_EMOTION_TO_IDX


def check_model_configuration():
    """Check if model is correctly configured."""
    print("=" * 70)
    print("1. MODEL CONFIGURATION CHECK")
    print("=" * 70)
    
    model_path = project_root / 'models' / 'voice_emotion' / 'emotion_model_best.pth'
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return False
    
    # Load model
    model = VoiceEmotionModel(input_dim=48, num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print(f"✅ Model loaded successfully from {model_path}")
    print(f"\nModel Architecture:")
    print(f"  Input dim: {model.input_dim}")
    print(f"  Num classes: {model.num_classes}")
    print(f"  Class names: {model.class_names}")
    print(f"  Expected classes: {TARGET_EMOTIONS}")
    
    # Verify class names match
    if model.class_names == TARGET_EMOTIONS:
        print(f"✅ Class names match expected emotions")
    else:
        print(f"❌ Class names mismatch!")
        print(f"   Expected: {TARGET_EMOTIONS}")
        print(f"   Got: {model.class_names}")
        return False
    
    # Check if happy is in the list
    if 'happy' in model.class_names:
        happy_idx = model.class_names.index('happy')
        print(f"✅ 'happy' is at index {happy_idx}")
    else:
        print(f"❌ 'happy' not found in class names!")
        return False
    
    return True


def check_label_mapping():
    """Check if label mapping includes happy."""
    print("\n" + "=" * 70)
    print("2. LABEL MAPPING CHECK")
    print("=" * 70)
    
    print(f"\nTarget Emotions: {TARGET_EMOTIONS}")
    print(f"Target Emotion to Index: {TARGET_EMOTION_TO_IDX}")
    
    if 'happy' in TARGET_EMOTIONS:
        print(f"✅ 'happy' is in TARGET_EMOTIONS at index {TARGET_EMOTIONS.index('happy')}")
    else:
        print(f"❌ 'happy' not in TARGET_EMOTIONS!")
        return False
    
    if 'happy' in TARGET_EMOTION_TO_IDX:
        print(f"✅ 'happy' mapping exists: happy → {TARGET_EMOTION_TO_IDX['happy']}")
    else:
        print(f"❌ 'happy' not in TARGET_EMOTION_TO_IDX!")
        return False
    
    return True


def check_dataset_distribution():
    """Check if dataset contains happy samples."""
    print("\n" + "=" * 70)
    print("3. DATASET DISTRIBUTION CHECK")
    print("=" * 70)
    
    data_root = project_root / 'data' / 'voice_emotion'
    
    # Check RAVDESS
    ravdess_path = data_root / 'RAVDESS'
    if ravdess_path.exists():
        ravdess_files = glob.glob(str(ravdess_path / '**' / '*.wav'), recursive=True)
        
        # Parse RAVDESS emotions (code 03 = happy)
        emotion_map = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        
        ravdess_emotions = []
        for f in ravdess_files:
            filename = os.path.basename(f)
            parts = filename.replace('.wav', '').split('-')
            if len(parts) == 7:
                emotion_code = parts[2]
                emotion = emotion_map.get(emotion_code, 'unknown')
                ravdess_emotions.append(emotion)
        
        print(f"\nRAVDESS Dataset ({len(ravdess_files)} files):")
        ravdess_counter = Counter(ravdess_emotions)
        for emotion, count in sorted(ravdess_counter.items()):
            emoji = "✅" if emotion == 'happy' else "  "
            print(f"{emoji}  {emotion}: {count}")
        
        if 'happy' in ravdess_counter and ravdess_counter['happy'] > 0:
            print(f"✅ RAVDESS contains {ravdess_counter['happy']} happy samples")
        else:
            print(f"❌ RAVDESS has no happy samples!")
    else:
        print(f"❌ RAVDESS not found at {ravdess_path}")
    
    # Check TESS
    tess_path = data_root / 'TESS'
    if tess_path.exists():
        tess_files = glob.glob(str(tess_path / '**' / '*.wav'), recursive=True)
        
        # Parse TESS emotions (last part of filename)
        tess_emotions = []
        for f in tess_files:
            filename = os.path.basename(f)
            parts = filename.replace('.wav', '').split('_')
            if len(parts) >= 3:
                emotion = parts[-1]
                tess_emotions.append(emotion)
        
        print(f"\nTESS Dataset ({len(tess_files)} files):")
        tess_counter = Counter(tess_emotions)
        for emotion, count in sorted(tess_counter.items()):
            emoji = "✅" if emotion == 'happy' else "  "
            print(f"{emoji}  {emotion}: {count}")
        
        if 'happy' in tess_counter and tess_counter['happy'] > 0:
            print(f"✅ TESS contains {tess_counter['happy']} happy samples")
        else:
            print(f"❌ TESS has no happy samples!")
    else:
        print(f"❌ TESS not found at {tess_path}")
    
    return True


def check_training_data():
    """Check if training splits contain happy samples."""
    print("\n" + "=" * 70)
    print("4. TRAINING DATA CHECK")
    print("=" * 70)
    
    splits_path = project_root / 'data' / 'voice_emotion' / 'dataset_splits.json'
    
    if not splits_path.exists():
        print(f"❌ dataset_splits.json not found at {splits_path}")
        print("   Run: python training/voice/download_datasets.py")
        return False
    
    with open(splits_path) as f:
        splits = json.load(f)
    
    # Count happy samples in each split
    emotion_map_ravdess = {'03': 'happy'}
    
    for split_name in ['train', 'val', 'test']:
        files = splits.get(split_name, [])
        happy_count = 0
        
        for file_path in files:
            filename = os.path.basename(file_path)
            
            # Check RAVDESS (code 03)
            if 'Actor_' in file_path:
                parts = filename.replace('.wav', '').split('-')
                if len(parts) == 7 and parts[2] == '03':
                    happy_count += 1
            
            # Check TESS (filename ends with _happy.wav)
            elif filename.endswith('_happy.wav'):
                happy_count += 1
        
        total = len(files)
        print(f"\n{split_name.upper()} SET:")
        print(f"  Total samples: {total}")
        print(f"  Happy samples: {happy_count}")
        
        if happy_count > 0:
            percentage = (happy_count / total) * 100
            print(f"  ✅ Contains {percentage:.1f}% happy samples")
        else:
            print(f"  ❌ No happy samples in {split_name} set!")
    
    return True


def test_model_predictions():
    """Test if model can predict happy."""
    print("\n" + "=" * 70)
    print("5. MODEL PREDICTION TEST")
    print("=" * 70)
    
    model_path = project_root / 'models' / 'voice_emotion' / 'emotion_model_best.pth'
    model = VoiceEmotionModel(input_dim=48, num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Test with random inputs
    print("\nTesting with 10 random inputs:")
    
    predicted_emotions = []
    for i in range(10):
        dummy_input = torch.randn(1, 48)
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = torch.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            predicted_emotion = model.class_names[predicted_idx]
            predicted_emotions.append(predicted_emotion)
            
            # Show detailed probabilities
            if i < 3:  # Show first 3 in detail
                print(f"\nTest {i+1}:")
                for emotion, prob in zip(model.class_names, probabilities[0]):
                    print(f"  {emotion}: {prob.item():.4f}")
                print(f"  → Predicted: {predicted_emotion}")
    
    print(f"\nPrediction Distribution (10 random tests):")
    pred_counter = Counter(predicted_emotions)
    for emotion, count in sorted(pred_counter.items()):
        emoji = "✅" if emotion == 'happy' else "  "
        print(f"{emoji}  {emotion}: {count}/10")
    
    if 'happy' in pred_counter:
        print(f"\n✅ Model CAN predict 'happy' (predicted {pred_counter['happy']} times)")
    else:
        print(f"\n⚠️  Model did NOT predict 'happy' in random tests")
        print(f"   This could mean:")
        print(f"   - Model is biased toward other emotions (check training)")
        print(f"   - Random features don't activate 'happy' neurons")
        print(f"   - Test with real audio features to confirm")
    
    return True


def main():
    print("\n" + "=" * 70)
    print("VOICE EMOTION MODEL DIAGNOSTICS")
    print("=" * 70)
    print("Checking if 'happy' emotion is properly configured and trained...\n")
    
    checks = [
        ("Model Configuration", check_model_configuration),
        ("Label Mapping", check_label_mapping),
        ("Dataset Distribution", check_dataset_distribution),
        ("Training Data", check_training_data),
        ("Model Predictions", test_model_predictions)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ Error in {check_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {check_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All checks passed! The model is properly configured for 'happy'.")
        print("\nIf microphone detection is not showing 'happy', possible reasons:")
        print("1. Training data imbalance (model biased to other emotions)")
        print("2. Input audio doesn't have happy characteristics")
        print("3. Feature extraction not capturing happy patterns")
        print("\nNext steps:")
        print("  - Check training accuracy: models/voice_emotion/config.json")
        print("  - Test with known happy audio files")
        print("  - Consider retraining with data augmentation")
    else:
        print("\n⚠️  Some checks failed. Review errors above.")


if __name__ == "__main__":
    main()
