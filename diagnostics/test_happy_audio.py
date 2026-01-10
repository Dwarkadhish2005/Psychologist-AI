"""
Test Voice Model with Real Happy Audio Files
============================================
Test the model with actual happy audio samples to see if it can detect them.
"""

import torch
import sys
import glob
import os
from pathlib import Path

# Add training modules to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'training' / 'voice'))

from voice_emotion_model import VoiceEmotionModel
from feature_extraction import extract_all_features
from audio_preprocessing import preprocess_audio


def test_happy_audio_files():
    """Test model with real happy audio files."""
    print("=" * 70)
    print("TESTING MODEL WITH REAL HAPPY AUDIO FILES")
    print("=" * 70)
    
    # Load model (try balanced first, then improved, fall back to original)
    balanced_model_path = project_root / 'models' / 'voice_emotion' / 'emotion_model_best_balanced.pth'
    improved_model_path = project_root / 'models' / 'voice_emotion' / 'emotion_model_best_improved.pth'
    original_model_path = project_root / 'models' / 'voice_emotion' / 'emotion_model_best.pth'
    
    if balanced_model_path.exists():
        model_path = balanced_model_path
        print(f"Testing BALANCED model: {model_path.name}")
    elif improved_model_path.exists():
        model_path = improved_model_path
        print(f"Testing IMPROVED model: {model_path.name}")
    else:
        model_path = original_model_path
        print(f"Testing ORIGINAL model: {model_path.name}")
    
    model = VoiceEmotionModel(input_dim=48, num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Find happy audio files from both datasets
    data_root = project_root / 'data' / 'voice_emotion'
    
    # RAVDESS happy files (code 03)
    ravdess_path = data_root / 'RAVDESS'
    ravdess_happy_files = []
    for actor_folder in ravdess_path.glob('Actor_*'):
        for wav_file in actor_folder.glob('*-03-*.wav'):  # Code 03 = happy
            ravdess_happy_files.append(wav_file)
    
    # TESS happy files (end with _happy.wav)
    tess_path = data_root / 'TESS'
    tess_happy_files = list(tess_path.glob('**/*_happy.wav'))
    
    all_happy_files = ravdess_happy_files + tess_happy_files
    
    print(f"\nFound {len(all_happy_files)} happy audio files")
    print(f"  RAVDESS: {len(ravdess_happy_files)}")
    print(f"  TESS: {len(tess_happy_files)}")
    
    if len(all_happy_files) == 0:
        print("❌ No happy audio files found!")
        return
    
    # Test on first 20 happy files
    print(f"\nTesting on first 20 happy files...")
    print("-" * 70)
    
    predictions = []
    confidences = []
    
    for i, audio_file in enumerate(all_happy_files[:20]):
        try:
            # Preprocess audio
            audio_data, sr, quality = preprocess_audio(str(audio_file))
            
            # Extract features
            features_dict, features = extract_all_features(audio_data, sr)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                logits = model(features_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
                
                predicted_emotion = model.class_names[predicted_idx]
                predictions.append(predicted_emotion)
                confidences.append(confidence)
                
                # Show details
                filename = audio_file.name
                correct = "✅" if predicted_emotion == 'happy' else "❌"
                print(f"{correct} File {i+1}: {filename[:40]:<40} → {predicted_emotion} ({confidence:.2%})")
                
                # Show all probabilities for first few
                if i < 3:
                    print("    Probabilities:")
                    for emotion, prob in zip(model.class_names, probabilities[0]):
                        print(f"      {emotion}: {prob.item():.4f}")
        
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    from collections import Counter
    pred_counter = Counter(predictions)
    
    print(f"\nPrediction Distribution (20 happy files):")
    for emotion, count in sorted(pred_counter.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(predictions)) * 100
        emoji = "✅" if emotion == 'happy' else "  "
        print(f"{emoji}  {emotion}: {count}/20 ({percentage:.1f}%)")
    
    happy_count = pred_counter.get('happy', 0)
    
    if len(predictions) > 0:
        accuracy = (happy_count / len(predictions)) * 100
    else:
        print("\n❌ All audio files failed to process!")
        return None, None
    
    print(f"\n{'='*70}")
    if happy_count > 0:
        print(f"✅ Model CAN detect happy: {happy_count}/20 ({accuracy:.1f}% accuracy)")
        if accuracy < 50:
            print(f"⚠️  But accuracy is low! Model is biased.")
    else:
        print(f"❌ Model CANNOT detect happy: 0/20 (0% accuracy)")
        print(f"   Model is severely biased toward: {pred_counter.most_common(1)[0][0]}")
    
    print(f"\nAverage confidence: {sum(confidences)/len(confidences):.2%}")
    
    return predictions, confidences


if __name__ == "__main__":
    test_happy_audio_files()
