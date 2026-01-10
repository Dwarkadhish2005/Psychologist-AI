"""
Real-Time Voice Emotion Detection from Microphone
==================================================
Phase 2: Live microphone → emotion + stress + confidence

Usage:
    python microphone_emotion_detection.py

Author: Psychologist AI Team
Phase: 2 (Voice Emotion Recognition)
"""

import sys
from pathlib import Path
import numpy as np
import torch
import sounddevice as sd
from collections import deque
import time
import os

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from training.voice.audio_preprocessing import preprocess_realtime_chunk, normalize_audio
from training.voice.feature_extraction import extract_all_features, extract_stress_score
from training.voice.voice_emotion_model import VoiceEmotionSystem


# ============================================
# CONFIGURATION
# ============================================

class Config:
    # Model paths
    MODEL_ROOT = Path(r"C:\Dwarka\Machiene Learning\Psycologist AI\models\voice_emotion")
    
    # Try balanced model first, then improved, then original
    BALANCED_MODEL = MODEL_ROOT / "emotion_model_best_balanced.pth"
    IMPROVED_MODEL = MODEL_ROOT / "emotion_model_best_improved.pth"
    ORIGINAL_MODEL = MODEL_ROOT / "emotion_model_best.pth"
    
    # Select best available model
    if BALANCED_MODEL.exists():
        EMOTION_MODEL_PATH = BALANCED_MODEL
        MODEL_VERSION = "BALANCED"
    elif IMPROVED_MODEL.exists():
        EMOTION_MODEL_PATH = IMPROVED_MODEL
        MODEL_VERSION = "IMPROVED"
    else:
        EMOTION_MODEL_PATH = ORIGINAL_MODEL
        MODEL_VERSION = "ORIGINAL"
    
    STRESS_MODEL_PATH = MODEL_ROOT / "stress_model_best.pth"
    
    # Audio parameters
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 3.0  # seconds
    HOP_DURATION = 1.5     # seconds (overlap)
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
    HOP_SAMPLES = int(SAMPLE_RATE * HOP_DURATION)
    
    # Feature parameters
    FEATURE_DIM = 48
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Display
    HISTORY_LENGTH = 10  # Keep last N predictions


# ============================================
# REAL-TIME EMOTION DETECTOR
# ============================================

class RealtimeVoiceEmotionDetector:
    """
    Real-time voice emotion detection from microphone.
    """
    
    def __init__(self, emotion_model_path=None, stress_model_path=None):
        """
        Initialize real-time detector.
        
        Args:
            emotion_model_path: Path to emotion model
            stress_model_path: Path to stress model
        """
        self.config = Config()
        
        # Load models
        print(f"[Init] Loading models (using {self.config.MODEL_VERSION} version)...")
        self.system = VoiceEmotionSystem(feature_dim=self.config.FEATURE_DIM)
        
        if emotion_model_path and Path(emotion_model_path).exists():
            self.system.emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=self.config.DEVICE))
            print(f"[Init] ✓ Loaded custom emotion model from {emotion_model_path}")
        elif self.config.EMOTION_MODEL_PATH.exists():
            self.system.emotion_model.load_state_dict(torch.load(self.config.EMOTION_MODEL_PATH, map_location=self.config.DEVICE))
            print(f"[Init] ✓ Loaded emotion model: {self.config.EMOTION_MODEL_PATH.name}")
        else:
            print(f"[Warning] Emotion model not found, using random weights")
        
        if stress_model_path and Path(stress_model_path).exists():
            self.system.stress_detector.load_state_dict(torch.load(stress_model_path, map_location=self.config.DEVICE))
            print(f"[Init] ✓ Loaded stress model from {stress_model_path}")
        else:
            print(f"[Warning] Stress model not found, using random weights")
        
        self.system.to(self.config.DEVICE)
        self.system.eval()
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.config.CHUNK_SAMPLES)
        
        # Prediction history
        self.emotion_history = deque(maxlen=self.config.HISTORY_LENGTH)
        self.stress_history = deque(maxlen=self.config.HISTORY_LENGTH)
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"[Init] Device: {self.config.DEVICE}")
        print(f"[Init] Sample rate: {self.config.SAMPLE_RATE}Hz")
        print(f"[Init] Chunk duration: {self.config.CHUNK_DURATION}s")
        print(f"[Init] Ready!")
    
    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback for audio input stream.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Status flags
        """
        if status:
            print(f"[Warning] Audio callback status: {status}")
        
        # Add to buffer (mono)
        audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
        self.audio_buffer.extend(audio_chunk)
    
    def process_audio_chunk(self):
        """
        Process current audio buffer and return prediction.
        
        Returns:
            result: Dictionary with emotion, stress, and confidence
        """
        if len(self.audio_buffer) < self.config.CHUNK_SAMPLES:
            return None
        
        # Get audio chunk
        audio_chunk = np.array(list(self.audio_buffer))
        
        # Quick preprocessing
        processed = preprocess_realtime_chunk(audio_chunk, sr=self.config.SAMPLE_RATE)
        
        if processed is None:
            return None  # Too quiet or empty
        
        try:
            # Extract features
            _, feature_vector = extract_all_features(processed, self.config.SAMPLE_RATE)
            
            # Extract stress score
            stress_result = extract_stress_score(processed, self.config.SAMPLE_RATE)
            stress_features = np.array([
                stress_result['jitter'],
                stress_result['shimmer'],
                stress_result['spectral_flatness'],
                stress_result['pitch_variance']
            ])
            
            # Convert to tensors
            features = torch.FloatTensor(feature_vector).to(self.config.DEVICE)
            stress_features_tensor = torch.FloatTensor(stress_features).to(self.config.DEVICE)
            
            # Predict
            result = self.system.predict(features, stress_features_tensor)
            
            # Add to history
            self.emotion_history.append(result['emotion'])
            self.stress_history.append(result['stress_level'])
            
            # Frame statistics
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            result['fps'] = fps
            result['frame_count'] = self.frame_count
            
            return result
            
        except Exception as e:
            print(f"[Error] Processing failed: {e}")
            return None
    
    def get_smoothed_prediction(self):
        """
        Get smoothed prediction from history.
        
        Returns:
            emotion: Most common emotion in history
            stress: Most common stress level in history
        """
        if len(self.emotion_history) == 0:
            return None, None
        
        # Most common emotion
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Most common stress
        stress_counts = {}
        for stress in self.stress_history:
            stress_counts[stress] = stress_counts.get(stress, 0) + 1
        stress = max(stress_counts, key=stress_counts.get)
        
        return emotion, stress
    
    def display_result(self, result):
        """Display prediction result."""
        if result is None:
            return
        
        # Clear screen (optional, might be annoying)
        # print("\033[H\033[J", end="")
        
        print(f"\r[Frame {result['frame_count']:4d}] "
              f"Emotion: {result['emotion']:8s} ({result['emotion_confidence']:.2f}) | "
              f"Stress: {result['stress_level']:6s} ({result['stress_confidence']:.2f}) | "
              f"FPS: {result['fps']:.1f}", end="")
    
    def run(self):
        """
        Run real-time detection from microphone.
        """
        print("\n" + "="*60)
        print("REAL-TIME VOICE EMOTION DETECTION")
        print("="*60)
        print("Controls:")
        print("  Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        try:
            # Open audio stream
            with sd.InputStream(
                samplerate=self.config.SAMPLE_RATE,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.config.HOP_SAMPLES
            ):
                print("[Audio] Listening from microphone...")
                print("[Audio] Speak into your microphone!\n")
                
                while True:
                    # Wait for buffer to fill
                    time.sleep(self.config.HOP_DURATION)
                    
                    # Process audio
                    result = self.process_audio_chunk()
                    
                    # Display
                    if result:
                        self.display_result(result)
        
        except KeyboardInterrupt:
            print("\n\n[Stop] Stopping detection...")
        
        except Exception as e:
            print(f"\n[Error] {e}")
        
        finally:
            # Summary
            print(f"\n\n{'='*60}")
            print("SESSION SUMMARY")
            print(f"{'='*60}")
            print(f"Total frames: {self.frame_count}")
            print(f"Duration: {time.time() - self.start_time:.1f}s")
            
            if len(self.emotion_history) > 0:
                print(f"\nEmotion distribution:")
                emotion_counts = {}
                for emotion in self.emotion_history:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                    pct = 100 * count / len(self.emotion_history)
                    print(f"  {emotion:8s}: {count:3d} ({pct:5.1f}%)")
                
                print(f"\nStress distribution:")
                stress_counts = {}
                for stress in self.stress_history:
                    stress_counts[stress] = stress_counts.get(stress, 0) + 1
                for stress, count in sorted(stress_counts.items(), key=lambda x: x[1], reverse=True):
                    pct = 100 * count / len(self.stress_history)
                    print(f"  {stress:6s}: {count:3d} ({pct:5.1f}%)")
            
            print(f"{'='*60}\n")


# ============================================
# MAIN
# ============================================

def main():
    config = Config()
    
    # Create detector
    detector = RealtimeVoiceEmotionDetector(
        emotion_model_path=config.EMOTION_MODEL_PATH,
        stress_model_path=config.STRESS_MODEL_PATH
    )
    
    # Run detection
    detector.run()


if __name__ == "__main__":
    main()
