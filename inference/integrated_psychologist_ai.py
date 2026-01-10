"""
INTEGRATED PSYCHOLOGIST AI - ALL PHASES
========================================
Real-time multi-modal psychological state detection.

Architecture:
    Phase 1: Face emotion detection (dual model)
    Phase 2: Voice emotion + stress detection
    Phase 3: Multi-modal fusion & psychological reasoning
    Phase 4: Long-term cognitive layer (personality, baseline, deviations)

Output:
    - Real-time psychological state (Phase 3)
    - Personality traits (Phase 4)
    - Behavioral baseline (Phase 4)
    - Deviation alerts (Phase 4)
    - Personalized risk assessment (Phase 4)

Usage:
    python inference/integrated_psychologist_ai.py

Author: Psychologist AI Team
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import json
import sounddevice as sd
from collections import deque
import time
import threading
from queue import Queue

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.model import EmotionCNN
from training.preprocessing import preprocess_face
from training.voice.audio_preprocessing import normalize_audio
from training.voice.feature_extraction import extract_all_features, extract_stress_score
from training.voice.voice_emotion_model import VoiceEmotionSystem
from inference.phase3_multimodal_fusion import Phase3MultiModalFusion, format_psychological_state
from inference.phase4_cognitive_layer import Phase4CognitiveFusion


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Central configuration"""
    
    # Paths
    PROJECT_ROOT = Path(r"C:\Dwarka\Machiene Learning\Psycologist AI")
    FACE_MODEL_ROOT = PROJECT_ROOT / "models" / "face_emotion"
    VOICE_MODEL_ROOT = PROJECT_ROOT / "models" / "voice_emotion"
    
    # Face models
    FACE_MAIN_MODEL = FACE_MODEL_ROOT / "emotion_cnn_best.pth"
    FACE_SPECIALIST_MODEL = FACE_MODEL_ROOT / "emotion_cnn_phase15_specialist.pth"
    FACE_CONFIG = FACE_MODEL_ROOT / "config.json"
    
    # Voice models (auto-select best)
    VOICE_BALANCED = VOICE_MODEL_ROOT / "emotion_model_best_balanced.pth"
    VOICE_IMPROVED = VOICE_MODEL_ROOT / "emotion_model_best_improved.pth"
    VOICE_ORIGINAL = VOICE_MODEL_ROOT / "emotion_model_best.pth"
    STRESS_MODEL = VOICE_MODEL_ROOT / "stress_model_best.pth"
    
    if VOICE_BALANCED.exists():
        VOICE_EMOTION_MODEL = VOICE_BALANCED
    elif VOICE_IMPROVED.exists():
        VOICE_EMOTION_MODEL = VOICE_IMPROVED
    else:
        VOICE_EMOTION_MODEL = VOICE_ORIGINAL
    
    # Audio settings
    SAMPLE_RATE = 16000
    AUDIO_CHUNK_DURATION = 3.0  # seconds
    AUDIO_BUFFER_SIZE = int(SAMPLE_RATE * AUDIO_CHUNK_DURATION)
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Display settings
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720
    INFO_PANEL_WIDTH = 400


# ============================================
# PHASE 1: FACE EMOTION DETECTOR
# ============================================

class FaceEmotionDetector:
    """Phase 1: Dual-model face emotion detection"""
    
    def __init__(self, main_model_path, specialist_model_path, config_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.num_classes = config['num_classes']
        self.class_names = config['class_names']
        
        # Load main model
        self.main_model = EmotionCNN(num_classes=self.num_classes, input_size=48)
        self.main_model.load_state_dict(torch.load(main_model_path, map_location=self.device))
        self.main_model.to(self.device)
        self.main_model.eval()
        
        # Load specialist model
        self.specialist_model = EmotionCNN(num_classes=self.num_classes, input_size=48)
        self.specialist_model.load_state_dict(torch.load(specialist_model_path, map_location=self.device))
        self.specialist_model.to(self.device)
        self.specialist_model.eval()
        
        # Face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self.minority_classes = ['disgust', 'fear']
        
        print("✓ Phase 1 loaded: Face Emotion Detection (Dual Model)")
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def predict_emotion(self, face_img):
        """Predict emotion using dual-model strategy"""
        face_array = preprocess_face(face_img, target_size=(48, 48))
        face_tensor = torch.from_numpy(face_array).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Main model
            main_output = self.main_model(face_tensor)
            main_probs = torch.softmax(main_output, dim=1).cpu().numpy()[0]
            main_pred_idx = np.argmax(main_probs)
            main_emotion = self.class_names[main_pred_idx]
            main_confidence = main_probs[main_pred_idx]
            
            # Use specialist if needed
            if main_emotion in self.minority_classes or main_confidence < 0.6:
                specialist_output = self.specialist_model(face_tensor)
                specialist_probs = torch.softmax(specialist_output, dim=1).cpu().numpy()[0]
                specialist_pred_idx = np.argmax(specialist_probs)
                specialist_emotion = self.class_names[specialist_pred_idx]
                specialist_confidence = specialist_probs[specialist_pred_idx]
                
                if specialist_emotion in self.minority_classes and specialist_confidence > 0.5:
                    return specialist_emotion, specialist_confidence
            
            return main_emotion, main_confidence


# ============================================
# PHASE 2: VOICE EMOTION DETECTOR
# ============================================

class VoiceEmotionDetector:
    """Phase 2: Voice emotion + stress detection"""
    
    def __init__(self, emotion_model_path, stress_model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sample_rate = Config.SAMPLE_RATE
        
        # Load models
        self.system = VoiceEmotionSystem(feature_dim=48)
        
        if emotion_model_path.exists():
            self.system.emotion_model.load_state_dict(
                torch.load(emotion_model_path, map_location=self.device)
            )
        
        if stress_model_path.exists():
            self.system.stress_detector.load_state_dict(
                torch.load(stress_model_path, map_location=self.device)
            )
        
        self.system.to(self.device)
        self.system.eval()
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=Config.AUDIO_BUFFER_SIZE)
        
        print("✓ Phase 2 loaded: Voice Emotion + Stress Detection")
    
    def add_audio(self, audio_chunk):
        """Add audio chunk to buffer"""
        self.audio_buffer.extend(audio_chunk.flatten())
    
    def predict_emotion_and_stress(self):
        """Predict emotion and stress from buffer"""
        if len(self.audio_buffer) < Config.AUDIO_BUFFER_SIZE:
            return 'neutral', 0.5, 'low', 0.5, 0.5  # Default
        
        # Get audio
        audio = np.array(list(self.audio_buffer))
        audio = normalize_audio(audio)
        
        # Extract features
        try:
            _, features = extract_all_features(audio, self.sample_rate)  # Unpack tuple
            if features is None:
                return 'neutral', 0.5, 'low', 0.5, 0.5
            
            # Ensure features is a 1D numpy array (already flattened by extract_all_features)
            features = np.array(features, dtype=np.float32).flatten()
            
            # Check for NaN/inf
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return 'neutral', 0.5, 'low', 0.5, 0.5
            
            # Ensure correct shape (48 features)
            if features.shape[0] != 48:
                return 'neutral', 0.5, 'low', 0.5, 0.5
            
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Extract stress-specific features (4 values)
            stress_result = extract_stress_score(audio, self.sample_rate)
            stress_features = np.array([
                stress_result['jitter'],
                stress_result['shimmer'],
                stress_result['spectral_flatness'],
                stress_result['pitch_variance']
            ], dtype=np.float32)
            stress_features_tensor = torch.FloatTensor(stress_features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Emotion prediction
                emotion_output = self.system.emotion_model(features_tensor)
                emotion_probs = torch.softmax(emotion_output, dim=1).cpu().numpy()[0]
                emotion_idx = np.argmax(emotion_probs)
                emotion = self.system.emotion_model.class_names[emotion_idx]
                emotion_confidence = emotion_probs[emotion_idx]
                
                # Stress prediction (using 4 stress-specific features)
                stress_output = self.system.stress_detector(stress_features_tensor)
                stress_probs = torch.softmax(stress_output, dim=1).cpu().numpy()[0]
                stress_idx = np.argmax(stress_probs)
                stress_prob = stress_probs[stress_idx]
                
                # Map index to stress level: 0=low, 1=medium, 2=high
                stress_levels = ['low', 'medium', 'high']
                stress_level = stress_levels[stress_idx]
                
                # Audio quality (simple SNR estimate)
                audio_quality = min(1.0, np.std(audio) * 10)
                
                return emotion, float(emotion_confidence), stress_level, float(stress_prob), audio_quality
        
        except Exception as e:
            print(f"[Warning] Voice prediction error: {e}")
            return 'neutral', 0.5, 'low', 0.5, 0.5


# ============================================
# AUDIO CAPTURE THREAD
# ============================================

class AudioCaptureThread(threading.Thread):
    """Background thread for audio capture"""
    
    def __init__(self, voice_detector):
        super().__init__(daemon=True)
        self.voice_detector = voice_detector
        self.running = True
        self.stream = None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback"""
        if status:
            print(f"[Audio] {status}")
        
        audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
        self.voice_detector.add_audio(audio_chunk)
    
    def run(self):
        """Run audio capture"""
        try:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=Config.SAMPLE_RATE,
                blocksize=2048
            )
            self.stream.start()
            
            while self.running:
                time.sleep(0.1)
            
            self.stream.stop()
            self.stream.close()
        
        except Exception as e:
            print(f"[Error] Audio capture: {e}")
    
    def stop(self):
        """Stop capture"""
        self.running = False


# ============================================
# INTEGRATED SYSTEM
# ============================================

class IntegratedPsychologistAI:
    """Main integrated system"""
    
    def __init__(self):
        self.config = Config()
        
        print("=" * 70)
        print("INITIALIZING PSYCHOLOGIST AI - ALL PHASES")
        print("=" * 70)
        
        # Initialize Phase 1 (Face)
        self.face_detector = FaceEmotionDetector(
            main_model_path=self.config.FACE_MAIN_MODEL,
            specialist_model_path=self.config.FACE_SPECIALIST_MODEL,
            config_path=self.config.FACE_CONFIG,
            device=self.config.DEVICE
        )
        
        # Initialize Phase 2 (Voice)
        self.voice_detector = VoiceEmotionDetector(
            emotion_model_path=self.config.VOICE_EMOTION_MODEL,
            stress_model_path=self.config.STRESS_MODEL,
            device=self.config.DEVICE
        )
        
        # Initialize Phase 3 (Fusion)
        self.phase3 = Phase3MultiModalFusion()
        
        # Initialize Phase 4 (Cognitive Layer)
        self.phase4 = Phase4CognitiveFusion(
            user_id="default_user",
            storage_dir="data/user_memory"
        )
        
        print("✓ Phase 4 loaded: Cognitive Layer (Personality, Baseline, Deviations)")
        
        print(f"\nDevice: {self.config.DEVICE}")
        print("=" * 70)
        
        # Start audio capture thread
        self.audio_thread = AudioCaptureThread(self.voice_detector)
        self.audio_thread.start()
        
        # Latest state
        self.latest_state = None
        self.latest_phase4_profile = None
        self.frame_count = 0
        self.start_time = time.time()
    
    def process_frame(self, frame):
        """Process one video frame"""
        self.frame_count += 1
        
        # Phase 1: Detect face emotion
        faces = self.face_detector.detect_faces(frame)
        
        if len(faces) > 0:
            # Use first face
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            face_emotion, face_confidence = self.face_detector.predict_emotion(face_img)
            face_detected = True
            
            # Draw face box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            face_emotion = 'neutral'
            face_confidence = 0.0
            face_detected = False
        
        # Phase 2: Get voice emotion and stress
        voice_emotion, voice_confidence, stress_level, stress_confidence, audio_quality = \
            self.voice_detector.predict_emotion_and_stress()
        
        # Phase 3: Fuse and reason
        self.latest_state = self.phase3.process_frame(
            face_emotion=face_emotion,
            face_confidence=face_confidence,
            face_detected=face_detected,
            voice_emotion=voice_emotion,
            voice_confidence=voice_confidence,
            audio_quality=audio_quality,
            stress_level=stress_level,
            stress_confidence=stress_confidence
        )
        
        # Phase 4: Process through cognitive layer for personality & deviations
        self.latest_phase4_profile = self.phase4.process_state(self.latest_state)
        
        return frame
    
    def draw_info_panel(self, frame):
        """Draw information panel on frame"""
        if self.latest_state is None:
            return frame
        
        # Create larger info panel for Phase 4 data
        h, w = frame.shape[:2]
        panel_width = 450  # Increased from 400
        
        # Expand frame to add panel
        expanded = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
        expanded[:, :w] = frame
        expanded[:, w:] = (40, 40, 40)  # Dark background
        
        # Draw info
        state = self.latest_state
        phase4 = self.latest_phase4_profile
        x_offset = w + 10
        y_offset = 20
        line_height = 22
        
        def draw_text(text, y, color=(255, 255, 255), size=0.45, bold=False):
            thickness = 2 if bold else 1
            cv2.putText(expanded, text, (x_offset, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
        
        # ===== PHASE 3: CURRENT STATE =====
        draw_text("=== CURRENT STATE (PHASE 3) ===", y_offset, (0, 255, 255), 0.5, True)
        y_offset += line_height + 5
        
        # Dominant emotion
        emotion_color = self.get_emotion_color(state.dominant_emotion)
        draw_text(f"Emotion: {state.dominant_emotion.upper()}", y_offset, emotion_color, 0.5, True)
        y_offset += line_height
        
        # Hidden emotion
        if state.hidden_emotion:
            draw_text(f"Hidden: {state.hidden_emotion}", y_offset, (150, 150, 255), 0.45)
            y_offset += line_height
        
        # Mental state
        mental_state_text = state.mental_state.value.replace('_', ' ').title()
        draw_text(f"State: {mental_state_text}", y_offset, (255, 200, 100), 0.5)
        y_offset += line_height
        
        # Risk level (Phase 3)
        risk_color = self.get_risk_color(state.risk_level.value)
        draw_text(f"Risk (P3): {state.risk_level.value.upper()}", y_offset, risk_color, 0.5, True)
        y_offset += line_height
        
        # Confidence & Stability
        draw_text(f"Confidence: {state.confidence*100:.0f}% | Stability: {state.stability_score*100:.0f}%", 
                 y_offset, (200, 200, 200), 0.42)
        y_offset += line_height + 10
        
        # ===== PHASE 4: PERSONALIZED ANALYSIS =====
        if phase4:
            draw_text("=== PHASE 4: PERSONALITY ===", y_offset, (255, 150, 255), 0.5, True)
            y_offset += line_height + 5
            
            pers = phase4.personality
            if pers.confidence > 0.3:
                draw_text(f"Reactivity: {pers.emotional_reactivity:.2f}", y_offset, (200, 200, 200), 0.42)
                y_offset += line_height - 3
                draw_text(f"Tolerance: {pers.stress_tolerance:.2f}", y_offset, (200, 200, 200), 0.42)
                y_offset += line_height - 3
                draw_text(f"Stability: {pers.emotional_stability:.2f}", y_offset, (200, 200, 200), 0.42)
                y_offset += line_height - 3
                draw_text(f"Mood: {pers.baseline_mood}", y_offset, (200, 200, 200), 0.42)
                y_offset += line_height - 3
                draw_text(f"({pers.data_days}d, {pers.confidence:.0%} conf)", y_offset, (150, 150, 150), 0.38)
            else:
                draw_text("Building profile...", y_offset, (150, 150, 150), 0.42)
            
            y_offset += line_height + 8
            
            # Deviations
            draw_text("=== DEVIATIONS ===", y_offset, (255, 100, 100), 0.5, True)
            y_offset += line_height + 3
            
            if phase4.deviations:
                for dev in phase4.deviations[:3]:  # Top 3
                    dev_text = dev.deviation_type.replace('_', ' ')[:20]
                    sev_pct = f"{dev.severity*100:.0f}%"
                    color = (0, 0, 255) if dev.severity > 0.7 else (0, 165, 255) if dev.severity > 0.4 else (0, 255, 255)
                    draw_text(f"! {dev_text}: {sev_pct}", y_offset, color, 0.4)
                    y_offset += line_height - 4
            else:
                draw_text("No unusual behavior", y_offset, (100, 255, 100), 0.42)
                y_offset += line_height - 2
            
            y_offset += 8
            
            # Adjusted Risk
            draw_text("=== ADJUSTED RISK ===", y_offset, (100, 255, 255), 0.5, True)
            y_offset += line_height + 3
            
            adj_risk_color = self.get_risk_color(phase4.adjusted_risk.value)
            draw_text(f"Risk (P4): {phase4.adjusted_risk.value.upper()}", y_offset, adj_risk_color, 0.5, True)
            y_offset += line_height
            
            # Risk adjustment reason (wrap text)
            reason = phase4.risk_adjustment_reason
            if len(reason) > 35:
                # Split into two lines
                words = reason.split()
                line1 = ""
                line2 = ""
                for word in words:
                    if len(line1) < 35:
                        line1 += word + " "
                    else:
                        line2 += word + " "
                draw_text(line1.strip(), y_offset, (180, 180, 180), 0.38)
                y_offset += line_height - 5
                if line2:
                    draw_text(line2.strip(), y_offset, (180, 180, 180), 0.38)
                    y_offset += line_height - 5
            else:
                draw_text(reason, y_offset, (180, 180, 180), 0.4)
                y_offset += line_height
        
        # FPS at bottom
        y_offset = h - 40
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        draw_text(f"FPS: {fps:.1f} | Frames: {self.frame_count}", y_offset, (100, 200, 100), 0.45)
        
        return expanded
    
    @staticmethod
    def get_emotion_color(emotion):
        """Get color for emotion"""
        colors = {
            'happy': (0, 255, 0),
            'sad': (255, 100, 100),
            'angry': (0, 0, 255),
            'fear': (255, 0, 255),
            'neutral': (200, 200, 200),
            'disgust': (100, 255, 100),
            'surprise': (255, 255, 0)
        }
        return colors.get(emotion, (255, 255, 255))
    
    @staticmethod
    def get_risk_color(risk):
        """Get color for risk level"""
        colors = {
            'low': (0, 255, 0),
            'moderate': (0, 255, 255),
            'high': (0, 165, 255),
            'critical': (0, 0, 255)
        }
        return colors.get(risk, (255, 255, 255))
    
    def run(self):
        """Run integrated system"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[Error] Cannot open webcam")
            return
        
        print("\n" + "=" * 70)
        print("PSYCHOLOGIST AI - RUNNING")
        print("=" * 70)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'p' - Pause/Resume")
        print("  'i' - Toggle info panel")
        print("=" * 70 + "\n")
        
        paused = False
        show_info = True
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    frame = self.process_frame(frame)
                    
                    # Add info panel
                    if show_info:
                        frame = self.draw_info_panel(frame)
                
                # Display
                cv2.imshow('Psychologist AI - Integrated System', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f'psychologist_ai_{int(time.time())}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"[Saved] {filename}")
                elif key == ord('p'):
                    paused = not paused
                    print("[Paused]" if paused else "[Resumed]")
                elif key == ord('i'):
                    show_info = not show_info
                    print(f"[Info panel: {'ON' if show_info else 'OFF'}]")
        
        finally:
            # Cleanup
            self.audio_thread.stop()
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 70)
            print("SESSION SUMMARY")
            print("=" * 70)
            
            if self.latest_state:
                print(format_psychological_state(self.latest_state))
            
            stats = self.phase3.get_summary_statistics()
            print("\nTemporal Statistics:")
            print(f"  Face emotion switches: {stats['face_switches']}")
            print(f"  Voice emotion switches: {stats['voice_switches']}")
            print(f"  Stress persistent: {stats['stress_persistent']}")
            print(f"  Masking detected: {stats['masking_detected']}")
            print("=" * 70)


# ============================================
# MAIN
# ============================================

def main():
    """Main function"""
    try:
        system = IntegratedPsychologistAI()
        system.run()
    except KeyboardInterrupt:
        print("\n[Stopped by user]")
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
