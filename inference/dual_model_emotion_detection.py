"""
DUAL-MODEL EMOTION DETECTION
============================
Uses both Phase 1 (main) and Phase 1.5 (specialist) models for optimal performance.

Strategy:
  - Phase 1 (Main): General emotion recognition (62.57% accuracy)
  - Phase 1.5 (Specialist): Minority class expert (disgust +30%, fear +2%)
  
Usage:
  python inference/dual_model_emotion_detection.py
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.model import EmotionCNN
from training.preprocessing import preprocess_face


class DualModelEmotionDetector:
    """
    Emotion detector using dual models:
    - Phase 1 (main): General emotion recognition
    - Phase 1.5 (specialist): Minority class expert
    """
    
    def __init__(self, main_model_path, specialist_model_path, config_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.num_classes = config['num_classes']
        self.class_names = config['class_names']
        
        # Load Phase 1 (Main) model
        print("\nLoading Phase 1 (Main) model...")
        self.main_model = EmotionCNN(num_classes=self.num_classes, input_size=48)
        self.main_model.load_state_dict(torch.load(main_model_path, map_location=self.device))
        self.main_model.to(self.device)
        self.main_model.eval()
        print("✓ Phase 1 model loaded (general emotion recognition)")
        
        # Load Phase 1.5 (Specialist) model
        print("\nLoading Phase 1.5 (Specialist) model...")
        self.specialist_model = EmotionCNN(num_classes=self.num_classes, input_size=48)
        self.specialist_model.load_state_dict(torch.load(specialist_model_path, map_location=self.device))
        self.specialist_model.to(self.device)
        self.specialist_model.eval()
        print("✓ Phase 1.5 model loaded (minority class expert)")
        
        # Load face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Define minority classes (disgust, fear)
        self.minority_classes = ['disgust', 'fear']
        
        print("\n" + "=" * 60)
        print("DUAL-MODEL STRATEGY")
        print("=" * 60)
        print("Phase 1 (Main): Use for initial prediction")
        print("Phase 1.5 (Specialist): Use when disgust/fear detected")
        print("=" * 60 + "\n")
    
    def detect_faces(self, frame):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def predict_emotion(self, face_img):
        """
        Predict emotion using dual-model strategy:
        1. Get main model prediction
        2. If minority class detected, also check specialist model
        3. Return best prediction with confidence
        """
        # Preprocess face
        face_array = preprocess_face(face_img, target_size=(48, 48))
        face_tensor = torch.from_numpy(face_array).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Phase 1 (Main) prediction
            main_output = self.main_model(face_tensor)
            main_probs = torch.softmax(main_output, dim=1).cpu().numpy()[0]
            main_pred_idx = np.argmax(main_probs)
            main_pred_emotion = self.class_names[main_pred_idx]
            main_confidence = main_probs[main_pred_idx]
            
            # If minority class detected or low confidence, consult specialist
            if main_pred_emotion in self.minority_classes or main_confidence < 0.6:
                # Phase 1.5 (Specialist) prediction
                specialist_output = self.specialist_model(face_tensor)
                specialist_probs = torch.softmax(specialist_output, dim=1).cpu().numpy()[0]
                specialist_pred_idx = np.argmax(specialist_probs)
                specialist_pred_emotion = self.class_names[specialist_pred_idx]
                specialist_confidence = specialist_probs[specialist_pred_idx]
                
                # Decision logic: Use specialist if it predicts minority class with good confidence
                if specialist_pred_emotion in self.minority_classes and specialist_confidence > 0.5:
                    return specialist_pred_emotion, specialist_confidence, 'specialist', main_probs, specialist_probs
                else:
                    return main_pred_emotion, main_confidence, 'main', main_probs, specialist_probs
            else:
                return main_pred_emotion, main_confidence, 'main', main_probs, None
    
    def draw_results(self, frame, face, emotion, confidence, model_used):
        """Draw bounding box and emotion label"""
        x, y, w, h = face
        
        # Color based on model used
        if model_used == 'specialist':
            color = (0, 165, 255)  # Orange for specialist
            label = f"{emotion} ({confidence*100:.1f}%) [SPECIALIST]"
        else:
            color = (0, 255, 0)  # Green for main
            label = f"{emotion} ({confidence*100:.1f}%)"
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y-label_size[1]-10), (x+label_size[0], y), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run_webcam(self):
        """Run real-time emotion detection from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\n" + "=" * 60)
        print("WEBCAM EMOTION DETECTION (DUAL-MODEL)")
        print("=" * 60)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'p' - Pause/Resume")
        print("=" * 60 + "\n")
        
        paused = False
        frame_count = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process each face
                for face in faces:
                    x, y, w, h = face
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Predict emotion
                    emotion, confidence, model_used, main_probs, specialist_probs = self.predict_emotion(face_img)
                    
                    # Draw results
                    self.draw_results(frame, face, emotion, confidence, model_used)
                
                # Show FPS
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Dual-Model Emotion Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'emotion_screenshot_{frame_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam closed.")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dual-Model Emotion Detection')
    parser.add_argument('--main-model', type=str, 
                       default='models/face_emotion/emotion_cnn_best.pth',
                       help='Path to Phase 1 (main) model')
    parser.add_argument('--specialist-model', type=str,
                       default='models/face_emotion/emotion_cnn_phase15_specialist.pth',
                       help='Path to Phase 1.5 (specialist) model')
    parser.add_argument('--config', type=str,
                       default='models/face_emotion/config.json',
                       help='Path to model config')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create detector
    detector = DualModelEmotionDetector(
        main_model_path=args.main_model,
        specialist_model_path=args.specialist_model,
        config_path=args.config,
        device=args.device
    )
    
    # Run webcam detection
    detector.run_webcam()


if __name__ == "__main__":
    main()
