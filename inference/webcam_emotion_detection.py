"""
Real-Time Emotion Detection from Webcam
========================================
Detect faces and predict emotions in real-time using trained EmotionCNN.

Usage:
    python webcam_emotion_detection.py

Controls:
    'q' - Quit
    's' - Save screenshot
    'p' - Pause/Resume

Requirements:
    - Trained model in models/face_emotion/emotion_cnn_best.pth
    - Webcam connected
    - OpenCV installed

Author: Psychologist AI Team
Phase: 1 (Facial Emotion Recognition)
"""

import torch
import cv2
import numpy as np
import json
import os
from datetime import datetime
import sys

# Add training directory to path
sys.path.append('training')
from model import EmotionCNN
from preprocessing import preprocess_face


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Inference configuration"""
    
    # Model paths
    MODEL_PATH = 'models/face_emotion/emotion_cnn_best.pth'
    LABELS_PATH = 'models/face_emotion/labels.json'
    CONFIG_PATH = 'models/face_emotion/config.json'
    
    # Model params
    INPUT_SIZE = 48
    NUM_CLASSES = 7  # Default, will be overridden from config file
    
    # Face detection
    FACE_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    SCALE_FACTOR = 1.3
    MIN_NEIGHBORS = 5
    MIN_FACE_SIZE = (30, 30)
    
    # Display
    WINDOW_NAME = 'Emotion Detection - Press Q to Quit'
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.9
    FONT_THICKNESS = 2
    BOX_COLOR = (0, 255, 0)  # Green
    TEXT_COLOR = (0, 255, 0)
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================
# EMOTION DETECTOR CLASS
# ============================================

class EmotionDetector:
    """
    Real-time emotion detector with face detection and CNN inference.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Load config file first to get num_classes
        if os.path.exists(config.CONFIG_PATH):
            print("Loading model config...")
            with open(config.CONFIG_PATH, 'r') as f:
                model_config = json.load(f)
            self.config.NUM_CLASSES = model_config.get('num_classes', 7)
            print(f"  Detected {self.config.NUM_CLASSES} classes")
        
        # Load model
        print("Loading model...")
        self.model = self._load_model()
        
        # Load labels
        print("Loading labels...")
        self.labels = self._load_labels()
        
        # Initialize face detector
        print("Loading face detector...")
        self.face_cascade = cv2.CascadeClassifier(config.FACE_CASCADE)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load face cascade classifier!")
        
        print("✓ Emotion detector ready!")
    
    def _load_model(self):
        """Load trained model"""
        if not os.path.exists(self.config.MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {self.config.MODEL_PATH}\n"
                "Please train the model first using train_emotion_model.py"
            )
        
        model = EmotionCNN(
            num_classes=self.config.NUM_CLASSES,
            input_size=self.config.INPUT_SIZE
        )
        model.load_state_dict(
            torch.load(self.config.MODEL_PATH, map_location=self.config.DEVICE)
        )
        model.to(self.config.DEVICE)
        model.eval()
        
        return model
    
    def _load_labels(self):
        """Load label mappings"""
        if not os.path.exists(self.config.LABELS_PATH):
            # Use default labels for 7 classes
            return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        
        with open(self.config.LABELS_PATH, 'r') as f:
            labels_dict = json.load(f)
        
        # Convert string keys to int
        return {int(k): v for k, v in labels_dict.items()}
    
    def detect_faces(self, frame):
        """
        Detect faces in frame using Haar Cascade.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.SCALE_FACTOR,
            minNeighbors=self.config.MIN_NEIGHBORS,
            minSize=self.config.MIN_FACE_SIZE
        )
        return faces
    
    def predict_emotion(self, face_roi):
        """
        Predict emotion from face ROI.
        
        Args:
            face_roi: Cropped face image (grayscale or BGR)
        
        Returns:
            (emotion_label, confidence)
        """
        # Preprocess
        preprocessed = preprocess_face(face_roi, target_size=(self.config.INPUT_SIZE, self.config.INPUT_SIZE))
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(preprocessed).unsqueeze(0)  # (1, 1, 48, 48)
        input_tensor = input_tensor.to(self.config.DEVICE)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get emotion label
        emotion_idx = predicted.item()
        emotion_label = self.labels.get(emotion_idx, 'unknown')
        confidence_score = confidence.item()
        
        return emotion_label, confidence_score
    
    def draw_results(self, frame, faces, emotions):
        """
        Draw bounding boxes and emotion labels on frame.
        
        Args:
            frame: Input frame
            faces: List of (x, y, w, h) bounding boxes
            emotions: List of (emotion, confidence) tuples
        
        Returns:
            Annotated frame
        """
        for (x, y, w, h), (emotion, confidence) in zip(faces, emotions):
            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                self.config.BOX_COLOR,
                2
            )
            
            # Prepare text
            text = f"{emotion} ({confidence*100:.1f}%)"
            
            # Draw text background
            (text_w, text_h), _ = cv2.getTextSize(
                text,
                self.config.FONT,
                self.config.FONT_SCALE,
                self.config.FONT_THICKNESS
            )
            cv2.rectangle(
                frame,
                (x, y - text_h - 10),
                (x + text_w, y),
                self.config.BOX_COLOR,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                text,
                (x, y - 5),
                self.config.FONT,
                self.config.FONT_SCALE,
                (0, 0, 0),  # Black text
                self.config.FONT_THICKNESS
            )
        
        return frame


# ============================================
# MAIN WEBCAM LOOP
# ============================================

def main():
    """Main webcam emotion detection loop"""
    
    config = Config()
    
    print("=" * 60)
    print("REAL-TIME EMOTION DETECTION")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'p' - Pause/Resume")
    print("=" * 60)
    
    # Initialize detector
    detector = EmotionDetector(config)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" Error: Could not open webcam!")
        return
    
    print("\n✓ Webcam opened successfully")
    print("Starting emotion detection...\n")
    
    # FPS calculation
    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    fps = 0
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print(" Error: Failed to read frame")
                break
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Predict emotions for each face
            emotions = []
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                # Predict emotion
                emotion, confidence = detector.predict_emotion(face_roi)
                emotions.append((emotion, confidence))
            
            # Draw results
            annotated_frame = detector.draw_results(frame, faces, emotions)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps_end_time = cv2.getTickCount()
                time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
                fps = fps_counter / time_diff
                fps_counter = 0
                fps_start_time = cv2.getTickCount()
            
            # Display FPS
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                config.FONT,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Display face count
            cv2.putText(
                annotated_frame,
                f"Faces: {len(faces)}",
                (10, 60),
                config.FONT,
                0.7,
                (255, 255, 255),
                2
            )
        else:
            # Paused - use last frame
            cv2.putText(
                annotated_frame,
                "PAUSED",
                (frame.shape[1]//2 - 80, frame.shape[0]//2),
                config.FONT,
                1.5,
                (0, 0, 255),
                3
            )
        
        # Display frame
        cv2.imshow(config.WINDOW_NAME, annotated_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"assets/reports/screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f" Screenshot saved: {filename}")
        elif key == ord('p'):
            # Pause/Resume
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n Emotion detection stopped")


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
