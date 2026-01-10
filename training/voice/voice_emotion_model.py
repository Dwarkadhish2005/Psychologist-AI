"""
Voice Emotion Model Architecture
=================================
Phase 2: Neural network for voice emotion recognition

Model Design:
  - Input: Feature vector (48 dimensions)
  - Output: 5 emotion classes (angry, fear, happy, neutral, sad)
  - Architecture: Fully connected network (simpler than Phase 1 CNN)

Philosophy:
  Voice features are already preprocessed (unlike raw images).
  We use FC layers instead of CNNs.

Author: Psychologist AI Team
Phase: 2 (Voice Emotion Recognition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceEmotionModel(nn.Module):
    """
    Voice emotion recognition model.
    
    Architecture:
      - Fully connected layers
      - Batch normalization
      - Dropout for regularization
      - 5-class output
    
    Input: Feature vector (48 dims)
    Output: Emotion logits (5 classes)
    """
    
    def __init__(self, input_dim=48, num_classes=5, hidden_dims=[256, 128, 64], dropout=0.3):
        """
        Initialize voice emotion model.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of emotion classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super(VoiceEmotionModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.class_names = ['angry', 'fear', 'happy', 'neutral', 'sad']
        
        # Build fully connected layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer
        self.classifier = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def predict_emotion(self, features):
        """
        Predict emotion from features with confidence.
        
        Args:
            features: Feature vector (input_dim,) or (batch, input_dim)
        
        Returns:
            emotion_idx: Predicted emotion index
            confidence: Prediction confidence
            probabilities: All class probabilities
        """
        self.eval()
        with torch.no_grad():
            # Handle single sample
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            # Forward pass
            logits = self.forward(features)
            probabilities = F.softmax(logits, dim=1)
            
            # Get prediction
            confidence, emotion_idx = torch.max(probabilities, dim=1)
            
            return emotion_idx.item(), confidence.item(), probabilities.squeeze().cpu().numpy()


class StressDetector(nn.Module):
    """
    Stress detection model (separate from emotion).
    
    Important: Stress ≠ Emotion
      - Can be happy + stressed
      - Can be neutral + stressed
    
    Input: Stress-specific features (4 dims: jitter, shimmer, flatness, pitch_var)
    Output: Stress level (3 classes: low, medium, high)
    """
    
    def __init__(self, input_dim=4, num_levels=3, hidden_dim=32):
        """
        Initialize stress detector.
        
        Args:
            input_dim: Stress feature dimension
            num_levels: Number of stress levels
            hidden_dim: Hidden layer dimension
        """
        super(StressDetector, self).__init__()
        
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.stress_levels = ['low', 'medium', 'high']
        
        # Simple 2-layer network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_levels)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Stress features (batch_size, input_dim)
        
        Returns:
            logits: Stress level logits (batch_size, num_levels)
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
    
    def predict_stress(self, stress_features):
        """
        Predict stress level from features.
        
        Args:
            stress_features: [jitter, shimmer, flatness, pitch_var]
        
        Returns:
            stress_level: 0=low, 1=medium, 2=high
            confidence: Prediction confidence
            probabilities: All level probabilities
        """
        self.eval()
        with torch.no_grad():
            # Handle single sample
            if stress_features.dim() == 1:
                stress_features = stress_features.unsqueeze(0)
            
            # Forward pass
            logits = self.forward(stress_features)
            probabilities = F.softmax(logits, dim=1)
            
            # Get prediction
            confidence, stress_idx = torch.max(probabilities, dim=1)
            
            return stress_idx.item(), confidence.item(), probabilities.squeeze().cpu().numpy()


class VoiceEmotionSystem(nn.Module):
    """
    Complete voice emotion system with stress detection.
    
    Outputs:
      1. Emotion (5 classes)
      2. Stress level (3 levels)
      3. Confidence scores
    
    This is what feeds into Phase 3 fusion.
    """
    
    def __init__(self, feature_dim=48, num_emotions=5, num_stress_levels=3):
        """
        Initialize complete voice emotion system.
        
        Args:
            feature_dim: Total feature dimension
            num_emotions: Number of emotion classes
            num_stress_levels: Number of stress levels
        """
        super(VoiceEmotionSystem, self).__init__()
        
        self.emotion_model = VoiceEmotionModel(
            input_dim=feature_dim,
            num_classes=num_emotions
        )
        
        self.stress_detector = StressDetector(
            input_dim=4,  # Stress-specific features
            num_levels=num_stress_levels
        )
        
        self.class_names = ['angry', 'fear', 'happy', 'neutral', 'sad']
        self.stress_levels = ['low', 'medium', 'high']
    
    def forward(self, features, stress_features):
        """
        Forward pass for both emotion and stress.
        
        Args:
            features: Full feature vector
            stress_features: Stress-specific features
        
        Returns:
            emotion_logits: Emotion predictions
            stress_logits: Stress predictions
        """
        emotion_logits = self.emotion_model(features)
        stress_logits = self.stress_detector(stress_features)
        return emotion_logits, stress_logits
    
    def predict(self, features, stress_features):
        """
        Complete prediction with emotion, stress, and confidence.
        
        Args:
            features: Full feature vector
            stress_features: [jitter, shimmer, flatness, pitch_var]
        
        Returns:
            result: Dictionary with all predictions
        """
        self.eval()
        with torch.no_grad():
            # Emotion prediction
            emotion_idx, emotion_conf, emotion_probs = self.emotion_model.predict_emotion(features)
            emotion_name = self.class_names[emotion_idx]
            
            # Stress prediction
            stress_idx, stress_conf, stress_probs = self.stress_detector.predict_stress(stress_features)
            stress_level = self.stress_levels[stress_idx]
            
            # Overall confidence (average)
            overall_confidence = (emotion_conf + stress_conf) / 2.0
            
            result = {
                'emotion': emotion_name,
                'emotion_confidence': emotion_conf,
                'emotion_probabilities': emotion_probs,
                'stress_level': stress_level,
                'stress_confidence': stress_conf,
                'stress_probabilities': stress_probs,
                'overall_confidence': overall_confidence,
                'emotion_idx': emotion_idx,
                'stress_idx': stress_idx
            }
            
            return result


def create_voice_emotion_model(feature_dim=48, num_classes=5, pretrained_path=None):
    """
    Create voice emotion model (factory function).
    
    Args:
        feature_dim: Feature vector dimension
        num_classes: Number of emotion classes
        pretrained_path: Path to pretrained model (optional)
    
    Returns:
        model: VoiceEmotionModel instance
    """
    model = VoiceEmotionModel(input_dim=feature_dim, num_classes=num_classes)
    
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))
        print(f"[Load] Loaded pretrained model from {pretrained_path}")
    
    return model


def create_voice_emotion_system(feature_dim=48, pretrained_emotion_path=None, 
                               pretrained_stress_path=None):
    """
    Create complete voice emotion system (factory function).
    
    Args:
        feature_dim: Feature vector dimension
        pretrained_emotion_path: Path to pretrained emotion model
        pretrained_stress_path: Path to pretrained stress model
    
    Returns:
        system: VoiceEmotionSystem instance
    """
    system = VoiceEmotionSystem(feature_dim=feature_dim)
    
    if pretrained_emotion_path:
        system.emotion_model.load_state_dict(torch.load(pretrained_emotion_path))
        print(f"[Load] Loaded emotion model from {pretrained_emotion_path}")
    
    if pretrained_stress_path:
        system.stress_detector.load_state_dict(torch.load(pretrained_stress_path))
        print(f"[Load] Loaded stress model from {pretrained_stress_path}")
    
    return system


if __name__ == "__main__":
    # Test model architecture
    print("=" * 60)
    print("VOICE EMOTION MODEL TEST")
    print("=" * 60)
    
    # Create model
    model = VoiceEmotionModel(input_dim=48, num_classes=5)
    print(f"\n[Model] Voice Emotion Model")
    print(f"  Input dim: 48")
    print(f"  Output classes: 5")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    test_features = torch.randn(batch_size, 48)
    output = model(test_features)
    print(f"\n[Test] Forward pass")
    print(f"  Input shape: {test_features.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test prediction
    single_feature = torch.randn(48)
    emotion_idx, confidence, probs = model.predict_emotion(single_feature)
    print(f"\n[Test] Single prediction")
    print(f"  Emotion index: {emotion_idx}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Probabilities: {probs}")
    
    # Test stress detector
    stress_model = StressDetector(input_dim=4, num_levels=3)
    print(f"\n[Model] Stress Detector")
    print(f"  Input dim: 4")
    print(f"  Output levels: 3")
    print(f"  Parameters: {sum(p.numel() for p in stress_model.parameters()):,}")
    
    # Test complete system
    system = VoiceEmotionSystem(feature_dim=48)
    test_stress_features = torch.randn(4)  # [jitter, shimmer, flatness, pitch_var]
    result = system.predict(test_features[0], test_stress_features)
    
    print(f"\n[Test] Complete system prediction")
    print(f"  Emotion: {result['emotion']}")
    print(f"  Emotion confidence: {result['emotion_confidence']:.3f}")
    print(f"  Stress level: {result['stress_level']}")
    print(f"  Stress confidence: {result['stress_confidence']:.3f}")
    print(f"  Overall confidence: {result['overall_confidence']:.3f}")
    
    print("\n" + "=" * 60)
    print("✓ Voice emotion model ready!")
    print(f"✓ Total parameters: {sum(p.numel() for p in system.parameters()):,}")
    print("=" * 60)
