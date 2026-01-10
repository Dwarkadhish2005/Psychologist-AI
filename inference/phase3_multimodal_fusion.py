"""
PHASE 3: MULTI-MODAL FUSION & PSYCHOLOGICAL REASONING
=====================================================
Combines face emotion + voice emotion + stress detection into psychological insights.

Architecture:
    Layer 1: Signal Normalization (reliability-weighted)
    Layer 2: Temporal Reasoning (memory & patterns)
    Layer 3: Fusion Logic (rule-based psychology)
    Layer 4: Psychological State Inference
    
Output:
    - Dominant emotion
    - Hidden emotion
    - Mental state (e.g., "happy_under_stress")
    - Confidence + explanation
    - Risk level
    - Stability score

Author: Psychologist AI Team
Phase: 3 (Multi-Modal Fusion)
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import time


# ============================================
# ENUMS & DATA STRUCTURES
# ============================================

class Modality(Enum):
    """Signal modalities with reliability scores"""
    FACE = ("face", 0.5)      # Medium reliability (people mask emotions)
    VOICE = ("voice", 0.7)    # High reliability (harder to fake)
    STRESS = ("stress", 0.9)  # Very high reliability (physiological)
    
    def __init__(self, name, reliability):
        self._name = name
        self.reliability = reliability


class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class MentalState(Enum):
    """Psychological state taxonomy"""
    CALM = "calm"
    STRESSED = "stressed"
    HAPPY_UNDER_STRESS = "happy_under_stress"
    EMOTIONALLY_MASKED = "emotionally_masked"
    ANXIOUS = "anxious"
    OVERWHELMED = "overwhelmed"
    EMOTIONALLY_FLAT = "emotionally_flat"
    JOYFUL = "joyful"
    FEARFUL = "fearful"
    ANGRY_STRESSED = "angry_stressed"
    SAD_DEPRESSED = "sad_depressed"
    CONFUSED = "confused"
    STABLE_NEGATIVE = "stable_negative"
    STABLE_POSITIVE = "stable_positive"
    EMOTIONALLY_UNSTABLE = "emotionally_unstable"


@dataclass
class NormalizedSignal:
    """Normalized signal from a modality"""
    modality: Modality
    emotion: str
    confidence: float
    reliability: float  # Modality base reliability
    signal_quality: float  # Current signal quality (0-1)
    timestamp: float
    
    @property
    def weighted_confidence(self) -> float:
        """Confidence weighted by reliability and quality"""
        return self.confidence * self.reliability * self.signal_quality


@dataclass
class TemporalPattern:
    """Detected temporal pattern"""
    pattern_type: str  # e.g., "emotion_switch", "stress_persistence"
    duration: float
    intensity: float
    description: str


@dataclass
class PsychologicalState:
    """Final psychological reasoning output"""
    dominant_emotion: str
    hidden_emotion: Optional[str]
    mental_state: MentalState
    confidence: float
    explanations: List[str]
    risk_level: RiskLevel
    stability_score: float  # 0-1 (0=unstable, 1=stable)
    temporal_patterns: List[TemporalPattern]
    raw_signals: Dict[str, NormalizedSignal]
    timestamp: float


# ============================================
# LAYER 1: SIGNAL NORMALIZATION
# ============================================

class SignalNormalizer:
    """
    Normalizes raw predictions into comparable signals.
    """
    
    # Emotion mappings between modalities
    FACE_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    VOICE_EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad']
    
    # Map face emotions to voice emotions (for comparison)
    EMOTION_MAPPING = {
        'disgust': 'angry',      # Disgust often manifests as anger
        'surprise': 'neutral',   # Surprise is transient, treat as neutral
    }
    
    def __init__(self):
        self.quality_threshold = 0.3  # Minimum confidence to consider signal valid
    
    def normalize_face_signal(self, emotion: str, confidence: float, 
                              face_detected: bool = True) -> NormalizedSignal:
        """Normalize face emotion signal"""
        
        # Signal quality based on detection and confidence
        if not face_detected:
            signal_quality = 0.0
        elif confidence < self.quality_threshold:
            signal_quality = 0.3
        else:
            signal_quality = min(1.0, confidence * 1.5)  # Boost good predictions
        
        # Map to common emotion space
        normalized_emotion = self.EMOTION_MAPPING.get(emotion, emotion)
        
        return NormalizedSignal(
            modality=Modality.FACE,
            emotion=normalized_emotion,
            confidence=confidence,
            reliability=Modality.FACE.reliability,
            signal_quality=signal_quality,
            timestamp=time.time()
        )
    
    def normalize_voice_signal(self, emotion: str, confidence: float,
                               audio_quality: float = 1.0) -> NormalizedSignal:
        """Normalize voice emotion signal"""
        
        # Signal quality based on audio quality and confidence
        if confidence < self.quality_threshold:
            signal_quality = 0.3
        else:
            signal_quality = min(1.0, confidence * audio_quality)
        
        return NormalizedSignal(
            modality=Modality.VOICE,
            emotion=emotion,
            confidence=confidence,
            reliability=Modality.VOICE.reliability,
            signal_quality=signal_quality,
            timestamp=time.time()
        )
    
    def normalize_stress_signal(self, stress_level: str, confidence: float) -> NormalizedSignal:
        """Normalize stress signal (special modality)"""
        
        # Stress is highly reliable, even at lower confidence
        signal_quality = max(0.5, confidence)  # Minimum 0.5 quality
        
        return NormalizedSignal(
            modality=Modality.STRESS,
            emotion=stress_level,  # 'low', 'medium', 'high'
            confidence=confidence,
            reliability=Modality.STRESS.reliability,
            signal_quality=signal_quality,
            timestamp=time.time()
        )


# ============================================
# LAYER 2: TEMPORAL REASONING
# ============================================

class TemporalMemory:
    """
    Maintains temporal history and detects patterns.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of frames to remember (at ~2 fps = 15 seconds)
        """
        self.window_size = window_size
        
        # Rolling windows
        self.face_history = deque(maxlen=window_size)
        self.voice_history = deque(maxlen=window_size)
        self.stress_history = deque(maxlen=window_size)
        
        # Pattern detection thresholds
        self.switch_threshold = 3  # Rapid switches = unstable
        self.persistence_threshold = 10  # 10 frames = pattern
    
    def add_signals(self, face: NormalizedSignal, voice: NormalizedSignal, 
                   stress: NormalizedSignal):
        """Add new signals to memory"""
        self.face_history.append(face)
        self.voice_history.append(voice)
        self.stress_history.append(stress)
    
    def get_emotion_frequency(self, modality: str) -> Dict[str, float]:
        """Get emotion frequency distribution for a modality"""
        if modality == 'face':
            history = self.face_history
        elif modality == 'voice':
            history = self.voice_history
        else:
            return {}
        
        if not history:
            return {}
        
        emotions = [signal.emotion for signal in history]
        unique, counts = np.unique(emotions, return_counts=True)
        total = len(emotions)
        
        return {emotion: count / total for emotion, count in zip(unique, counts)}
    
    def count_emotion_switches(self, modality: str, window: int = 10) -> int:
        """Count emotion switches in recent window (instability metric)"""
        if modality == 'face':
            history = list(self.face_history)[-window:]
        elif modality == 'voice':
            history = list(self.voice_history)[-window:]
        else:
            return 0
        
        if len(history) < 2:
            return 0
        
        switches = 0
        for i in range(1, len(history)):
            if history[i].emotion != history[i-1].emotion:
                switches += 1
        
        return switches
    
    def detect_stress_persistence(self) -> Tuple[bool, float]:
        """
        Detect if stress has been persistently high.
        Returns: (is_persistent, duration_ratio)
        """
        if len(self.stress_history) < self.persistence_threshold:
            return False, 0.0
        
        recent_stress = list(self.stress_history)[-self.persistence_threshold:]
        high_stress_count = sum(1 for s in recent_stress if s.emotion == 'high')
        
        duration_ratio = high_stress_count / len(recent_stress)
        is_persistent = duration_ratio > 0.7  # 70% of time
        
        return is_persistent, duration_ratio
    
    def detect_masking(self) -> Tuple[bool, str]:
        """
        Detect emotional masking: face neutral but voice emotional.
        Returns: (is_masked, description)
        """
        if len(self.face_history) < 5 or len(self.voice_history) < 5:
            return False, ""
        
        # Recent history
        recent_face = list(self.face_history)[-5:]
        recent_voice = list(self.voice_history)[-5:]
        
        # Check: face mostly neutral, voice mostly emotional
        face_neutral_ratio = sum(1 for f in recent_face if f.emotion == 'neutral') / len(recent_face)
        voice_emotions = [v.emotion for v in recent_voice]
        voice_emotional_ratio = sum(1 for e in voice_emotions if e != 'neutral') / len(voice_emotions)
        
        if face_neutral_ratio > 0.6 and voice_emotional_ratio > 0.6:
            dominant_voice_emotion = max(set(voice_emotions), key=voice_emotions.count)
            return True, f"Face neutral, voice shows {dominant_voice_emotion}"
        
        return False, ""
    
    def detect_patterns(self) -> List[TemporalPattern]:
        """Detect all temporal patterns"""
        patterns = []
        
        # Pattern 1: Stress persistence
        stress_persistent, stress_duration = self.detect_stress_persistence()
        if stress_persistent:
            patterns.append(TemporalPattern(
                pattern_type="stress_persistence",
                duration=stress_duration * self.persistence_threshold / 2,  # Convert to seconds (approx)
                intensity=stress_duration,
                description=f"High stress persisting ({stress_duration*100:.0f}% of time)"
            ))
        
        # Pattern 2: Emotional masking
        is_masked, mask_desc = self.detect_masking()
        if is_masked:
            patterns.append(TemporalPattern(
                pattern_type="emotional_masking",
                duration=2.5,  # Recent 5 frames
                intensity=0.8,
                description=mask_desc
            ))
        
        # Pattern 3: Emotional instability
        face_switches = self.count_emotion_switches('face', window=10)
        voice_switches = self.count_emotion_switches('voice', window=10)
        
        if face_switches >= self.switch_threshold or voice_switches >= self.switch_threshold:
            patterns.append(TemporalPattern(
                pattern_type="emotional_instability",
                duration=5.0,  # Recent 10 frames
                intensity=min(1.0, (face_switches + voice_switches) / 10),
                description=f"Rapid emotion switches (face: {face_switches}, voice: {voice_switches})"
            ))
        
        return patterns


# ============================================
# LAYER 3: FUSION LOGIC (CORE BRAIN)
# ============================================

class FusionEngine:
    """
    Rule-based fusion of face, voice, and stress signals.
    Psychology-inspired decision making.
    """
    
    def __init__(self):
        self.normalizer = SignalNormalizer()
        self.memory = TemporalMemory(window_size=30)
    
    def fuse_signals(self, face: NormalizedSignal, voice: NormalizedSignal,
                    stress: NormalizedSignal) -> Tuple[str, Optional[str], List[str]]:
        """
        Core fusion logic.
        Returns: (dominant_emotion, hidden_emotion, explanations)
        """
        explanations = []
        
        # Rule 1: Stress dominance
        if stress.emotion == 'high' and stress.weighted_confidence > 0.5:
            explanations.append(f"High stress detected ({stress.confidence*100:.0f}% confidence)")
            
            # If face/voice show happiness, it's masked
            if voice.emotion == 'happy' or face.emotion == 'happy':
                explanations.append("Happiness detected but stress is high → masked emotion")
                return voice.emotion if voice.emotion == 'happy' else face.emotion, 'stress', explanations
            
            # Otherwise stress is dominant
            dominant = voice.emotion if voice.weighted_confidence > face.weighted_confidence else face.emotion
            return dominant, 'stress', explanations
        
        # Rule 2: Voice > Face (voice is more reliable)
        if abs(voice.weighted_confidence - face.weighted_confidence) > 0.2:
            if voice.weighted_confidence > face.weighted_confidence:
                explanations.append(f"Voice signal stronger ({voice.confidence*100:.0f}% vs face {face.confidence*100:.0f}%)")
                
                # Check for masking
                if voice.emotion != face.emotion and voice.emotion != 'neutral':
                    explanations.append(f"Face shows {face.emotion}, voice shows {voice.emotion} → trusting voice")
                    return voice.emotion, face.emotion, explanations
                
                return voice.emotion, None, explanations
            else:
                explanations.append(f"Face signal stronger ({face.confidence*100:.0f}% vs voice {voice.confidence*100:.0f}%)")
                return face.emotion, None, explanations
        
        # Rule 3: Agreement - both modalities agree
        if voice.emotion == face.emotion:
            explanations.append(f"Face and voice agree: {voice.emotion}")
            return voice.emotion, None, explanations
        
        # Rule 4: Conflict resolution - weighted average
        if voice.weighted_confidence > face.weighted_confidence:
            explanations.append(f"Conflict: voice ({voice.emotion}) vs face ({face.emotion}), trusting voice")
            return voice.emotion, face.emotion, explanations
        else:
            explanations.append(f"Conflict: face ({face.emotion}) vs voice ({voice.emotion}), trusting face")
            return face.emotion, voice.emotion, explanations


# ============================================
# LAYER 4: PSYCHOLOGICAL REASONING
# ============================================

class PsychologicalReasoner:
    """
    Infer mental states from fused signals and temporal patterns.
    """
    
    def infer_mental_state(self, dominant_emotion: str, hidden_emotion: Optional[str],
                          stress_level: str, patterns: List[TemporalPattern],
                          face_signal: NormalizedSignal, voice_signal: NormalizedSignal) -> MentalState:
        """
        Infer psychological state from all available information.
        """
        
        # Check for specific patterns first
        pattern_types = [p.pattern_type for p in patterns]
        
        # Emotional masking detected
        if 'emotional_masking' in pattern_types:
            return MentalState.EMOTIONALLY_MASKED
        
        # Emotional instability
        if 'emotional_instability' in pattern_types:
            return MentalState.EMOTIONALLY_UNSTABLE
        
        # Stress-based states
        if stress_level == 'high':
            if dominant_emotion == 'happy':
                return MentalState.HAPPY_UNDER_STRESS
            elif dominant_emotion == 'angry':
                return MentalState.ANGRY_STRESSED
            elif dominant_emotion == 'fear':
                return MentalState.ANXIOUS
            elif 'stress_persistence' in pattern_types:
                return MentalState.OVERWHELMED
            else:
                return MentalState.STRESSED
        
        # Low emotion, low stress = flat
        if dominant_emotion == 'neutral' and stress_level == 'low':
            if face_signal.confidence < 0.5 and voice_signal.confidence < 0.5:
                return MentalState.EMOTIONALLY_FLAT
        
        # Positive states
        if dominant_emotion == 'happy' and stress_level == 'low':
            if face_signal.confidence > 0.7 and voice_signal.confidence > 0.7:
                return MentalState.JOYFUL
            else:
                return MentalState.STABLE_POSITIVE
        
        # Negative states
        if dominant_emotion in ['sad', 'fear', 'angry']:
            if stress_level == 'medium' or stress_level == 'high':
                if dominant_emotion == 'fear':
                    return MentalState.ANXIOUS
                else:
                    return MentalState.STABLE_NEGATIVE
            elif dominant_emotion == 'sad' and stress_level == 'low':
                return MentalState.SAD_DEPRESSED
            elif dominant_emotion == 'fear':
                return MentalState.FEARFUL
        
        # Default: calm or confused
        if dominant_emotion == 'neutral' and stress_level == 'medium':
            return MentalState.CONFUSED
        
        return MentalState.CALM


# ============================================
# CONFIDENCE & EXPLANATION ENGINE
# ============================================

class ConfidenceCalculator:
    """
    Calculate overall confidence with penalties.
    """
    
    @staticmethod
    def calculate_confidence(face: NormalizedSignal, voice: NormalizedSignal,
                            stress: NormalizedSignal, patterns: List[TemporalPattern],
                            dominant_emotion: str, hidden_emotion: Optional[str]) -> float:
        """
        Calculate final confidence score.
        """
        
        # Base confidence: weighted average of modalities
        base_confidence = (
            face.weighted_confidence * 0.25 +
            voice.weighted_confidence * 0.35 +
            stress.weighted_confidence * 0.40
        )
        
        # Penalty 1: Conflict between modalities
        conflict_penalty = 0.0
        if hidden_emotion is not None:
            conflict_penalty = 0.15
        
        # Penalty 2: Instability
        instability_penalty = 0.0
        for pattern in patterns:
            if pattern.pattern_type == 'emotional_instability':
                instability_penalty = 0.10 * pattern.intensity
        
        # Penalty 3: Low signal quality
        avg_quality = (face.signal_quality + voice.signal_quality + stress.signal_quality) / 3
        quality_penalty = (1.0 - avg_quality) * 0.15
        
        # Final confidence
        final_confidence = base_confidence - conflict_penalty - instability_penalty - quality_penalty
        
        return max(0.0, min(1.0, final_confidence))


# ============================================
# RISK & SAFETY ASSESSMENT
# ============================================

class RiskAssessor:
    """
    Assess risk level based on mental state.
    """
    
    @staticmethod
    def assess_risk(mental_state: MentalState, stress_level: str,
                   patterns: List[TemporalPattern], confidence: float) -> RiskLevel:
        """
        Determine risk level.
        """
        
        # Critical risks
        if mental_state in [MentalState.OVERWHELMED, MentalState.ANXIOUS]:
            if any(p.pattern_type == 'stress_persistence' for p in patterns):
                return RiskLevel.CRITICAL
        
        # High risks
        if mental_state in [MentalState.ANGRY_STRESSED, MentalState.SAD_DEPRESSED]:
            return RiskLevel.HIGH
        
        if stress_level == 'high':
            stress_persistent = any(p.pattern_type == 'stress_persistence' for p in patterns)
            if stress_persistent:
                return RiskLevel.HIGH
            else:
                return RiskLevel.MODERATE
        
        # Moderate risks
        if mental_state in [MentalState.STRESSED, MentalState.EMOTIONALLY_MASKED, 
                           MentalState.EMOTIONALLY_UNSTABLE]:
            return RiskLevel.MODERATE
        
        if mental_state in [MentalState.FEARFUL, MentalState.STABLE_NEGATIVE]:
            return RiskLevel.MODERATE
        
        # Low risk
        return RiskLevel.LOW


# ============================================
# STABILITY CALCULATOR
# ============================================

class StabilityCalculator:
    """
    Calculate emotional stability score.
    """
    
    @staticmethod
    def calculate_stability(memory: TemporalMemory, patterns: List[TemporalPattern]) -> float:
        """
        Calculate stability score (0=unstable, 1=stable).
        """
        
        # Start with perfect stability
        stability = 1.0
        
        # Penalty 1: Emotion switches
        face_switches = memory.count_emotion_switches('face', window=10)
        voice_switches = memory.count_emotion_switches('voice', window=10)
        switch_penalty = min(0.5, (face_switches + voice_switches) * 0.05)
        stability -= switch_penalty
        
        # Penalty 2: Instability pattern
        for pattern in patterns:
            if pattern.pattern_type == 'emotional_instability':
                stability -= 0.3 * pattern.intensity
        
        # Bonus: Consistent emotions
        face_freq = memory.get_emotion_frequency('face')
        voice_freq = memory.get_emotion_frequency('voice')
        
        if face_freq and voice_freq:
            # If one emotion dominates (>60%), it's stable
            max_face_freq = max(face_freq.values()) if face_freq else 0
            max_voice_freq = max(voice_freq.values()) if voice_freq else 0
            
            if max_face_freq > 0.6 and max_voice_freq > 0.6:
                stability += 0.1
        
        return max(0.0, min(1.0, stability))


# ============================================
# PHASE 3 MAIN SYSTEM
# ============================================

class Phase3MultiModalFusion:
    """
    Complete Phase 3 system integrating all layers.
    """
    
    def __init__(self):
        self.normalizer = SignalNormalizer()
        self.memory = TemporalMemory(window_size=30)
        self.fusion_engine = FusionEngine()
        self.psychological_reasoner = PsychologicalReasoner()
        self.confidence_calculator = ConfidenceCalculator()
        self.risk_assessor = RiskAssessor()
        self.stability_calculator = StabilityCalculator()
        
        print("=" * 70)
        print("PHASE 3: MULTI-MODAL FUSION & PSYCHOLOGICAL REASONING")
        print("=" * 70)
        print("✓ Signal Normalization Layer")
        print("✓ Temporal Reasoning Layer")
        print("✓ Fusion Logic Layer")
        print("✓ Psychological Reasoning Layer")
        print("✓ Confidence & Explanation Engine")
        print("✓ Risk & Safety Assessment")
        print("=" * 70)
    
    def process_frame(self, 
                     face_emotion: str, face_confidence: float, face_detected: bool,
                     voice_emotion: str, voice_confidence: float, audio_quality: float,
                     stress_level: str, stress_confidence: float) -> PsychologicalState:
        """
        Process one frame of multi-modal input.
        
        Args:
            face_emotion: Predicted face emotion
            face_confidence: Face prediction confidence
            face_detected: Whether face was detected
            voice_emotion: Predicted voice emotion
            voice_confidence: Voice prediction confidence
            audio_quality: Audio signal quality (0-1)
            stress_level: Stress level ('low', 'medium', 'high')
            stress_confidence: Stress prediction confidence
        
        Returns:
            PsychologicalState: Complete psychological analysis
        """
        
        # Layer 1: Normalize signals
        face_signal = self.normalizer.normalize_face_signal(
            face_emotion, face_confidence, face_detected
        )
        voice_signal = self.normalizer.normalize_voice_signal(
            voice_emotion, voice_confidence, audio_quality
        )
        stress_signal = self.normalizer.normalize_stress_signal(
            stress_level, stress_confidence
        )
        
        # Layer 2: Update temporal memory
        self.memory.add_signals(face_signal, voice_signal, stress_signal)
        
        # Detect temporal patterns
        patterns = self.memory.detect_patterns()
        
        # Layer 3: Fuse signals
        dominant_emotion, hidden_emotion, fusion_explanations = self.fusion_engine.fuse_signals(
            face_signal, voice_signal, stress_signal
        )
        
        # Layer 4: Infer mental state
        mental_state = self.psychological_reasoner.infer_mental_state(
            dominant_emotion, hidden_emotion, stress_level, patterns,
            face_signal, voice_signal
        )
        
        # Calculate confidence
        confidence = self.confidence_calculator.calculate_confidence(
            face_signal, voice_signal, stress_signal, patterns,
            dominant_emotion, hidden_emotion
        )
        
        # Assess risk
        risk_level = self.risk_assessor.assess_risk(
            mental_state, stress_level, patterns, confidence
        )
        
        # Calculate stability
        stability = self.stability_calculator.calculate_stability(
            self.memory, patterns
        )
        
        # Build full explanation
        explanations = fusion_explanations.copy()
        explanations.append(f"Mental state: {mental_state.value}")
        
        for pattern in patterns:
            explanations.append(f"Pattern: {pattern.description}")
        
        explanations.append(f"Stability: {stability*100:.0f}%")
        explanations.append(f"Risk: {risk_level.value}")
        
        # Create psychological state
        state = PsychologicalState(
            dominant_emotion=dominant_emotion,
            hidden_emotion=hidden_emotion,
            mental_state=mental_state,
            confidence=confidence,
            explanations=explanations,
            risk_level=risk_level,
            stability_score=stability,
            temporal_patterns=patterns,
            raw_signals={
                'face': face_signal,
                'voice': voice_signal,
                'stress': stress_signal
            },
            timestamp=time.time()
        )
        
        return state
    
    def get_summary_statistics(self) -> Dict[str, any]:
        """Get temporal statistics"""
        return {
            'face_emotion_distribution': self.memory.get_emotion_frequency('face'),
            'voice_emotion_distribution': self.memory.get_emotion_frequency('voice'),
            'face_switches': self.memory.count_emotion_switches('face'),
            'voice_switches': self.memory.count_emotion_switches('voice'),
            'stress_persistent': self.memory.detect_stress_persistence()[0],
            'masking_detected': self.memory.detect_masking()[0],
        }


# ============================================
# VISUALIZATION & TESTING
# ============================================

def format_psychological_state(state: PsychologicalState) -> str:
    """Format psychological state for display"""
    lines = []
    lines.append("=" * 70)
    lines.append("PSYCHOLOGICAL STATE ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"Dominant Emotion: {state.dominant_emotion.upper()}")
    if state.hidden_emotion:
        lines.append(f"Hidden Emotion: {state.hidden_emotion.upper()}")
    lines.append(f"Mental State: {state.mental_state.value.upper().replace('_', ' ')}")
    lines.append(f"Confidence: {state.confidence*100:.1f}%")
    lines.append(f"Stability: {state.stability_score*100:.1f}%")
    lines.append(f"Risk Level: {state.risk_level.value.upper()}")
    lines.append("")
    lines.append("Reasoning:")
    for i, explanation in enumerate(state.explanations, 1):
        lines.append(f"  {i}. {explanation}")
    
    if state.temporal_patterns:
        lines.append("")
        lines.append("Temporal Patterns:")
        for pattern in state.temporal_patterns:
            lines.append(f"  • {pattern.description}")
    
    lines.append("=" * 70)
    return "\n".join(lines)


def test_phase3_scenarios():
    """Test Phase 3 with predefined scenarios"""
    
    phase3 = Phase3MultiModalFusion()
    
    print("\n\n" + "=" * 70)
    print("TESTING PHASE 3 SCENARIOS")
    print("=" * 70)
    
    # Scenario 1: Happy under stress
    print("\n\nSCENARIO 1: Person smiling but stressed")
    print("-" * 70)
    for i in range(15):
        state = phase3.process_frame(
            face_emotion='happy', face_confidence=0.75, face_detected=True,
            voice_emotion='happy', voice_confidence=0.65, audio_quality=0.9,
            stress_level='high', stress_confidence=0.82
        )
    print(format_psychological_state(state))
    
    # Scenario 2: Emotional masking
    print("\n\nSCENARIO 2: Neutral face but fearful voice")
    print("-" * 70)
    phase3 = Phase3MultiModalFusion()  # Reset
    for i in range(15):
        state = phase3.process_frame(
            face_emotion='neutral', face_confidence=0.70, face_detected=True,
            voice_emotion='fear', voice_confidence=0.68, audio_quality=0.85,
            stress_level='medium', stress_confidence=0.60
        )
    print(format_psychological_state(state))
    
    # Scenario 3: Emotional instability
    print("\n\nSCENARIO 3: Rapid emotion switches")
    print("-" * 70)
    phase3 = Phase3MultiModalFusion()  # Reset
    emotions = ['happy', 'sad', 'angry', 'neutral', 'fear', 'happy', 'angry', 'sad'] * 2
    for emotion in emotions:
        state = phase3.process_frame(
            face_emotion=emotion, face_confidence=0.60, face_detected=True,
            voice_emotion=emotion, voice_confidence=0.55, audio_quality=0.8,
            stress_level='medium', stress_confidence=0.65
        )
    print(format_psychological_state(state))
    
    # Scenario 4: Calm and stable
    print("\n\nSCENARIO 4: Calm and stable")
    print("-" * 70)
    phase3 = Phase3MultiModalFusion()  # Reset
    for i in range(15):
        state = phase3.process_frame(
            face_emotion='neutral', face_confidence=0.80, face_detected=True,
            voice_emotion='neutral', voice_confidence=0.75, audio_quality=0.95,
            stress_level='low', stress_confidence=0.85
        )
    print(format_psychological_state(state))


if __name__ == "__main__":
    test_phase3_scenarios()
