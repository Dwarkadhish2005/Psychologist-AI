"""
🧠 PHASE 5: PERSONALITY INFERENCE ENGINE
========================================

Purpose: Learn long-term behavioral patterns from emotional data
Author: Psychologist AI System
Date: January 11, 2026

Core Principle:
    PERSONALITY ≠ EMOTION
    
    Emotion:         Short-lived, noisy, context-dependent
    Personality:     Long-term, aggregated, statistically stable
    
Architecture:
    Phase 3 (Emotion) → Phase 4 (Temporal State) → Phase 5 (PSV)
    
Key Innovation:
    - Personality State Vector (PSV) - numerical trait representation
    - Exponential decay weighting - recent > ancient
    - Slow updates - prevents mood swings from becoming personality
    - No diagnosis - only behavioral descriptors

Safety & Ethics:
    ❌ Does NOT diagnose disorders
    ❌ Does NOT label people
    ❌ Does NOT make absolute claims
    ✅ Probabilistic, context-aware, adjustable
"""

import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

# Import Phase 4 structures
from inference.phase4_cognitive_layer import (
    SessionMetrics,
    DailyProfile,
    MentalState
)


# ============================================================================
# 📊 PERSONALITY STATE VECTOR (PSV) - THE HEART OF PHASE 5
# ============================================================================

@dataclass
class PersonalityStateVector:
    """
    🎯 Numerical representation of long-term behavioral tendencies
    
    These are NOT labels. These are measurements.
    Think of PSV as a behavioral fingerprint, not a diagnosis.
    
    All traits are:
        - Normalized to [0, 1]
        - Computed mathematically
        - Updated slowly over time
        - Context-aware
    """
    
    # ===== CORE TRAITS =====
    
    # 1. Emotional Stability
    # Measures: How much emotions fluctuate over time
    # Formula: 1 - variance(emotional_states)
    # High value = Consistent emotions
    # Low value = High emotional fluctuation
    emotional_stability: float = 0.5  # Default: neutral
    
    # 2. Stress Sensitivity
    # Measures: How easily stress rises
    # Formula: avg(stress_increase_rate) + stress_reaction_to_neutral
    # High value = Quick stress response
    # Low value = Stress-resistant
    stress_sensitivity: float = 0.5
    
    # 3. Recovery Speed
    # Measures: How quickly system returns to baseline after stress
    # Formula: 1 / avg(time_to_baseline)
    # High value = Fast emotional recovery (resilient)
    # Low value = Slow recovery (lingering effects)
    recovery_speed: float = 0.5
    
    # 4. Positivity Bias
    # Measures: Long-term emotional leaning
    # Formula: positive_time / total_time
    # High value = Positive emotional orientation
    # Low value = Negative emotional orientation
    positivity_bias: float = 0.5
    
    # 5. Volatility
    # Measures: How often emotional state changes
    # Formula: transitions / time
    # High value = Emotionally dynamic
    # Low value = Emotionally stable (few transitions)
    volatility: float = 0.5
    
    # ===== METADATA =====
    
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    total_sessions_processed: int = 0
    confidence: float = 0.0  # 0-1, based on data quantity
    
    # Trend tracking (last 10 updates)
    emotional_stability_history: List[float] = field(default_factory=list)
    stress_sensitivity_history: List[float] = field(default_factory=list)
    recovery_speed_history: List[float] = field(default_factory=list)
    positivity_bias_history: List[float] = field(default_factory=list)
    volatility_history: List[float] = field(default_factory=list)
    
    def get_trait_trends(self) -> Dict[str, str]:
        """
        Analyze trends in PSV traits over time
        
        Returns:
            Dictionary with trend direction for each trait
            Values: "increasing", "decreasing", "stable"
        """
        def analyze_trend(history: List[float]) -> str:
            """Determine trend from recent history"""
            if len(history) < 3:
                return "insufficient_data"
            
            recent = history[-5:]  # Last 5 updates
            
            # Linear regression slope
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent, 1)[0]
            
            if slope > 0.02:
                return "increasing"
            elif slope < -0.02:
                return "decreasing"
            else:
                return "stable"
        
        return {
            "emotional_stability": analyze_trend(self.emotional_stability_history),
            "stress_sensitivity": analyze_trend(self.stress_sensitivity_history),
            "recovery_speed": analyze_trend(self.recovery_speed_history),
            "positivity_bias": analyze_trend(self.positivity_bias_history),
            "volatility": analyze_trend(self.volatility_history)
        }
    
    def get_confidence_level(self) -> str:
        """
        Get human-readable confidence level
        
        Returns:
            "very_low", "low", "moderate", "high", "very_high"
        """
        if self.confidence < 0.2:
            return "very_low"
        elif self.confidence < 0.4:
            return "low"
        elif self.confidence < 0.6:
            return "moderate"
        elif self.confidence < 0.8:
            return "high"
        else:
            return "very_high"
    
    def get_behavioral_descriptor(self) -> str:
        """
        Generate professional, ethical behavioral description
        
        This is NOT a diagnosis. This is a contextual summary.
        
        Returns:
            Human-readable description of behavioral patterns
        """
        descriptors = []
        
        # Emotional stability descriptor
        if self.emotional_stability > 0.7:
            descriptors.append("demonstrates consistent emotional patterns")
        elif self.emotional_stability < 0.3:
            descriptors.append("shows variable emotional responses")
        else:
            descriptors.append("exhibits moderate emotional consistency")
        
        # Stress sensitivity descriptor
        if self.stress_sensitivity > 0.7:
            descriptors.append("with heightened stress responsiveness")
        elif self.stress_sensitivity < 0.3:
            descriptors.append("with notable stress resilience")
        else:
            descriptors.append("with moderate stress sensitivity")
        
        # Recovery speed descriptor
        if self.recovery_speed > 0.7:
            descriptors.append("and demonstrates quick emotional recovery")
        elif self.recovery_speed < 0.3:
            descriptors.append("and shows slower return to baseline")
        else:
            descriptors.append("with typical recovery patterns")
        
        # Combine into professional description
        base = f"User {descriptors[0]} {descriptors[1]} {descriptors[2]}."
        
        # Add confidence qualifier
        confidence_level = self.get_confidence_level()
        if confidence_level in ["very_low", "low"]:
            qualifier = " (Limited data - preliminary assessment)"
        elif confidence_level == "moderate":
            qualifier = " (Moderate confidence - ongoing assessment)"
        else:
            qualifier = ""
        
        return base + qualifier


# ============================================================================
# 🧮 PERSONALITY INFERENCE ENGINE
# ============================================================================

class PersonalityEngine:
    """
    🎯 PURPOSE: Learn long-term personality traits from emotional patterns
    
    KEY FEATURES:
        - Exponential decay weighting (recent > old)
        - Slow PSV updates (η = 0.01-0.05)
        - Confidence intervals based on data quantity
        - Ethical safety layer (no diagnoses)
    
    UPDATE RULE:
        PSV_new = (1 - η) * PSV_old + η * observation
        
        Where:
            η = learning rate (0.01 - 0.05)
            observation = aggregated behavior from new session
    
    TEMPORAL WINDOWS:
        - Short-term: Single session (minutes)
        - Mid-term: Daily aggregation (all sessions today)
        - Long-term: Multi-day patterns (weeks/months)
    """
    
    def __init__(
        self,
        user_id: str,
        storage_dir: str = "data/user_memory",
        learning_rate: float = 0.03,  # η - how fast PSV updates
        decay_lambda: float = 0.1,     # λ - exponential decay rate
        min_sessions_required: int = 3  # Minimum sessions before inference
    ):
        """
        Initialize Personality Engine
        
        Args:
            user_id: Unique user identifier
            storage_dir: Directory for PSV storage
            learning_rate: PSV update rate (0.01-0.05 recommended)
            decay_lambda: Exponential decay rate for temporal weighting
            min_sessions_required: Minimum sessions before PSV updates
        """
        self.user_id = user_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.decay_lambda = decay_lambda
        self.min_sessions_required = min_sessions_required
        
        # Initialize or load PSV
        self.psv = self._load_psv()
        
        # Session buffer for recent data
        self.session_buffer: List[SessionMetrics] = []
    
    def _load_psv(self) -> PersonalityStateVector:
        """
        Load existing PSV from disk or create new one
        
        Returns:
            PersonalityStateVector instance
        """
        psv_file = self.storage_dir / f"{self.user_id}_psv.json"
        
        if psv_file.exists():
            try:
                with open(psv_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct PSV
                psv = PersonalityStateVector(
                    emotional_stability=data["emotional_stability"],
                    stress_sensitivity=data["stress_sensitivity"],
                    recovery_speed=data["recovery_speed"],
                    positivity_bias=data["positivity_bias"],
                    volatility=data["volatility"],
                    last_updated=data["last_updated"],
                    total_sessions_processed=data["total_sessions_processed"],
                    confidence=data["confidence"],
                    emotional_stability_history=data.get("emotional_stability_history", []),
                    stress_sensitivity_history=data.get("stress_sensitivity_history", []),
                    recovery_speed_history=data.get("recovery_speed_history", []),
                    positivity_bias_history=data.get("positivity_bias_history", []),
                    volatility_history=data.get("volatility_history", [])
                )
                
                return psv
                
            except Exception as e:
                print(f"⚠️ Warning: Could not load PSV, creating new one. Error: {e}")
        
        # Create new PSV
        return PersonalityStateVector()
    
    def _save_psv(self):
        """Save PSV to disk"""
        psv_file = self.storage_dir / f"{self.user_id}_psv.json"
        
        # Convert to dict
        psv_dict = asdict(self.psv)
        
        # Save
        with open(psv_file, 'w') as f:
            json.dump(psv_dict, f, indent=2)
    
    def add_session(self, session_metrics: SessionMetrics):
        """
        Add new session data to buffer
        
        Args:
            session_metrics: Metrics from completed session
        """
        self.session_buffer.append(session_metrics)
    
    def can_infer_personality(self) -> bool:
        """
        Check if enough data exists for personality inference
        
        Returns:
            True if >= min_sessions_required sessions available
        """
        total_sessions = self.psv.total_sessions_processed + len(self.session_buffer)
        return total_sessions >= self.min_sessions_required
    
    def update_psv(self, daily_profiles: List[DailyProfile]):
        """
        Update PSV from daily profiles using exponential decay weighting
        
        Args:
            daily_profiles: List of DailyProfile objects (recent first)
        """
        if not self.can_infer_personality():
            return  # Not enough data yet
        
        # ===== COMPUTE NEW TRAIT OBSERVATIONS =====
        
        # 1. Emotional Stability (1 - variance of emotional states)
        stability_obs = self._compute_emotional_stability(daily_profiles)
        
        # 2. Stress Sensitivity (how easily stress rises)
        stress_sensitivity_obs = self._compute_stress_sensitivity(daily_profiles)
        
        # 3. Recovery Speed (how fast return to baseline)
        recovery_obs = self._compute_recovery_speed(daily_profiles)
        
        # 4. Positivity Bias (positive emotion ratio)
        positivity_obs = self._compute_positivity_bias(daily_profiles)
        
        # 5. Volatility (state transition frequency)
        volatility_obs = self._compute_volatility(daily_profiles)
        
        # ===== UPDATE PSV WITH SLOW LEARNING =====
        
        η = self.learning_rate
        
        # Update each trait
        self.psv.emotional_stability = (1 - η) * self.psv.emotional_stability + η * stability_obs
        self.psv.stress_sensitivity = (1 - η) * self.psv.stress_sensitivity + η * stress_sensitivity_obs
        self.psv.recovery_speed = (1 - η) * self.psv.recovery_speed + η * recovery_obs
        self.psv.positivity_bias = (1 - η) * self.psv.positivity_bias + η * positivity_obs
        self.psv.volatility = (1 - η) * self.psv.volatility + η * volatility_obs
        
        # ===== UPDATE HISTORY =====
        
        self.psv.emotional_stability_history.append(self.psv.emotional_stability)
        self.psv.stress_sensitivity_history.append(self.psv.stress_sensitivity)
        self.psv.recovery_speed_history.append(self.psv.recovery_speed)
        self.psv.positivity_bias_history.append(self.psv.positivity_bias)
        self.psv.volatility_history.append(self.psv.volatility)
        
        # Keep only last 10 updates
        if len(self.psv.emotional_stability_history) > 10:
            self.psv.emotional_stability_history = self.psv.emotional_stability_history[-10:]
            self.psv.stress_sensitivity_history = self.psv.stress_sensitivity_history[-10:]
            self.psv.recovery_speed_history = self.psv.recovery_speed_history[-10:]
            self.psv.positivity_bias_history = self.psv.positivity_bias_history[-10:]
            self.psv.volatility_history = self.psv.volatility_history[-10:]
        
        # ===== UPDATE METADATA =====
        
        self.psv.total_sessions_processed += len(self.session_buffer)
        self.psv.last_updated = datetime.now().isoformat()
        
        # Update confidence based on data quantity
        # Confidence grows with more sessions, caps at 1.0
        self.psv.confidence = min(1.0, self.psv.total_sessions_processed / 50.0)
        
        # Clear session buffer
        self.session_buffer = []
        
        # Save updated PSV
        self._save_psv()
    
    # ========================================================================
    # 🧮 TRAIT COMPUTATION METHODS
    # ========================================================================
    
    def _compute_emotional_stability(self, daily_profiles: List[DailyProfile]) -> float:
        """
        Compute emotional stability from daily profiles
        
        Formula: 1 - weighted_variance(emotional_states)
        
        High variance → Low stability
        Low variance → High stability
        
        Args:
            daily_profiles: Recent daily profiles (weighted by recency)
        
        Returns:
            Emotional stability score [0, 1]
        """
        if not daily_profiles:
            return 0.5  # Default neutral
        
        # Extract emotional metrics with exponential decay weights
        weights = self._compute_temporal_weights(len(daily_profiles))
        
        # Collect emotional variance indicators
        variances = []
        for profile in daily_profiles:
            # Use confidence variance as proxy for emotional stability
            # Also consider state switch frequency
            if hasattr(profile, 'emotional_volatility'):
                variances.append(profile.emotional_volatility)
            else:
                # Approximate from available metrics
                state_switches = profile.state_switches_per_minute if hasattr(profile, 'state_switches_per_minute') else 3.0
                variances.append(state_switches / 10.0)  # Normalize
        
        # Weighted average variance
        weighted_variance = np.average(variances, weights=weights)
        
        # Convert to stability (1 - variance)
        # Clip to ensure [0, 1]
        stability = 1.0 - np.clip(weighted_variance, 0, 1)
        
        return float(stability)
    
    def _compute_stress_sensitivity(self, daily_profiles: List[DailyProfile]) -> float:
        """
        Compute stress sensitivity from daily profiles
        
        Formula: weighted_avg(stress_ratio + high_stress_ratio)
        
        High stress time → High sensitivity
        Low stress time → Low sensitivity
        
        Args:
            daily_profiles: Recent daily profiles
        
        Returns:
            Stress sensitivity score [0, 1]
        """
        if not daily_profiles:
            return 0.5
        
        weights = self._compute_temporal_weights(len(daily_profiles))
        
        # Collect stress metrics
        stress_scores = []
        for profile in daily_profiles:
            # Combine multiple stress indicators
            stress_score = (
                profile.avg_stress_ratio * 0.5 +  # Regular stress
                profile.avg_high_stress_ratio * 1.5  # High stress (weighted more)
            )
            stress_scores.append(np.clip(stress_score, 0, 1))
        
        # Weighted average
        sensitivity = np.average(stress_scores, weights=weights)
        
        return float(sensitivity)
    
    def _compute_recovery_speed(self, daily_profiles: List[DailyProfile]) -> float:
        """
        Compute recovery speed from daily profiles
        
        Formula: 1 / avg(time_to_baseline)
        
        Fast recovery → High score
        Slow recovery → Low score
        
        Args:
            daily_profiles: Recent daily profiles
        
        Returns:
            Recovery speed score [0, 1]
        """
        if not daily_profiles:
            return 0.5
        
        weights = self._compute_temporal_weights(len(daily_profiles))
        
        # Use stability trend as proxy for recovery speed
        # Increasing stability = good recovery
        # Decreasing stability = poor recovery
        recovery_scores = []
        
        for profile in daily_profiles:
            # High stability + low stress escalations = fast recovery
            if hasattr(profile, 'avg_stability') and hasattr(profile, 'total_risk_escalations'):
                base_recovery = profile.avg_stability
                escalation_penalty = min(0.3, profile.total_risk_escalations / 20.0)
                recovery_scores.append(base_recovery - escalation_penalty)
            else:
                recovery_scores.append(0.5)
        
        # Weighted average
        recovery = np.average(recovery_scores, weights=weights)
        
        return float(np.clip(recovery, 0, 1))
    
    def _compute_positivity_bias(self, daily_profiles: List[DailyProfile]) -> float:
        """
        Compute positivity bias from daily profiles
        
        Formula: weighted_avg(positive_ratio)
        
        High positive time → High bias
        High negative time → Low bias
        
        Args:
            daily_profiles: Recent daily profiles
        
        Returns:
            Positivity bias score [0, 1]
        """
        if not daily_profiles:
            return 0.5
        
        weights = self._compute_temporal_weights(len(daily_profiles))
        
        # Collect positive/negative ratios
        positivity_scores = []
        for profile in daily_profiles:
            # Use positive ratio directly
            positivity_scores.append(profile.positive_ratio)
        
        # Weighted average
        positivity = np.average(positivity_scores, weights=weights)
        
        return float(np.clip(positivity, 0, 1))
    
    def _compute_volatility(self, daily_profiles: List[DailyProfile]) -> float:
        """
        Compute emotional volatility from daily profiles
        
        Formula: weighted_avg(state_switches_per_minute / baseline)
        
        High transitions → High volatility
        Low transitions → Low volatility
        
        Args:
            daily_profiles: Recent daily profiles
        
        Returns:
            Volatility score [0, 1]
        """
        if not daily_profiles:
            return 0.5
        
        weights = self._compute_temporal_weights(len(daily_profiles))
        
        # Collect state switch rates
        volatility_scores = []
        for profile in daily_profiles:
            if hasattr(profile, 'state_switches_per_minute'):
                # Normalize switches per minute (assume 5/min is high)
                normalized = profile.state_switches_per_minute / 5.0
                volatility_scores.append(np.clip(normalized, 0, 1))
            else:
                volatility_scores.append(0.5)
        
        # Weighted average
        volatility = np.average(volatility_scores, weights=weights)
        
        return float(volatility)
    
    def _compute_temporal_weights(self, n: int) -> np.ndarray:
        """
        Compute exponential decay weights for temporal data
        
        Recent data gets higher weight than old data.
        
        Formula: weight = e^(-λ * age)
        
        Args:
            n: Number of data points
        
        Returns:
            Array of weights (sum = 1.0)
        """
        # Age = 0 for most recent, n-1 for oldest
        ages = np.arange(n)
        
        # Exponential decay
        weights = np.exp(-self.decay_lambda * ages)
        
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        return weights
    
    # ========================================================================
    # 📊 OUTPUT & REPORTING
    # ========================================================================
    
    def get_personality_summary(self) -> Dict:
        """
        Get comprehensive personality summary
        
        Returns:
            Dictionary with PSV traits, trends, and behavioral descriptor
        """
        trends = self.psv.get_trait_trends()
        
        return {
            "personality_state_vector": {
                "emotional_stability": round(self.psv.emotional_stability, 3),
                "stress_sensitivity": round(self.psv.stress_sensitivity, 3),
                "recovery_speed": round(self.psv.recovery_speed, 3),
                "positivity_bias": round(self.psv.positivity_bias, 3),
                "volatility": round(self.psv.volatility, 3)
            },
            "trends": trends,
            "confidence": {
                "level": self.psv.get_confidence_level(),
                "score": round(self.psv.confidence, 3),
                "sessions_processed": self.psv.total_sessions_processed
            },
            "behavioral_descriptor": self.psv.get_behavioral_descriptor(),
            "last_updated": self.psv.last_updated,
            "ethical_notice": "This is a behavioral pattern assessment, not a clinical diagnosis."
        }
    
    def export_psv_for_research(self) -> Dict:
        """
        Export PSV data in research-friendly format
        
        Returns:
            Dictionary with complete PSV history and metadata
        """
        return {
            "user_id": self.user_id,
            "export_timestamp": datetime.now().isoformat(),
            "current_psv": {
                "emotional_stability": self.psv.emotional_stability,
                "stress_sensitivity": self.psv.stress_sensitivity,
                "recovery_speed": self.psv.recovery_speed,
                "positivity_bias": self.psv.positivity_bias,
                "volatility": self.psv.volatility
            },
            "psv_history": {
                "emotional_stability": self.psv.emotional_stability_history,
                "stress_sensitivity": self.psv.stress_sensitivity_history,
                "recovery_speed": self.psv.recovery_speed_history,
                "positivity_bias": self.psv.positivity_bias_history,
                "volatility": self.psv.volatility_history
            },
            "metadata": {
                "total_sessions": self.psv.total_sessions_processed,
                "confidence": self.psv.confidence,
                "confidence_level": self.psv.get_confidence_level(),
                "first_recorded": self.psv.last_updated,
                "learning_rate": self.learning_rate,
                "decay_lambda": self.decay_lambda
            },
            "ethical_disclaimer": "This data represents behavioral patterns aggregated over time. It is not a psychological assessment or clinical diagnosis. Use for research purposes only with appropriate ethical oversight."
        }


# ============================================================================
# 🎨 VISUALIZATION HELPERS (OPTIONAL)
# ============================================================================

def create_psv_radar_data(psv: PersonalityStateVector) -> Dict:
    """
    Create data structure for radar chart visualization
    
    Args:
        psv: PersonalityStateVector instance
    
    Returns:
        Dictionary with radar chart data
    """
    return {
        "labels": [
            "Emotional Stability",
            "Stress Sensitivity",
            "Recovery Speed",
            "Positivity Bias",
            "Volatility"
        ],
        "values": [
            psv.emotional_stability,
            psv.stress_sensitivity,
            psv.recovery_speed,
            psv.positivity_bias,
            psv.volatility
        ],
        "max_value": 1.0,
        "confidence": psv.confidence
    }


def generate_personality_report(personality_engine: PersonalityEngine) -> str:
    """
    Generate human-readable personality report
    
    Args:
        personality_engine: PersonalityEngine instance
    
    Returns:
        Formatted text report
    """
    summary = personality_engine.get_personality_summary()
    psv = summary["personality_state_vector"]
    trends = summary["trends"]
    confidence = summary["confidence"]
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║            PERSONALITY PATTERN ASSESSMENT REPORT                     ║
╚══════════════════════════════════════════════════════════════════════╝

User ID: {personality_engine.user_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

─────────────────────────────────────────────────────────────────────

PERSONALITY STATE VECTOR (PSV)

  • Emotional Stability:    {psv['emotional_stability']:.3f}  [{trends['emotional_stability']}]
  • Stress Sensitivity:     {psv['stress_sensitivity']:.3f}  [{trends['stress_sensitivity']}]
  • Recovery Speed:         {psv['recovery_speed']:.3f}  [{trends['recovery_speed']}]
  • Positivity Bias:        {psv['positivity_bias']:.3f}  [{trends['positivity_bias']}]
  • Volatility:             {psv['volatility']:.3f}  [{trends['volatility']}]

─────────────────────────────────────────────────────────────────────

BEHAVIORAL DESCRIPTOR

{summary['behavioral_descriptor']}

─────────────────────────────────────────────────────────────────────

ASSESSMENT CONFIDENCE

  Level:              {confidence['level'].upper()}
  Score:              {confidence['score']:.1%}
  Sessions Analyzed:  {confidence['sessions_processed']}

─────────────────────────────────────────────────────────────────────

ETHICAL NOTICE

{summary['ethical_notice']}

This assessment is based on aggregated behavioral patterns and does not
constitute a medical or psychological diagnosis. Results should be
interpreted in context with appropriate professional guidance.

╚══════════════════════════════════════════════════════════════════════╝
"""
    
    return report
