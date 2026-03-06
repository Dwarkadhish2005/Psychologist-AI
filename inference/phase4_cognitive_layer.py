"""
ðŸ§  PHASE 4: LONG-TERM COGNITIVE LAYER
=====================================

Purpose: Convert short-term states â†’ long-term traits & personalized risk
Author: Psychologist AI System
Date: January 10, 2026

Architecture:
    PHASE 3 (PsychologicalState) â†’ PHASE 4 â†’ UserPsychologicalProfile
    
Key Concepts:
    - STATE â‰  TRAIT (momentary vs personality)
    - BASELINE > ABSOLUTE (personalized thresholds)
    - DEVIATION DETECTION (unusual behavior alerts)
    - PERSONALIZED RISK (adjusted by history)

Modules:
    1. SessionMemory - Short-term tracking (30-90 mins)
    2. LongTermMemory - Cross-session storage (days/weeks)
    3. PersonalityProfile - Trait inference
    4. BaselineProfile - Personal normal
    5. DeviationDetector - Anomaly detection
    6. UserPsychologicalProfile - Final output
"""

import time
import json
import pickle
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import numpy as np

# Import Phase 3 structures
from inference.phase3_multimodal_fusion import (
    PsychologicalState,
    MentalState,
    RiskLevel
)

# Phase 5 will be imported after shared dataclasses are defined (see below)
PHASE5_AVAILABLE = False
PersonalityEngine = None
PersonalityStateVector = None
generate_personality_report = None


# ============================================================================
# ðŸ“Š MODULE 1: SESSION MEMORY (SHORT-TERM TRACKING)
# ============================================================================

@dataclass
class SessionMetrics:
    """Aggregated statistics from a single session (30-90 mins)"""
    
    # Temporal info
    session_start: float  # Unix timestamp
    session_duration: float  # seconds
    total_frames: int
    
    # Mental state distribution
    dominant_mental_states: List[Tuple[MentalState, float]]  # (state, frequency)
    mental_state_switches: int  # How many times state changed
    
    # Confidence & stability
    avg_confidence: float  # 0-1
    avg_stability: float  # 0-1
    confidence_variance: float
    stability_variance: float
    
    # Stress metrics
    stress_duration_ratio: float  # % of session under stress
    high_stress_duration_ratio: float  # % under high stress
    avg_stress_intensity: float  # 0-1
    
    # Masking & hidden emotions
    masking_frequency: float  # events per minute
    total_masking_events: int
    masking_duration_ratio: float  # % of session with hidden emotions
    
    # Risk assessment
    avg_risk_level: float  # 0-3 (LOW=0, MODERATE=1, HIGH=2, CRITICAL=3)
    high_risk_duration_ratio: float  # % at HIGH or CRITICAL
    risk_escalations: int  # Times risk level increased
    
    # Emotional patterns
    positive_emotion_ratio: float  # % calm/joyful/happy
    negative_emotion_ratio: float  # % anxious/depressed/unstable
    neutral_emotion_ratio: float  # % emotionally_masked/neutral states
    dominant_emotions: List[Tuple[str, float]]  # (emotion, frequency)


class SessionMemory:
    """
    ðŸŽ¯ PURPOSE: Track psychological states within one sitting (30-90 mins)
    
    STORES:
        - PsychologicalState objects with timestamps
        - Real-time aggregated metrics
        
    PROVIDES:
        - Session summary statistics
        - Temporal pattern analysis
        - Foundation for long-term memory
    """
    
    def __init__(self):
        """Initialize empty session"""
        self.states: List[Tuple[float, PsychologicalState]] = []  # (timestamp, state)
        self.start_time: float = time.time()
        self._last_mental_state: Optional[MentalState] = None
        self._state_switch_count: int = 0
        
    def add_state(self, state: PsychologicalState) -> None:
        """
        Add a new psychological state to the session
        
        Args:
            state: PsychologicalState from Phase 3
        """
        current_time = time.time()
        self.states.append((current_time, state))
        
        # Track state switches
        if self._last_mental_state is not None:
            if state.mental_state != self._last_mental_state:
                self._state_switch_count += 1
        self._last_mental_state = state.mental_state
    
    def get_duration(self) -> float:
        """Get session duration in seconds"""
        return time.time() - self.start_time
    
    def get_frame_count(self) -> int:
        """Get total frames/states recorded"""
        return len(self.states)
    
    def is_active(self, timeout_minutes: float = 5.0) -> bool:
        """
        Check if session is still active (recent activity)
        
        Args:
            timeout_minutes: Minutes of inactivity before session ends
            
        Returns:
            True if session has recent activity
        """
        if not self.states:
            return True  # New session
        
        last_timestamp, _ = self.states[-1]
        time_since_last = time.time() - last_timestamp
        return time_since_last < (timeout_minutes * 60)
    
    def calculate_metrics(self) -> SessionMetrics:
        """
        ðŸ“Š CORE METHOD: Calculate comprehensive session statistics
        
        Returns:
            SessionMetrics with all aggregated data
        """
        if not self.states:
            # Return empty metrics for empty session
            return SessionMetrics(
                session_start=self.start_time,
                session_duration=0.0,
                total_frames=0,
                dominant_mental_states=[],
                mental_state_switches=0,
                avg_confidence=0.0,
                avg_stability=0.0,
                confidence_variance=0.0,
                stability_variance=0.0,
                stress_duration_ratio=0.0,
                high_stress_duration_ratio=0.0,
                avg_stress_intensity=0.0,
                masking_frequency=0.0,
                total_masking_events=0,
                masking_duration_ratio=0.0,
                avg_risk_level=0.0,
                high_risk_duration_ratio=0.0,
                risk_escalations=0,
                positive_emotion_ratio=0.0,
                negative_emotion_ratio=0.0,
                neutral_emotion_ratio=0.0,
                dominant_emotions=[]
            )
        
        # Extract data arrays
        confidences = [state.confidence for _, state in self.states]
        stabilities = [state.stability_score for _, state in self.states]
        mental_states = [state.mental_state for _, state in self.states]
        risk_levels = [state.risk_level for _, state in self.states]
        hidden_emotions = [state.hidden_emotion for _, state in self.states]
        
        # Mental state distribution
        state_counts = Counter(mental_states)
        total_states = len(mental_states)
        dominant_states = [
            (state, count / total_states)
            for state, count in state_counts.most_common(5)
        ]
        
        # Stress metrics
        stress_states = [
            MentalState.STRESSED,
            MentalState.OVERWHELMED,
            MentalState.ANXIOUS,
            MentalState.EMOTIONALLY_UNSTABLE
        ]
        stress_count = sum(1 for s in mental_states if s in stress_states)
        stress_duration_ratio = stress_count / total_states if total_states > 0 else 0.0
        
        high_stress_states = [MentalState.OVERWHELMED, MentalState.EMOTIONALLY_UNSTABLE]
        high_stress_count = sum(1 for s in mental_states if s in high_stress_states)
        high_stress_ratio = high_stress_count / total_states if total_states > 0 else 0.0
        
        # Stress intensity (rough estimate: stressed=0.5, overwhelmed=1.0)
        stress_intensity_map = {
            MentalState.STRESSED: 0.5,
            MentalState.OVERWHELMED: 1.0,
            MentalState.ANXIOUS: 0.6,
            MentalState.EMOTIONALLY_UNSTABLE: 0.8
        }
        stress_intensities = [stress_intensity_map.get(s, 0.0) for s in mental_states]
        avg_stress_intensity = np.mean(stress_intensities) if stress_intensities else 0.0
        
        # Masking metrics
        masking_count = sum(1 for h in hidden_emotions if h is not None)
        masking_duration_ratio = masking_count / total_states if total_states > 0 else 0.0
        duration_minutes = self.get_duration() / 60
        masking_frequency = masking_count / duration_minutes if duration_minutes > 0 else 0.0
        
        # Risk metrics
        risk_level_map = {
            RiskLevel.LOW: 0,
            RiskLevel.MODERATE: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.CRITICAL: 3
        }
        risk_values = [risk_level_map[r] for r in risk_levels]
        avg_risk = np.mean(risk_values) if risk_values else 0.0
        
        high_risk_count = sum(1 for r in risk_levels if r in [RiskLevel.HIGH, RiskLevel.CRITICAL])
        high_risk_ratio = high_risk_count / total_states if total_states > 0 else 0.0
        
        # Risk escalations (times risk increased)
        risk_escalations = 0
        for i in range(1, len(risk_values)):
            if risk_values[i] > risk_values[i-1]:
                risk_escalations += 1
        
        # Emotional polarity
        positive_states = [MentalState.CALM, MentalState.JOYFUL, MentalState.STABLE_POSITIVE]
        negative_states = [
            MentalState.ANXIOUS, MentalState.SAD_DEPRESSED, 
            MentalState.EMOTIONALLY_UNSTABLE, MentalState.OVERWHELMED
        ]
        neutral_states = [MentalState.EMOTIONALLY_MASKED, MentalState.STRESSED]
        
        positive_count = sum(1 for s in mental_states if s in positive_states)
        negative_count = sum(1 for s in mental_states if s in negative_states)
        neutral_count = sum(1 for s in mental_states if s in neutral_states)
        
        positive_ratio = positive_count / total_states if total_states > 0 else 0.0
        negative_ratio = negative_count / total_states if total_states > 0 else 0.0
        neutral_ratio = neutral_count / total_states if total_states > 0 else 0.0
        
        # Emotion distribution
        emotions = [state.dominant_emotion for _, state in self.states]
        emotion_counts = Counter(emotions)
        dominant_emotions = [
            (emotion, count / total_states)
            for emotion, count in emotion_counts.most_common(5)
        ]
        
        # Build metrics object
        return SessionMetrics(
            session_start=self.start_time,
            session_duration=self.get_duration(),
            total_frames=total_states,
            dominant_mental_states=dominant_states,
            mental_state_switches=self._state_switch_count,
            avg_confidence=float(np.mean(confidences)),
            avg_stability=float(np.mean(stabilities)),
            confidence_variance=float(np.var(confidences)),
            stability_variance=float(np.var(stabilities)),
            stress_duration_ratio=stress_duration_ratio,
            high_stress_duration_ratio=high_stress_ratio,
            avg_stress_intensity=float(avg_stress_intensity),
            masking_frequency=masking_frequency,
            total_masking_events=masking_count,
            masking_duration_ratio=masking_duration_ratio,
            avg_risk_level=float(avg_risk),
            high_risk_duration_ratio=high_risk_ratio,
            risk_escalations=risk_escalations,
            positive_emotion_ratio=positive_ratio,
            negative_emotion_ratio=negative_ratio,
            neutral_emotion_ratio=neutral_ratio,
            dominant_emotions=dominant_emotions
        )
    
    def get_summary(self) -> Dict:
        """
        Get human-readable session summary
        
        Returns:
            Dictionary with key session insights
        """
        metrics = self.calculate_metrics()
        
        # Get top 3 mental states
        top_states = [
            f"{state.value} ({freq*100:.1f}%)"
            for state, freq in metrics.dominant_mental_states[:3]
        ]
        
        return {
            "duration_minutes": metrics.session_duration / 60,
            "total_frames": metrics.total_frames,
            "top_mental_states": top_states,
            "avg_confidence": f"{metrics.avg_confidence:.2f}",
            "avg_stability": f"{metrics.avg_stability:.2f}",
            "stress_time_percent": f"{metrics.stress_duration_ratio * 100:.1f}%",
            "masking_events": metrics.total_masking_events,
            "masking_frequency": f"{metrics.masking_frequency:.2f}/min",
            "avg_risk_level": f"{metrics.avg_risk_level:.2f}",
            "state_switches": metrics.mental_state_switches
        }
    
    def to_dict(self) -> Dict:
        """
        Serialize session to dictionary (for saving)
        
        Returns:
            Dictionary representation of session
        """
        metrics = self.calculate_metrics()
        metrics_dict = asdict(metrics)
        # Convert MentalState enums to strings so result is JSON-serializable
        metrics_dict["dominant_mental_states"] = [
            [s.value, f] for s, f in metrics.dominant_mental_states
        ]
        return {
            "start_time": self.start_time,
            "states": [
                {
                    "timestamp": ts,
                    "state": {
                        **asdict(state),
                        "mental_state": state.mental_state.value,
                        "risk_level": state.risk_level.value
                    }
                }
                for ts, state in self.states
            ],
            "metrics": metrics_dict
        }
    
    def reset(self) -> SessionMetrics:
        """
        Reset session and return final metrics
        
        Returns:
            Final SessionMetrics before reset
        """
        final_metrics = self.calculate_metrics()
        
        # Clear session
        self.states = []
        self.start_time = time.time()
        self._last_mental_state = None
        self._state_switch_count = 0
        
        return final_metrics


# ============================================================================
# ðŸ“š MODULE 2: LONG-TERM MEMORY (CROSS-SESSION STORAGE)
# ============================================================================

@dataclass
class DailyProfile:
    """Summary of all sessions from one day"""
    
    date: str  # YYYY-MM-DD format
    total_sessions: int
    total_duration_minutes: float
    
    # Aggregated mental state metrics
    avg_stress_ratio: float  # Average % of time stressed
    avg_high_stress_ratio: float
    avg_stress_intensity: float
    dominant_mental_states: List[Tuple[MentalState, float]]  # Top 3
    
    # Confidence & stability
    avg_confidence: float
    avg_stability: float
    confidence_trend: str  # "increasing", "decreasing", "stable"
    stability_trend: str
    
    # Masking behavior
    total_masking_events: int
    avg_masking_frequency: float  # events per minute
    masking_duration_ratio: float
    
    # Risk assessment
    avg_risk_level: float
    high_risk_duration_ratio: float
    total_risk_escalations: int
    
    # Emotional polarity
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    dominant_emotions: List[Tuple[str, float]]  # Top emotions
    
    # Session-level patterns
    state_switches_per_minute: float
    emotional_volatility: float  # Variance in emotion changes


@dataclass
class WeeklyAggregate:
    """Summary of entire week (7 days)"""
    
    week_start: str  # YYYY-MM-DD of Monday
    week_end: str
    total_days_active: int
    total_sessions: int
    
    # Weekly averages
    avg_daily_stress: float
    avg_daily_masking: float
    avg_daily_risk: float
    avg_daily_stability: float
    
    # Trends (slope coefficients)
    stress_trend: float  # Positive = increasing stress
    masking_trend: float
    risk_trend: float
    stability_trend: float
    
    # Most common patterns
    dominant_mental_states: List[Tuple[MentalState, float]]
    dominant_emotions: List[Tuple[str, float]]


# Phase 5 is imported lazily inside Phase4CognitiveFusion.__init__ to avoid
# circular import (phase5 imports SessionMetrics/DailyProfile from this module).


class LongTermMemory:
    """
    ðŸŽ¯ PURPOSE: Store and analyze psychological data across days/weeks
    
    STORES:
        - Daily summaries (DailyProfile)
        - Weekly aggregates (WeeklyAggregate)
        - Historical trends
        
    PROVIDES:
        - Baseline personality inference
        - Behavioral trend detection
        - Personalized risk calibration
        - Long-term pattern analysis
    
    PERSISTENCE:
        - Saves to JSON file
        - Auto-loads on init
        - Survives sessions
    """
    
    def __init__(self, user_id: str = "default_user", storage_dir: str = "data/memory", 
                 max_sessions_per_day: int = 10, max_days_stored: int = 90):
        """
        Initialize long-term memory for a user
        
        Args:
            user_id: Unique identifier for user
            storage_dir: Directory to store memory files
            max_sessions_per_day: Maximum sessions per day (prevents abuse)
            max_days_stored: Maximum days to keep in memory
        """
        self.user_id = user_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 4.2: Configuration
        self.max_sessions_per_day = max_sessions_per_day
        self.max_days_stored = max_days_stored
        
        self.memory_file = self.storage_dir / f"{user_id}_longterm_memory.json"
        self.archive_dir = self.storage_dir / "archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Data structures
        self.daily_profiles: Dict[str, DailyProfile] = {}  # date -> DailyProfile
        self.weekly_aggregates: Dict[str, WeeklyAggregate] = {}  # week_start -> Weekly
        
        # Load existing memory
        self.load()
    
    def add_session(self, session_metrics: SessionMetrics) -> None:
        """
        Add a completed session to long-term memory
        
        Args:
            session_metrics: Metrics from a SessionMemory.calculate_metrics()
        """
        # Get date from session start time
        session_date = datetime.fromtimestamp(session_metrics.session_start)
        date_str = session_date.strftime("%Y-%m-%d")
        
        # Get or create daily profile
        if date_str not in self.daily_profiles:
            self.daily_profiles[date_str] = self._create_empty_daily_profile(date_str)
        
        # PHASE 4.2: Check session limit per day
        if self.daily_profiles[date_str].total_sessions >= self.max_sessions_per_day:
            print(f" Warning: Session limit reached for {date_str} ({self.max_sessions_per_day} max)")
            print(f"   Merging into existing statistics instead of adding new session.")
            # Still update stats but don't increment session count
        
        # Update daily profile with session data
        self._update_daily_profile(date_str, session_metrics)
        
        # Update weekly aggregate
        self._update_weekly_aggregate(date_str)
        
        # PHASE 4.2: Cleanup old data and archive
        self._cleanup_old_data()
        
        # Auto-save after each session
        self.save()
    
    def _create_empty_daily_profile(self, date: str) -> DailyProfile:
        """Create empty daily profile for new day"""
        return DailyProfile(
            date=date,
            total_sessions=0,
            total_duration_minutes=0.0,
            avg_stress_ratio=0.0,
            avg_high_stress_ratio=0.0,
            avg_stress_intensity=0.0,
            dominant_mental_states=[],
            avg_confidence=0.0,
            avg_stability=0.0,
            confidence_trend="stable",
            stability_trend="stable",
            total_masking_events=0,
            avg_masking_frequency=0.0,
            masking_duration_ratio=0.0,
            avg_risk_level=0.0,
            high_risk_duration_ratio=0.0,
            total_risk_escalations=0,
            positive_ratio=0.0,
            negative_ratio=0.0,
            neutral_ratio=0.0,
            dominant_emotions=[],
            state_switches_per_minute=0.0,
            emotional_volatility=0.0
        )
    
    def _update_daily_profile(self, date: str, metrics: SessionMetrics) -> None:
        """
        Update daily profile with new session metrics
        
        Uses incremental averaging to combine sessions
        """
        profile = self.daily_profiles[date]
        n = profile.total_sessions  # Previous session count
        
        # Incremental averaging: new_avg = (old_avg * n + new_value) / (n + 1)
        def inc_avg(old_val: float, new_val: float) -> float:
            return (old_val * n + new_val) / (n + 1)
        
        # Update session count and duration
        profile.total_sessions += 1
        profile.total_duration_minutes += metrics.session_duration / 60
        
        # Update stress metrics
        profile.avg_stress_ratio = inc_avg(profile.avg_stress_ratio, metrics.stress_duration_ratio)
        profile.avg_high_stress_ratio = inc_avg(profile.avg_high_stress_ratio, metrics.high_stress_duration_ratio)
        profile.avg_stress_intensity = inc_avg(profile.avg_stress_intensity, metrics.avg_stress_intensity)
        
        # Update confidence & stability
        profile.avg_confidence = inc_avg(profile.avg_confidence, metrics.avg_confidence)
        profile.avg_stability = inc_avg(profile.avg_stability, metrics.avg_stability)
        
        # Update masking
        profile.total_masking_events += metrics.total_masking_events
        profile.avg_masking_frequency = inc_avg(profile.avg_masking_frequency, metrics.masking_frequency)
        profile.masking_duration_ratio = inc_avg(profile.masking_duration_ratio, metrics.masking_duration_ratio)
        
        # Update risk
        profile.avg_risk_level = inc_avg(profile.avg_risk_level, metrics.avg_risk_level)
        profile.high_risk_duration_ratio = inc_avg(profile.high_risk_duration_ratio, metrics.high_risk_duration_ratio)
        profile.total_risk_escalations += metrics.risk_escalations
        
        # Update emotional polarity
        profile.positive_ratio = inc_avg(profile.positive_ratio, metrics.positive_emotion_ratio)
        profile.negative_ratio = inc_avg(profile.negative_ratio, metrics.negative_emotion_ratio)
        profile.neutral_ratio = inc_avg(profile.neutral_ratio, metrics.neutral_emotion_ratio)
        
        # Update volatility
        duration_mins = metrics.session_duration / 60
        switches_per_min = metrics.mental_state_switches / duration_mins if duration_mins > 0 else 0
        profile.state_switches_per_minute = inc_avg(profile.state_switches_per_minute, switches_per_min)
        
        # Emotional volatility (using confidence variance as proxy)
        profile.emotional_volatility = inc_avg(profile.emotional_volatility, metrics.confidence_variance)
        
        # Merge dominant mental states
        profile.dominant_mental_states = self._merge_mental_states(
            profile.dominant_mental_states,
            metrics.dominant_mental_states,
            n,
            1
        )
        
        # Merge dominant emotions
        profile.dominant_emotions = self._merge_emotions(
            profile.dominant_emotions,
            metrics.dominant_emotions,
            n,
            1
        )
    
    def _merge_mental_states(
        self,
        old_states: List[Tuple[MentalState, float]],
        new_states: List[Tuple[MentalState, float]],
        old_weight: int,
        new_weight: int
    ) -> List[Tuple[MentalState, float]]:
        """
        Merge two mental state distributions with weights
        
        Returns top 5 mental states
        """
        # Combine all states
        state_freqs: Dict[MentalState, float] = {}
        
        for state, freq in old_states:
            state_freqs[state] = state_freqs.get(state, 0.0) + (freq * old_weight)
        
        for state, freq in new_states:
            state_freqs[state] = state_freqs.get(state, 0.0) + (freq * new_weight)
        
        # Normalize
        total_weight = old_weight + new_weight
        for state in state_freqs:
            state_freqs[state] /= total_weight
        
        # Return top 5
        sorted_states = sorted(state_freqs.items(), key=lambda x: x[1], reverse=True)
        return sorted_states[:5]
    
    def _merge_emotions(self, old_emotions: List[Tuple[str, float]], 
                       new_emotions: List[Tuple[str, float]],
                       old_weight: float, new_weight: float) -> List[Tuple[str, float]]:
        """
        Merge two emotion distributions with weights
        
        Returns top 5 emotions
        """
        # Combine all emotions
        emotion_freqs: Dict[str, float] = {}
        
        for emotion, freq in old_emotions:
            emotion_freqs[emotion] = emotion_freqs.get(emotion, 0.0) + (freq * old_weight)
        
        for emotion, freq in new_emotions:
            emotion_freqs[emotion] = emotion_freqs.get(emotion, 0.0) + (freq * new_weight)
        
        # Normalize
        total_weight = old_weight + new_weight
        for emotion in emotion_freqs:
            emotion_freqs[emotion] /= total_weight
        
        # Return top 5
        sorted_emotions = sorted(emotion_freqs.items(), key=lambda x: x[1], reverse=True)
        return sorted_emotions[:5]
    
    def _update_weekly_aggregate(self, date: str) -> None:
        """Update weekly aggregate for the week containing this date"""
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        
        # Get Monday of this week
        days_since_monday = date_obj.weekday()
        monday = date_obj - timedelta(days=days_since_monday)
        week_start = monday.strftime("%Y-%m-%d")
        
        # Get all days in this week that have data
        week_dates = []
        for i in range(7):
            day = monday + timedelta(days=i)
            day_str = day.strftime("%Y-%m-%d")
            if day_str in self.daily_profiles:
                week_dates.append(day_str)
        
        if not week_dates:
            return
        
        # Calculate weekly aggregate
        week_end = (monday + timedelta(days=6)).strftime("%Y-%m-%d")
        
        daily_data = [self.daily_profiles[d] for d in week_dates]
        total_sessions = sum(d.total_sessions for d in daily_data)
        
        # Averages
        avg_stress = np.mean([d.avg_stress_ratio for d in daily_data])
        avg_masking = np.mean([d.avg_masking_frequency for d in daily_data])
        avg_risk = np.mean([d.avg_risk_level for d in daily_data])
        avg_stability = np.mean([d.avg_stability for d in daily_data])
        
        # Trends (linear regression slope)
        def calculate_trend(values: List[float]) -> float:
            """Calculate linear trend (slope)"""
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            y = np.array(values)
            # Simple linear regression
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        
        stress_values = [d.avg_stress_ratio for d in daily_data]
        masking_values = [d.avg_masking_frequency for d in daily_data]
        risk_values = [d.avg_risk_level for d in daily_data]
        stability_values = [d.avg_stability for d in daily_data]
        
        # Dominant patterns across week
        all_mental_states: List[Tuple[MentalState, float]] = []
        for daily in daily_data:
            all_mental_states.extend(daily.dominant_mental_states)
        
        state_counter: Dict[MentalState, float] = {}
        for state, freq in all_mental_states:
            state_counter[state] = state_counter.get(state, 0.0) + freq
        
        dominant_states = sorted(state_counter.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Aggregate emotions across week
        all_emotions: List[Tuple[str, float]] = []
        for daily in daily_data:
            all_emotions.extend(daily.dominant_emotions)
        
        emotion_counter: Dict[str, float] = {}
        for emotion, freq in all_emotions:
            emotion_counter[emotion] = emotion_counter.get(emotion, 0.0) + freq
        
        dominant_emotions = sorted(emotion_counter.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Create weekly aggregate
        weekly = WeeklyAggregate(
            week_start=week_start,
            week_end=week_end,
            total_days_active=len(week_dates),
            total_sessions=total_sessions,
            avg_daily_stress=float(avg_stress),
            avg_daily_masking=float(avg_masking),
            avg_daily_risk=float(avg_risk),
            avg_daily_stability=float(avg_stability),
            stress_trend=calculate_trend(stress_values),
            masking_trend=calculate_trend(masking_values),
            risk_trend=calculate_trend(risk_values),
            stability_trend=calculate_trend(stability_values),
            dominant_mental_states=dominant_states,
            dominant_emotions=dominant_emotions
        )
        
        self.weekly_aggregates[week_start] = weekly
    
    def get_recent_days(self, num_days: int = 7) -> List[DailyProfile]:
        """
        Get most recent N days of daily profiles
        
        Args:
            num_days: Number of recent days to retrieve
            
        Returns:
            List of DailyProfile objects (newest first)
        """
        sorted_dates = sorted(self.daily_profiles.keys(), reverse=True)
        recent_dates = sorted_dates[:num_days]
        return [self.daily_profiles[d] for d in recent_dates]
    
    def get_recent_weeks(self, num_weeks: int = 4) -> List[WeeklyAggregate]:
        """
        Get most recent N weeks of weekly aggregates
        
        Args:
            num_weeks: Number of recent weeks to retrieve
            
        Returns:
            List of WeeklyAggregate objects (newest first)
        """
        sorted_weeks = sorted(self.weekly_aggregates.keys(), reverse=True)
        recent_weeks = sorted_weeks[:num_weeks]
        return [self.weekly_aggregates[w] for w in recent_weeks]
    
    def get_overall_trends(self) -> Dict[str, float]:
        """
        Calculate overall trends across all historical data
        
        Returns:
            Dictionary with trend slopes for key metrics
        """
        if not self.daily_profiles:
            return {}
        
        # Get all days sorted chronologically
        sorted_dates = sorted(self.daily_profiles.keys())
        daily_data = [self.daily_profiles[d] for d in sorted_dates]
        
        if len(daily_data) < 2:
            return {metric: 0.0 for metric in ["stress", "masking", "risk", "stability"]}
        
        # Calculate trends using linear regression
        def trend(values: List[float]) -> float:
            x = np.arange(len(values))
            y = np.array(values)
            return float(np.polyfit(x, y, 1)[0])
        
        return {
            "stress_trend": trend([d.avg_stress_ratio for d in daily_data]),
            "masking_trend": trend([d.avg_masking_frequency for d in daily_data]),
            "risk_trend": trend([d.avg_risk_level for d in daily_data]),
            "stability_trend": trend([d.avg_stability for d in daily_data]),
            "confidence_trend": trend([d.avg_confidence for d in daily_data])
        }
    
    def save(self) -> None:
        """Save long-term memory to JSON file"""
        
        def convert_to_serializable(obj):
            """Convert complex objects to JSON-serializable format"""
            if isinstance(obj, MentalState):
                return obj.value
            return obj
        
        # Prepare data with serialized enums
        daily_data = {}
        for date, profile in self.daily_profiles.items():
            profile_dict = asdict(profile)
            # Convert MentalState enums to strings
            profile_dict["dominant_mental_states"] = [
                [state.value, freq] for state, freq in profile.dominant_mental_states
            ]
            # Ensure emotions are in correct format
            profile_dict["dominant_emotions"] = [
                [emotion, freq] for emotion, freq in profile.dominant_emotions
            ]
            daily_data[date] = profile_dict
        
        weekly_data = {}
        for week, agg in self.weekly_aggregates.items():
            agg_dict = asdict(agg)
            # Convert MentalState enums to strings
            agg_dict["dominant_mental_states"] = [
                [state.value, freq] for state, freq in agg.dominant_mental_states
            ]
            agg_dict["dominant_emotions"] = [
                [emotion, freq] for emotion, freq in agg.dominant_emotions
            ]
            weekly_data[week] = agg_dict
        
        data = {
            "user_id": self.user_id,
            "last_updated": datetime.now().isoformat(),
            "daily_profiles": daily_data,
            "weekly_aggregates": weekly_data
        }
        
        with open(self.memory_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self) -> None:
        """Load long-term memory from JSON file"""
        if not self.memory_file.exists():
            return
        
        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct daily profiles
            for date, profile_dict in data.get("daily_profiles", {}).items():
                # Convert mental state strings back to enums
                if "dominant_mental_states" in profile_dict:
                    profile_dict["dominant_mental_states"] = [
                        (MentalState(state), freq)
                        for state, freq in profile_dict["dominant_mental_states"]
                    ]
                # Convert emotions to tuples
                if "dominant_emotions" in profile_dict:
                    profile_dict["dominant_emotions"] = [
                        (emotion, freq)
                        for emotion, freq in profile_dict.get("dominant_emotions", [])
                    ]
                else:
                    # For backward compatibility with old data
                    profile_dict["dominant_emotions"] = []
                self.daily_profiles[date] = DailyProfile(**profile_dict)
            
            # Reconstruct weekly aggregates
            for week, agg_dict in data.get("weekly_aggregates", {}).items():
                if "dominant_mental_states" in agg_dict:
                    agg_dict["dominant_mental_states"] = [
                        (MentalState(state), freq)
                        for state, freq in agg_dict["dominant_mental_states"]
                    ]
                if "dominant_emotions" in agg_dict:
                    # Already in correct format [emotion_str, freq]
                    agg_dict["dominant_emotions"] = [
                        (emotion, freq)
                        for emotion, freq in agg_dict.get("dominant_emotions", [])
                    ]
                self.weekly_aggregates[week] = WeeklyAggregate(**agg_dict)
                
        except Exception as e:
            print(f"âš ï¸ Warning: Could not load long-term memory: {e}")
            # Continue with empty memory
    
    def get_summary(self) -> Dict:
        """
        Get human-readable summary of long-term memory
        
        Returns:
            Dictionary with key insights
        """
        if not self.daily_profiles:
            return {"status": "No historical data"}
        
        recent_days = self.get_recent_days(7)
        recent_weeks = self.get_recent_weeks(4)
        trends = self.get_overall_trends()
        
        return {
            "total_days_tracked": len(self.daily_profiles),
            "total_weeks_tracked": len(self.weekly_aggregates),
            "recent_7_days_avg_stress": f"{np.mean([d.avg_stress_ratio for d in recent_days]):.2%}" if recent_days else "N/A",
            "recent_7_days_avg_stability": f"{np.mean([d.avg_stability for d in recent_days]):.2f}" if recent_days else "N/A",
            "stress_trend": "â†‘ increasing" if trends.get("stress_trend", 0) > 0.01 else "â†“ decreasing" if trends.get("stress_trend", 0) < -0.01 else "â†’ stable",
            "stability_trend": "â†‘ improving" if trends.get("stability_trend", 0) > 0.01 else "â†“ declining" if trends.get("stability_trend", 0) < -0.01 else "â†’ stable",
            "last_updated": datetime.fromtimestamp(os.path.getmtime(self.memory_file)).strftime("%Y-%m-%d %H:%M") if self.memory_file.exists() else "Never"
        }
    
    # ========================================================================
    # PHASE 4.2: SESSION LIMITS, EXPORT, AND ARCHIVE
    # ========================================================================
    
    def _cleanup_old_data(self) -> None:
        """
        Remove data older than max_days_stored
        Archives old data before deletion
        """
        cutoff_date = datetime.now() - timedelta(days=self.max_days_stored)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
        
        # Find old daily profiles
        old_dates = [
            date for date in self.daily_profiles.keys()
            if date < cutoff_str
        ]
        
        if not old_dates:
            return  # Nothing to clean up
        
        # Archive before deleting
        self._archive_data(old_dates)
        
        # Delete from memory
        for date in old_dates:
            del self.daily_profiles[date]
        
        print(f"ðŸ—‘ï¸ Cleaned up {len(old_dates)} days older than {self.max_days_stored} days")
        
        # Also clean up old weekly aggregates
        old_weeks = [
            week for week in self.weekly_aggregates.keys()
            if week < cutoff_str
        ]
        
        for week in old_weeks:
            del self.weekly_aggregates[week]
    
    def _archive_data(self, dates: List[str]) -> None:
        """
        Archive old data before deletion
        
        Args:
            dates: List of dates to archive
        """
        if not dates:
            return
        
        # Create archive filename with date range
        oldest = min(dates)
        newest = max(dates)
        archive_filename = f"{self.user_id}_archive_{oldest}_to_{newest}.json"
        archive_path = self.archive_dir / archive_filename
        
        # Collect data to archive
        archived_data = {
            "user_id": self.user_id,
            "archive_created": datetime.now().isoformat(),
            "date_range": {
                "start": oldest,
                "end": newest
            },
            "daily_profiles": {}
        }
        
        for date in dates:
            if date in self.daily_profiles:
                profile = self.daily_profiles[date]
                profile_dict = asdict(profile)
                # Convert enums to strings
                profile_dict["dominant_mental_states"] = [
                    [state.value, freq] for state, freq in profile.dominant_mental_states
                ]
                archived_data["daily_profiles"][date] = profile_dict
        
        # Save archive
        with open(archive_path, 'w') as f:
            json.dump(archived_data, f, indent=2)
        
        print(f"ðŸ“¦ Archived {len(dates)} days to {archive_filename}")
    
    def export_to_csv(self, filepath: str = None) -> str:
        """
        Export data to CSV for external analysis
        
        Args:
            filepath: Output CSV path (default: auto-generated)
            
        Returns:
            Path to created CSV file
        """
        import csv
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.storage_dir / f"{self.user_id}_export_{timestamp}.csv"
        
        filepath = Path(filepath)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Date',
                'Total_Sessions',
                'Duration_Minutes',
                'Avg_Stress_Ratio',
                'Avg_High_Stress_Ratio',
                'Avg_Confidence',
                'Avg_Stability',
                'Total_Masking_Events',
                'Avg_Risk_Level',
                'High_Risk_Duration_Ratio',
                'Positive_Ratio',
                'Negative_Ratio',
                'Dominant_Mental_State'
            ])
            
            # Data rows (sorted by date)
            sorted_dates = sorted(self.daily_profiles.keys())
            for date in sorted_dates:
                profile = self.daily_profiles[date]
                
                # Get dominant mental state
                dominant_state = profile.dominant_mental_states[0][0].value if profile.dominant_mental_states else "N/A"
                
                writer.writerow([
                    date,
                    profile.total_sessions,
                    f"{profile.total_duration_minutes:.1f}",
                    f"{profile.avg_stress_ratio:.4f}",
                    f"{profile.avg_high_stress_ratio:.4f}",
                    f"{profile.avg_confidence:.4f}",
                    f"{profile.avg_stability:.4f}",
                    profile.total_masking_events,
                    f"{profile.avg_risk_level:.4f}",
                    f"{profile.high_risk_duration_ratio:.4f}",
                    f"{profile.positive_ratio:.4f}",
                    f"{profile.negative_ratio:.4f}",
                    dominant_state
                ])
        
        print(f"ðŸ“Š Exported data to {filepath}")
        return str(filepath)
    
    def export_to_json(self, filepath: str = None) -> str:
        """
        Export complete data to JSON
        
        Args:
            filepath: Output JSON path (default: auto-generated)
            
        Returns:
            Path to created JSON file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.storage_dir / f"{self.user_id}_export_{timestamp}.json"
        
        filepath = Path(filepath)
        
        # Prepare export data (same as save but with metadata)
        daily_data = {}
        for date, profile in self.daily_profiles.items():
            profile_dict = asdict(profile)
            profile_dict["dominant_mental_states"] = [
                [state.value, freq] for state, freq in profile.dominant_mental_states
            ]
            daily_data[date] = profile_dict
        
        weekly_data = {}
        for week, agg in self.weekly_aggregates.items():
            agg_dict = asdict(agg)
            agg_dict["dominant_mental_states"] = [
                [state.value, freq] for state, freq in agg.dominant_mental_states
            ]
            agg_dict["dominant_emotions"] = [
                [emotion, freq] for emotion, freq in agg.dominant_emotions
            ]
            weekly_data[week] = agg_dict
        
        export_data = {
            "export_info": {
                "user_id": self.user_id,
                "export_date": datetime.now().isoformat(),
                "total_days": len(self.daily_profiles),
                "total_weeks": len(self.weekly_aggregates),
                "date_range": {
                    "earliest": min(self.daily_profiles.keys()) if self.daily_profiles else None,
                    "latest": max(self.daily_profiles.keys()) if self.daily_profiles else None
                }
            },
            "daily_profiles": daily_data,
            "weekly_aggregates": weekly_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"ðŸ“Š Exported data to {filepath}")
        return str(filepath)
    
    def get_archive_list(self) -> List[Dict]:
        """
        List all archived files
        
        Returns:
            List of archive metadata
        """
        archives = []
        
        for archive_file in self.archive_dir.glob(f"{self.user_id}_archive_*.json"):
            # Extract date range from filename
            parts = archive_file.stem.split('_')
            if len(parts) >= 5:
                start_date = parts[2]
                end_date = parts[4]
                
                archives.append({
                    'filename': archive_file.name,
                    'path': str(archive_file),
                    'date_range': f"{start_date} to {end_date}",
                    'size_kb': archive_file.stat().st_size / 1024,
                    'created': datetime.fromtimestamp(archive_file.stat().st_ctime).isoformat()
                })
        
        return sorted(archives, key=lambda x: x['created'], reverse=True)


# ============================================================================
# ðŸŽ­ MODULE 3: PERSONALITY PROFILE (TRAIT INFERENCE)
# ============================================================================

@dataclass
class PersonalityProfile:
    """
    ðŸŽ­ PERSONALITY TRAITS inferred from long-term behavioral patterns
    
    All traits scored 0.0-1.0
    These are SOFT psychological tendencies, NOT clinical diagnoses
    """
    
    # Core personality dimensions
    emotional_reactivity: float  # 0=stable, 1=highly reactive (fast emotion changes)
    stress_tolerance: float  # 0=low tolerance, 1=high tolerance
    masking_tendency: float  # 0=rarely masks, 1=frequently masks emotions
    emotional_stability: float  # 0=unstable, 1=very stable
    baseline_mood: str  # "positive", "neutral", "negative"
    
    # Derived insights
    volatility_score: float  # Overall emotional volatility (0=calm, 1=chaotic)
    resilience_score: float  # Ability to recover from stress (0=fragile, 1=resilient)
    authenticity_score: float  # Shows true emotions (0=masks often, 1=authentic)
    
    # Metadata
    confidence: float  # 0-1 (how much data supports this profile)
    data_days: int  # Number of days used for inference
    last_updated: str  # ISO timestamp


class PersonalityInferenceEngine:
    """
    ðŸ§  INFERS PERSONALITY TRAITS FROM LONG-TERM MEMORY
    
    Uses pure statistics (no ML) for explainability:
        - Aggregates behavioral patterns
        - Calculates trait scores
        - Provides reasoning for each trait
    
    Minimum 3 days of data recommended for reliable inference
    """
    
    def __init__(self, min_days: int = 3):
        """
        Initialize inference engine
        
        Args:
            min_days: Minimum days of data needed for reliable inference
        """
        self.min_days = min_days
    
    def infer_personality(self, ltm: LongTermMemory) -> PersonalityProfile:
        """
        ðŸŽ¯ CORE METHOD: Infer personality profile from long-term memory
        
        Args:
            ltm: LongTermMemory instance with historical data
            
        Returns:
            PersonalityProfile with all trait scores
        """
        recent_days = ltm.get_recent_days(30)  # Use last 30 days
        
        if len(recent_days) < self.min_days:
            # Not enough data - return neutral profile
            return self._default_profile(len(recent_days))
        
        # Calculate each personality dimension
        emotional_reactivity = self._calculate_emotional_reactivity(recent_days)
        stress_tolerance = self._calculate_stress_tolerance(recent_days)
        masking_tendency = self._calculate_masking_tendency(recent_days)
        emotional_stability = self._calculate_emotional_stability(recent_days)
        baseline_mood = self._calculate_baseline_mood(recent_days)
        
        # Calculate derived scores
        volatility_score = self._calculate_volatility(emotional_reactivity, emotional_stability)
        resilience_score = self._calculate_resilience(stress_tolerance, emotional_stability)
        authenticity_score = 1.0 - masking_tendency  # Inverse of masking
        
        # Confidence based on data quantity and consistency
        confidence = self._calculate_confidence(recent_days)
        
        return PersonalityProfile(
            emotional_reactivity=emotional_reactivity,
            stress_tolerance=stress_tolerance,
            masking_tendency=masking_tendency,
            emotional_stability=emotional_stability,
            baseline_mood=baseline_mood,
            volatility_score=volatility_score,
            resilience_score=resilience_score,
            authenticity_score=authenticity_score,
            confidence=confidence,
            data_days=len(recent_days),
            last_updated=datetime.now().isoformat()
        )
    
    def _calculate_emotional_reactivity(self, days: List[DailyProfile]) -> float:
        """
        Emotional Reactivity: How quickly emotions change
        
        Metrics:
            - State switches per minute
            - Emotional volatility
            
        Returns: 0.0 (stable) to 1.0 (highly reactive)
        """
        avg_switches = np.mean([d.state_switches_per_minute for d in days])
        avg_volatility = np.mean([d.emotional_volatility for d in days])
        
        # Normalize switches (assume 0-10 switches/min is typical range)
        switches_normalized = min(avg_switches / 10.0, 1.0)
        
        # Combine metrics (60% switches, 40% volatility)
        reactivity = (switches_normalized * 0.6) + (avg_volatility * 0.4)
        
        return float(np.clip(reactivity, 0.0, 1.0))
    
    def _calculate_stress_tolerance(self, days: List[DailyProfile]) -> float:
        """
        Stress Tolerance: Ability to handle stress without risk escalation
        
        Metrics:
            - Inverse of stress ratio (less stress = higher tolerance)
            - Inverse of high-risk time
            
        Returns: 0.0 (low tolerance) to 1.0 (high tolerance)
        """
        avg_stress = np.mean([d.avg_stress_ratio for d in days])
        avg_high_risk = np.mean([d.high_risk_duration_ratio for d in days])
        
        # Tolerance is inverse of stress/risk
        stress_tolerance = 1.0 - avg_stress
        risk_tolerance = 1.0 - avg_high_risk
        
        # Combine (70% stress, 30% risk)
        tolerance = (stress_tolerance * 0.7) + (risk_tolerance * 0.3)
        
        return float(np.clip(tolerance, 0.0, 1.0))
    
    def _calculate_masking_tendency(self, days: List[DailyProfile]) -> float:
        """
        Masking Tendency: How often emotions are hidden
        
        Metrics:
            - Masking frequency (events per minute)
            - Masking duration ratio
            
        Returns: 0.0 (rarely masks) to 1.0 (frequently masks)
        """
        avg_masking_freq = np.mean([d.avg_masking_frequency for d in days])
        avg_masking_duration = np.mean([d.masking_duration_ratio for d in days])
        
        # Normalize frequency (assume 0-5 events/min is typical)
        freq_normalized = min(avg_masking_freq / 5.0, 1.0)
        
        # Combine (50/50 split)
        masking = (freq_normalized * 0.5) + (avg_masking_duration * 0.5)
        
        return float(np.clip(masking, 0.0, 1.0))
    
    def _calculate_emotional_stability(self, days: List[DailyProfile]) -> float:
        """
        Emotional Stability: Consistency of emotional state over time
        
        Metrics:
            - Average stability scores
            - Variance in daily stability
            
        Returns: 0.0 (unstable) to 1.0 (very stable)
        """
        stability_scores = [d.avg_stability for d in days]
        avg_stability = np.mean(stability_scores)
        stability_variance = np.var(stability_scores)
        
        # Low variance = more consistent = more stable
        consistency_bonus = 1.0 - min(stability_variance * 2.0, 0.3)  # Max 0.3 penalty
        
        # Combine base stability with consistency
        total_stability = avg_stability * consistency_bonus
        
        return float(np.clip(total_stability, 0.0, 1.0))
    
    def _calculate_baseline_mood(self, days: List[DailyProfile]) -> str:
        """
        Baseline Mood: Overall typical emotional polarity
        
        Metrics:
            - Average positive/negative/neutral ratios
            
        Returns: "positive", "neutral", or "negative"
        """
        avg_positive = np.mean([d.positive_ratio for d in days])
        avg_negative = np.mean([d.negative_ratio for d in days])
        avg_neutral = np.mean([d.neutral_ratio for d in days])
        
        # Determine dominant mood
        max_ratio = max(avg_positive, avg_negative, avg_neutral)
        
        if max_ratio == avg_positive and avg_positive > 0.4:
            return "positive"
        elif max_ratio == avg_negative and avg_negative > 0.4:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_volatility(self, reactivity: float, stability: float) -> float:
        """
        Volatility: Overall emotional chaos/predictability
        
        High volatility = reactive + unstable
        Low volatility = calm + stable
        """
        # Reactivity contributes positively, stability negatively
        volatility = (reactivity * 0.6) + ((1.0 - stability) * 0.4)
        return float(np.clip(volatility, 0.0, 1.0))
    
    def _calculate_resilience(self, stress_tolerance: float, stability: float) -> float:
        """
        Resilience: Ability to maintain composure under pressure
        
        High resilience = tolerates stress + stays stable
        """
        resilience = (stress_tolerance * 0.6) + (stability * 0.4)
        return float(np.clip(resilience, 0.0, 1.0))
    
    def _calculate_confidence(self, days: List[DailyProfile]) -> float:
        """
        Confidence: How much we trust this personality profile
        
        Factors:
            - Number of days (more = higher confidence)
            - Data consistency (less variance = higher confidence)
        """
        num_days = len(days)
        
        # Base confidence from data quantity
        if num_days < 3:
            quantity_confidence = num_days / 3.0
        elif num_days < 7:
            quantity_confidence = 0.7 + ((num_days - 3) / 4.0 * 0.2)  # 0.7 to 0.9
        else:
            quantity_confidence = 0.9 + min((num_days - 7) / 30.0 * 0.1, 0.1)  # 0.9 to 1.0
        
        # Consistency bonus (low variance in key metrics)
        stress_variance = np.var([d.avg_stress_ratio for d in days])
        stability_variance = np.var([d.avg_stability for d in days])
        
        avg_variance = (stress_variance + stability_variance) / 2.0
        consistency_confidence = 1.0 - min(avg_variance * 5.0, 0.2)  # Max 0.2 penalty
        
        # Combine
        total_confidence = quantity_confidence * consistency_confidence
        
        return float(np.clip(total_confidence, 0.0, 1.0))
    
    def _default_profile(self, data_days: int) -> PersonalityProfile:
        """Return neutral profile when insufficient data"""
        return PersonalityProfile(
            emotional_reactivity=0.5,
            stress_tolerance=0.5,
            masking_tendency=0.5,
            emotional_stability=0.5,
            baseline_mood="neutral",
            volatility_score=0.5,
            resilience_score=0.5,
            authenticity_score=0.5,
            confidence=0.0,
            data_days=data_days,
            last_updated=datetime.now().isoformat()
        )
    
    def get_personality_description(self, profile: PersonalityProfile) -> str:
        """
        Generate human-readable personality description
        
        Args:
            profile: PersonalityProfile to describe
            
        Returns:
            Natural language description
        """
        if profile.confidence < 0.3:
            return "âš ï¸ Insufficient data for reliable personality assessment"
        
        # Interpret each trait
        def interpret(value: float, low: str, mid: str, high: str) -> str:
            if value < 0.33:
                return low
            elif value < 0.67:
                return mid
            else:
                return high
        
        reactivity_desc = interpret(
            profile.emotional_reactivity,
            "emotionally steady",
            "moderately reactive",
            "highly emotionally reactive"
        )
        
        tolerance_desc = interpret(
            profile.stress_tolerance,
            "stress-sensitive",
            "moderate stress tolerance",
            "high stress tolerance"
        )
        
        masking_desc = interpret(
            profile.masking_tendency,
            "emotionally authentic",
            "occasionally masks emotions",
            "frequently masks emotions"
        )
        
        stability_desc = interpret(
            profile.emotional_stability,
            "emotionally variable",
            "moderately stable",
            "emotionally stable"
        )
        
        description = f"""
ðŸŽ­ PERSONALITY PROFILE (Confidence: {profile.confidence:.0%})

Emotional Reactivity: {reactivity_desc} ({profile.emotional_reactivity:.2f})
Stress Tolerance: {tolerance_desc} ({profile.stress_tolerance:.2f})
Masking Tendency: {masking_desc} ({profile.masking_tendency:.2f})
Emotional Stability: {stability_desc} ({profile.emotional_stability:.2f})
Baseline Mood: {profile.baseline_mood}

ðŸ’¡ Derived Insights:
  â€¢ Volatility: {profile.volatility_score:.2f} (0=calm, 1=chaotic)
  â€¢ Resilience: {profile.resilience_score:.2f} (0=fragile, 1=resilient)
  â€¢ Authenticity: {profile.authenticity_score:.2f} (0=masked, 1=genuine)

ðŸ“Š Based on {profile.data_days} days of data
        """.strip()
        
        return description


# ============================================================================
# ðŸ“Š MODULE 4: BASELINE PROFILE (PERSONAL "NORMAL")
# ============================================================================

@dataclass
class BaselineProfile:
    """
    ðŸ“Š PERSONAL BASELINE - What's "normal" for this user
    
    This becomes the reference point for deviation detection
    All values are averages from recent history (7-30 days)
    """
    
    # Core metrics (averages)
    avg_stress_level: float  # 0-1
    avg_confidence: float  # 0-1
    avg_stability: float  # 0-1
    avg_masking_frequency: float  # Events per minute
    normal_risk_level: float  # 0-3 (LOW=0, CRITICAL=3)
    
    # Mental state distribution
    typical_mental_states: List[Tuple[MentalState, float]]  # Top 3
    typical_emotions: List[Tuple[str, float]]  # Top 3
    
    # Behavioral patterns
    typical_state_switches_per_min: float
    typical_emotional_volatility: float
    
    # Thresholds for deviation detection (derived from variance)
    stress_threshold: float  # When stress is "unusually high"
    stability_threshold: float  # When stability is "unusually low"
    masking_threshold: float  # When masking is "unusually high"
    
    # Metadata
    confidence: float  # 0-1 (how stable is this baseline)
    data_days: int
    last_updated: str


class BaselineBuilder:
    """
    ðŸ”¨ BUILDS BASELINE PROFILE FROM LONG-TERM MEMORY
    
    Uses recent history (7-30 days) to define "normal"
    Calculates deviation thresholds based on variance
    """
    
    def __init__(self, history_days: int = 14):
        """
        Initialize baseline builder
        
        Args:
            history_days: Number of recent days to use for baseline
        """
        self.history_days = history_days
    
    def build_baseline(self, ltm: LongTermMemory) -> BaselineProfile:
        """
        ðŸŽ¯ CORE METHOD: Build baseline from long-term memory
        
        Args:
            ltm: LongTermMemory with historical data
            
        Returns:
            BaselineProfile defining "normal" for this user
        """
        recent_days = ltm.get_recent_days(self.history_days)
        
        if len(recent_days) < 3:
            # Not enough data
            return self._default_baseline(len(recent_days))
        
        # Calculate averages
        avg_stress = np.mean([d.avg_stress_ratio for d in recent_days])
        avg_confidence = np.mean([d.avg_confidence for d in recent_days])
        avg_stability = np.mean([d.avg_stability for d in recent_days])
        avg_masking_freq = np.mean([d.avg_masking_frequency for d in recent_days])
        avg_risk = np.mean([d.avg_risk_level for d in recent_days])
        
        # Behavioral patterns
        avg_switches = np.mean([d.state_switches_per_minute for d in recent_days])
        avg_volatility = np.mean([d.emotional_volatility for d in recent_days])
        
        # Typical mental states (aggregate all days)
        all_states: Dict[MentalState, float] = {}
        for day in recent_days:
            for state, freq in day.dominant_mental_states:
                all_states[state] = all_states.get(state, 0.0) + freq
        
        # Normalize and get top 3
        total_freq = sum(all_states.values())
        if total_freq > 0:
            all_states = {s: f / total_freq for s, f in all_states.items()}
        typical_states = sorted(all_states.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Calculate deviation thresholds (mean + 1.5 * std_dev)
        stress_values = [d.avg_stress_ratio for d in recent_days]
        stability_values = [d.avg_stability for d in recent_days]
        masking_values = [d.avg_masking_frequency for d in recent_days]
        
        stress_threshold = avg_stress + (1.5 * np.std(stress_values))
        stability_threshold = avg_stability - (1.5 * np.std(stability_values))  # Low stability is bad
        masking_threshold = avg_masking_freq + (1.5 * np.std(masking_values))
        
        # Confidence (higher for more days + lower variance)
        confidence = self._calculate_baseline_confidence(recent_days)
        
        return BaselineProfile(
            avg_stress_level=float(avg_stress),
            avg_confidence=float(avg_confidence),
            avg_stability=float(avg_stability),
            avg_masking_frequency=float(avg_masking_freq),
            normal_risk_level=float(avg_risk),
            typical_mental_states=typical_states,
            typical_emotions=[],  # TODO: Track emotions separately
            typical_state_switches_per_min=float(avg_switches),
            typical_emotional_volatility=float(avg_volatility),
            stress_threshold=float(stress_threshold),
            stability_threshold=float(max(stability_threshold, 0.0)),
            masking_threshold=float(masking_threshold),
            confidence=confidence,
            data_days=len(recent_days),
            last_updated=datetime.now().isoformat()
        )
    
    def _calculate_baseline_confidence(self, days: List[DailyProfile]) -> float:
        """Calculate confidence in baseline stability"""
        num_days = len(days)
        
        # Quantity confidence
        if num_days < 7:
            quantity_conf = num_days / 7.0
        else:
            quantity_conf = min(0.7 + ((num_days - 7) / 14.0 * 0.3), 1.0)
        
        # Consistency confidence (low variance = high confidence)
        stress_var = np.var([d.avg_stress_ratio for d in days])
        stability_var = np.var([d.avg_stability for d in days])
        
        avg_var = (stress_var + stability_var) / 2.0
        consistency_conf = 1.0 - min(avg_var * 10.0, 0.3)
        
        return float(quantity_conf * consistency_conf)
    
    def _default_baseline(self, data_days: int) -> BaselineProfile:
        """Return neutral baseline when insufficient data"""
        return BaselineProfile(
            avg_stress_level=0.3,
            avg_confidence=0.7,
            avg_stability=0.7,
            avg_masking_frequency=1.0,
            normal_risk_level=0.5,
            typical_mental_states=[],
            typical_emotions=[],
            typical_state_switches_per_min=2.0,
            typical_emotional_volatility=0.3,
            stress_threshold=0.7,
            stability_threshold=0.4,
            masking_threshold=3.0,
            confidence=0.0,
            data_days=data_days,
            last_updated=datetime.now().isoformat()
        )


# ============================================================================
# ðŸš¨ MODULE 5: DEVIATION DETECTOR (ANOMALY DETECTION)
# ============================================================================

@dataclass
class Deviation:
    """Single deviation from baseline"""
    deviation_type: str  # "sudden_stress_spike", "prolonged_instability", etc.
    severity: float  # 0-1 (how unusual is this)
    description: str  # Human-readable explanation
    metric_name: str  # Which metric deviated
    current_value: float
    baseline_value: float
    threshold: float


class DeviationDetector:
    """
    ðŸš¨ DETECTS BEHAVIORAL ANOMALIES
    
    Compares current state vs baseline to identify:
        - Sudden stress spikes
        - Prolonged instability
        - Unusual masking behavior
        - Mood polarity shifts
    
    This is WHERE Phase 4 becomes intelligent:
        "Is this person acting unusual TODAY?"
    """
    
    def __init__(self, sensitivity: float = 1.0):
        """
        Initialize deviation detector
        
        Args:
            sensitivity: Detection sensitivity (0.5=lenient, 1.0=normal, 1.5=strict)
        """
        self.sensitivity = sensitivity
    
    def detect_deviations(
        self,
        current_metrics: SessionMetrics,
        baseline: BaselineProfile
    ) -> List[Deviation]:
        """
        ðŸŽ¯ CORE METHOD: Detect all deviations from baseline
        
        Args:
            current_metrics: Metrics from current session
            baseline: User's baseline profile
            
        Returns:
            List of detected deviations
        """
        deviations = []
        
        # Check each deviation type
        deviations.extend(self._check_stress_deviation(current_metrics, baseline))
        deviations.extend(self._check_stability_deviation(current_metrics, baseline))
        deviations.extend(self._check_masking_deviation(current_metrics, baseline))
        deviations.extend(self._check_mood_shift(current_metrics, baseline))
        deviations.extend(self._check_risk_escalation(current_metrics, baseline))
        
        # Sort by severity (most severe first)
        deviations.sort(key=lambda d: d.severity, reverse=True)
        
        return deviations
    
    def _check_stress_deviation(
        self,
        metrics: SessionMetrics,
        baseline: BaselineProfile
    ) -> List[Deviation]:
        """Detect sudden stress spikes"""
        deviations = []
        
        # Adjust threshold by sensitivity.
        # stress_threshold is an upper bound: divide by sensitivity so that
        # higher sensitivity (e.g. 1.5) lowers the threshold, triggering more easily.
        threshold = baseline.stress_threshold / self.sensitivity
        current_stress = metrics.stress_duration_ratio
        
        if current_stress > threshold:
            # Calculate severity (how far above threshold)
            severity = min((current_stress - threshold) / (1.0 - threshold), 1.0)
            
            deviations.append(Deviation(
                deviation_type="sudden_stress_spike",
                severity=severity,
                description=f"Stress level ({current_stress:.0%}) significantly higher than normal ({baseline.avg_stress_level:.0%})",
                metric_name="stress_duration_ratio",
                current_value=current_stress,
                baseline_value=baseline.avg_stress_level,
                threshold=threshold
            ))
        
        return deviations
    
    def _check_stability_deviation(
        self,
        metrics: SessionMetrics,
        baseline: BaselineProfile
    ) -> List[Deviation]:
        """Detect prolonged instability"""
        deviations = []
        
        threshold = baseline.stability_threshold * self.sensitivity
        current_stability = metrics.avg_stability
        
        if current_stability < threshold:
            # Calculate severity
            severity = min((threshold - current_stability) / threshold, 1.0)
            
            deviations.append(Deviation(
                deviation_type="prolonged_instability",
                severity=severity,
                description=f"Emotional stability ({current_stability:.2f}) significantly lower than normal ({baseline.avg_stability:.2f})",
                metric_name="stability_score",
                current_value=current_stability,
                baseline_value=baseline.avg_stability,
                threshold=threshold
            ))
        
        return deviations
    
    def _check_masking_deviation(
        self,
        metrics: SessionMetrics,
        baseline: BaselineProfile
    ) -> List[Deviation]:
        """Detect unusual masking behavior"""
        deviations = []
        
        # masking_threshold is an upper bound: divide by sensitivity so that
        # higher sensitivity (e.g. 1.5) lowers the trigger threshold.
        threshold = baseline.masking_threshold / self.sensitivity
        current_masking = metrics.masking_frequency
        
        if current_masking > threshold:
            # Calculate severity
            severity = min((current_masking - threshold) / max(threshold, 1.0), 1.0)
            
            deviations.append(Deviation(
                deviation_type="unusual_masking",
                severity=severity,
                description=f"Hiding emotions ({current_masking:.1f}/min) much more than usual ({baseline.avg_masking_frequency:.1f}/min)",
                metric_name="masking_frequency",
                current_value=current_masking,
                baseline_value=baseline.avg_masking_frequency,
                threshold=threshold
            ))
        
        return deviations
    
    def _check_mood_shift(
        self,
        metrics: SessionMetrics,
        baseline: BaselineProfile
    ) -> List[Deviation]:
        """Detect mood polarity shifts"""
        deviations = []
        
        # Check if dominant emotions shifted dramatically
        # (e.g., usually positive, now negative)
        
        current_positive = metrics.positive_emotion_ratio
        current_negative = metrics.negative_emotion_ratio
        
        # Significant shift = polarity reversed
        if current_positive < 0.2 and current_negative > 0.5:
            # Shifted to negative
            severity = current_negative
            
            deviations.append(Deviation(
                deviation_type="mood_polarity_shift",
                severity=severity,
                description=f"Mood shifted significantly negative ({current_negative:.0%} negative emotions)",
                metric_name="emotional_polarity",
                current_value=current_negative,
                baseline_value=0.0,  # No specific baseline for polarity
                threshold=0.5
            ))
        
        elif current_negative < 0.2 and current_positive > 0.6:
            # Shifted to positive (less concerning but noteworthy)
            severity = current_positive * 0.5  # Lower severity for positive shift
            
            deviations.append(Deviation(
                deviation_type="mood_polarity_shift",
                severity=severity,
                description=f"Mood shifted significantly positive ({current_positive:.0%} positive emotions)",
                metric_name="emotional_polarity",
                current_value=current_positive,
                baseline_value=0.0,
                threshold=0.6
            ))
        
        return deviations
    
    def _check_risk_escalation(
        self,
        metrics: SessionMetrics,
        baseline: BaselineProfile
    ) -> List[Deviation]:
        """Detect risk level escalations"""
        deviations = []
        
        risk_increase = metrics.avg_risk_level - baseline.normal_risk_level
        
        # Significant if risk increased by >1.0 level
        if risk_increase > 1.0:
            severity = min(risk_increase / 2.0, 1.0)
            
            deviations.append(Deviation(
                deviation_type="risk_escalation",
                severity=severity,
                description=f"Risk level ({metrics.avg_risk_level:.1f}) elevated above normal ({baseline.normal_risk_level:.1f})",
                metric_name="risk_level",
                current_value=metrics.avg_risk_level,
                baseline_value=baseline.normal_risk_level,
                threshold=baseline.normal_risk_level + 1.0
            ))
        
        return deviations


# ============================================================================
# ðŸŽ¯ MODULE 6: USER PSYCHOLOGICAL PROFILE (FINAL OUTPUT)
# ============================================================================

@dataclass
class UserPsychologicalProfile:
    """
    ðŸŽ¯ COMPLETE PHASE 4 OUTPUT
    
    This is what Phase 4 produces - the ULTIMATE psychological understanding:
        - Who this person is (Personality)
        - What's normal for them (Baseline)
        - What's happening now (Current State)
        - What's unusual (Deviations)
        - What risk level adjusted for this individual (Adjusted Risk)
    
    This goes BEYOND Phase 3's momentary assessment
    """
    
    # Long-term understanding
    personality: PersonalityProfile
    baseline: BaselineProfile
    
    # Current moment
    current_state: PsychologicalState
    current_session_metrics: SessionMetrics
    
    # Deviation analysis
    deviations: List[Deviation]
    deviation_summary: str  # Human-readable
    
    # Personalized risk (Phase 4's main contribution)
    phase3_risk: RiskLevel  # Original Phase 3 risk
    adjusted_risk: RiskLevel  # Personalized risk accounting for personality/baseline
    risk_adjustment_reason: str  # Why risk was adjusted
    
    # Metadata
    confidence: float  # Overall confidence in this profile
    timestamp: str


class Phase4CognitiveFusion:
    """
    ðŸ§  MAIN PHASE 4 ORCHESTRATOR
    
    Brings together all Phase 4 modules to produce final output:
        1. Maintains session memory
        2. Updates long-term memory
        3. Infers personality periodically
        4. Builds/updates baseline
        5. Detects deviations
        6. Adjusts risk based on personalization
    
    Usage:
        phase4 = Phase4CognitiveFusion(user_id="john_doe")
        profile = phase4.process_state(psychological_state)  # From Phase 3
        # profile now contains complete understanding
    """
    
    def __init__(
        self,
        user_id: str = "default_user",
        storage_dir: str = "data/memory",
        session_timeout_minutes: float = 5.0
    ):
        """
        Initialize Phase 4 system
        
        Args:
            user_id: Unique user identifier
            storage_dir: Directory for persistent storage
            session_timeout_minutes: Minutes before ending session
        """
        self.user_id = user_id
        self.session_timeout_minutes = session_timeout_minutes
        
        # Initialize all modules
        self.session_memory = SessionMemory()
        self.long_term_memory = LongTermMemory(user_id, storage_dir)
        
        # Phase 4 personality (trait inference from stats)
        self.personality_engine = PersonalityInferenceEngine()
        
        # Phase 5 personality (PSV - long-term behavioral patterns)
        # Lazy import avoids circular dependency (phase5 imports from this module)
        try:
            from inference.phase5_personality_engine import PersonalityEngine as _PE
            self.phase5_engine = _PE(
                user_id=user_id,
                storage_dir=storage_dir,
                learning_rate=0.03,  # Slow updates
                min_sessions_required=3  # Need 3+ sessions
            )
        except Exception as _e:
            print(f"[WARNING] Phase 5 not available - personality inference disabled: {_e}")
            self.phase5_engine = None
        
        self.baseline_builder = BaselineBuilder()
        self.deviation_detector = DeviationDetector()
        
        # Cached profiles (updated periodically)
        self.personality: Optional[PersonalityProfile] = None
        self.baseline: Optional[BaselineProfile] = None
        self.last_profile_update = 0.0
        
        # Initial profile building
        self._update_profiles()
    
    def process_state(self, state: PsychologicalState) -> UserPsychologicalProfile:
        """
        ðŸŽ¯ MAIN METHOD: Process Phase 3 state through Phase 4
        
        Args:
            state: PsychologicalState from Phase 3
            
        Returns:
            UserPsychologicalProfile with complete analysis
        """
        # Add state to session
        self.session_memory.add_state(state)
        
        # Check if session ended (timeout)
        if not self.session_memory.is_active(self.session_timeout_minutes):
            self._end_session()
        
        # Get current session metrics
        session_metrics = self.session_memory.calculate_metrics()
        
        # Update profiles if needed (every 24 hours or on first run)
        current_time = time.time()
        if current_time - self.last_profile_update > 86400 or self.personality is None:
            self._update_profiles()
        
        # Detect deviations from baseline
        deviations = []
        if self.baseline and self.baseline.confidence > 0.3:
            deviations = self.deviation_detector.detect_deviations(
                session_metrics,
                self.baseline
            )
        
        # Adjust risk based on personality + deviations
        adjusted_risk, adjustment_reason = self._adjust_risk(
            state.risk_level,
            deviations,
            self.personality,
            self.baseline
        )
        
        # Generate deviation summary
        deviation_summary = self._summarize_deviations(deviations)
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(
            state.confidence,
            self.personality.confidence if self.personality else 0.0,
            self.baseline.confidence if self.baseline else 0.0
        )
        
        # Build complete profile
        return UserPsychologicalProfile(
            personality=self.personality or self.personality_engine._default_profile(0),
            baseline=self.baseline or self.baseline_builder._default_baseline(0),
            current_state=state,
            current_session_metrics=session_metrics,
            deviations=deviations,
            deviation_summary=deviation_summary,
            phase3_risk=state.risk_level,
            adjusted_risk=adjusted_risk,
            risk_adjustment_reason=adjustment_reason,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
    
    def _end_session(self) -> None:
        """End current session and save to long-term memory"""
        if self.session_memory.get_frame_count() > 0:
            # Calculate final metrics
            final_metrics = self.session_memory.reset()
            
            # Save to long-term memory
            self.long_term_memory.add_session(final_metrics)
            
            # Update Phase 5 PSV (if available)
            if self.phase5_engine:
                # Add session to Phase 5 buffer
                self.phase5_engine.add_session(final_metrics)
                
                # Update PSV if enough data
                if self.phase5_engine.can_infer_personality():
                    # Get recent daily profiles for PSV update
                    recent_dates = sorted(self.long_term_memory.daily_profiles.keys())[-7:]  # Last 7 days
                    recent_profiles = [self.long_term_memory.daily_profiles[d] for d in recent_dates]
                    
                    # Update PSV and persist to disk
                    self.phase5_engine.update_psv(recent_profiles)
                    # Note: update_psv already calls _save_psv internally
            
            # Trigger profile update
            self._update_profiles()
    
    def _update_profiles(self) -> None:
        """Update personality and baseline profiles from long-term memory"""
        self.personality = self.personality_engine.infer_personality(self.long_term_memory)
        self.baseline = self.baseline_builder.build_baseline(self.long_term_memory)
        self.last_profile_update = time.time()
    
    def _adjust_risk(
        self,
        phase3_risk: RiskLevel,
        deviations: List[Deviation],
        personality: Optional[PersonalityProfile],
        baseline: Optional[BaselineProfile]
    ) -> Tuple[RiskLevel, str]:
        """
        ðŸ”¥ CORE PERSONALIZATION: Adjust risk based on individual context
        
        This is WHERE Phase 4 adds intelligence:
            - Same behavior = different risk for different people
            - Deviations matter more than absolute values
            - Personality traits inform interpretation
        
        Returns:
            (adjusted_risk_level, reason_for_adjustment)
        """
        risk_map = {RiskLevel.LOW: 0, RiskLevel.MODERATE: 1, RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}
        reverse_map = {0: RiskLevel.LOW, 1: RiskLevel.MODERATE, 2: RiskLevel.HIGH, 3: RiskLevel.CRITICAL}
        
        current_risk_value = risk_map[phase3_risk]
        adjustment = 0
        reasons = []
        
        # 1. Deviation-based adjustment (MOST IMPORTANT)
        if deviations:
            high_severity_deviations = [d for d in deviations if d.severity > 0.7]
            
            if high_severity_deviations:
                adjustment += 1
                reasons.append(f"{len(high_severity_deviations)} severe deviations detected")
            
            # Specific deviation types
            for dev in high_severity_deviations:
                if dev.deviation_type in ["sudden_stress_spike", "prolonged_instability"]:
                    adjustment += 1
                    reasons.append(f"{dev.deviation_type}")
                    break  # Don't double-count
        
        # 2. Personality-based adjustment
        if personality and personality.confidence > 0.5:
            # Low resilience + high volatility = higher risk
            if personality.resilience_score < 0.4 and personality.volatility_score > 0.6:
                adjustment += 1
                reasons.append("low resilience + high volatility personality")
            
            # High resilience = potential risk reduction (but not if deviations present)
            elif personality.resilience_score > 0.7 and not deviations:
                adjustment -= 1
                reasons.append("high resilience, no deviations")
        
        # 3. Baseline comparison
        if baseline and baseline.confidence > 0.5:
            # If usually stressed, current stress is less concerning
            if baseline.avg_stress_level > 0.5 and not deviations:
                adjustment -= 1
                reasons.append("stress is normal for this user")
        
        # Apply adjustment
        adjusted_risk_value = max(0, min(3, current_risk_value + adjustment))
        adjusted_risk = reverse_map[adjusted_risk_value]
        
        # Generate reason
        if adjusted_risk_value > current_risk_value:
            reason = f"â¬†ï¸ Risk elevated: {', '.join(reasons)}"
        elif adjusted_risk_value < current_risk_value:
            reason = f"â¬‡ï¸ Risk reduced: {', '.join(reasons)}"
        else:
            reason = "Risk unchanged (personalized analysis confirms Phase 3)"
        
        return adjusted_risk, reason
    
    def _summarize_deviations(self, deviations: List[Deviation]) -> str:
        """Generate human-readable deviation summary"""
        if not deviations:
            return "âœ… No unusual behavior detected. All metrics within normal range."
        
        summary_lines = [f"âš ï¸ {len(deviations)} behavioral deviation(s) detected:"]
        
        for i, dev in enumerate(deviations[:3], 1):  # Top 3
            summary_lines.append(f"  {i}. {dev.description} (severity: {dev.severity:.0%})")
        
        if len(deviations) > 3:
            summary_lines.append(f"  ... and {len(deviations) - 3} more")
        
        return "\n".join(summary_lines)
    
    def _calculate_overall_confidence(
        self,
        state_confidence: float,
        personality_confidence: float,
        baseline_confidence: float
    ) -> float:
        """Calculate weighted overall confidence in the profile"""
        # Weighted average: state=40%, personality=30%, baseline=30%
        confidence = (
            state_confidence * 0.4 +
            personality_confidence * 0.3 +
            baseline_confidence * 0.3
        )
        return float(np.clip(confidence, 0.0, 1.0))
    
    def get_full_report(self, profile: UserPsychologicalProfile) -> str:
        """
        Generate comprehensive human-readable report
        
        Args:
            profile: UserPsychologicalProfile to report on
            
        Returns:
            Formatted multi-section report
        """
        report = f"""
{'='*80}
ðŸ§  COMPLETE PSYCHOLOGICAL PROFILE
{'='*80}

ðŸ‘¤ USER: {self.user_id}
ðŸ“… TIMESTAMP: {profile.timestamp}
ðŸŽ¯ CONFIDENCE: {profile.confidence:.0%}

{'â”€'*80}
ðŸ“Š CURRENT STATE (PHASE 3)
{'â”€'*80}
Dominant Emotion: {profile.current_state.dominant_emotion}
Hidden Emotion: {profile.current_state.hidden_emotion or 'None detected'}
Mental State: {profile.current_state.mental_state.value}
Risk Level: {profile.current_state.risk_level.value}
Stability: {profile.current_state.stability_score:.2f}

{'â”€'*80}
ðŸŽ­ PERSONALITY TRAITS (LONG-TERM)
{'â”€'*80}
Emotional Reactivity: {profile.personality.emotional_reactivity:.2f} (0=stable, 1=reactive)
Stress Tolerance: {profile.personality.stress_tolerance:.2f} (0=sensitive, 1=tolerant)
Masking Tendency: {profile.personality.masking_tendency:.2f} (0=authentic, 1=masks)
Emotional Stability: {profile.personality.emotional_stability:.2f} (0=unstable, 1=stable)
Baseline Mood: {profile.personality.baseline_mood}

Volatility: {profile.personality.volatility_score:.2f} | Resilience: {profile.personality.resilience_score:.2f}
Data: {profile.personality.data_days} days | Confidence: {profile.personality.confidence:.0%}

{'â”€'*80}
ðŸ“ BASELINE (WHAT'S NORMAL)
{'â”€'*80}
Typical Stress: {profile.baseline.avg_stress_level:.0%}
Typical Stability: {profile.baseline.avg_stability:.2f}
Normal Risk: {profile.baseline.normal_risk_level:.1f}
Data: {profile.baseline.data_days} days | Confidence: {profile.baseline.confidence:.0%}

{'â”€'*80}
ðŸš¨ DEVIATION ANALYSIS
{'â”€'*80}
{profile.deviation_summary}

{'â”€'*80}
ðŸŽ¯ PERSONALIZED RISK ASSESSMENT
{'â”€'*80}
Phase 3 Risk: {profile.phase3_risk.value}
Adjusted Risk: {profile.adjusted_risk.value}
Adjustment: {profile.risk_adjustment_reason}

{'='*80}
        """.strip()
        
        return report
    
    def get_phase5_personality_summary(self) -> Optional[Dict]:
        """
        Get Phase 5 Personality State Vector (PSV) summary
        
        Returns:
            Dictionary with PSV traits and behavioral descriptor, or None if Phase 5 unavailable
        """
        if not self.phase5_engine:
            return None
        
        if not self.phase5_engine.can_infer_personality():
            return {
                "available": False,
                "reason": f"Need {self.phase5_engine.min_sessions_required} sessions minimum",
                "current_sessions": self.phase5_engine.psv.total_sessions_processed
            }
        
        return {
            "available": True,
            **self.phase5_engine.get_personality_summary()
        }
    
    def get_phase5_full_report(self) -> Optional[str]:
        """
        Generate Phase 5 personality report
        
        Returns:
            Formatted personality report or None if unavailable
        """
        if not self.phase5_engine:
            return "âš ï¸ Phase 5 Personality Engine not available"
        
        if not self.phase5_engine.can_infer_personality():
            return f"""
{'='*80}
â³ PHASE 5: PERSONALITY INFERENCE
{'='*80}

Status: Insufficient Data
Required Sessions: {self.phase5_engine.min_sessions_required}
Current Sessions: {self.phase5_engine.psv.total_sessions_processed}

Please complete more sessions to enable personality inference.
{'='*80}
            """.strip()
        
        from inference.phase5_personality_engine import generate_personality_report
        return generate_personality_report(self.phase5_engine)
