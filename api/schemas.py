"""
Pydantic schemas for Psychologist AI REST API
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class UserCreate(BaseModel):
    name: str


class UserResponse(BaseModel):
    user_id: str
    name: str
    created_at: str
    last_active: str
    total_sessions: int


class SessionStartRequest(BaseModel):
    user_id: str


class SessionStatus(BaseModel):
    is_running: bool
    active_user_id: Optional[str] = None


class PersonalityData(BaseModel):
    emotional_reactivity: float
    stress_tolerance: float
    emotional_stability: float
    baseline_mood: str
    confidence: float
    data_days: int


class DeviationAlert(BaseModel):
    type: str
    severity: float
    description: str


class EmotionState(BaseModel):
    """Real-time psychological state sent over WebSocket"""
    face_emotion: str = "neutral"
    hidden_emotion: Optional[str] = None
    mental_state: str = "calm"
    confidence: float = 0.5
    risk_level: str = "low"
    adjusted_risk: Optional[str] = None
    stability_score: float = 0.5
    explanations: List[str] = []
    timestamp: float = 0.0
    voice_emotion: Optional[str] = None
    voice_confidence: Optional[float] = None
    stress_level: Optional[str] = None
    personality: Optional[PersonalityData] = None
    deviations: Optional[List[DeviationAlert]] = None
    risk_adjustment_reason: Optional[str] = None
    fps: Optional[float] = None


class PSVData(BaseModel):
    emotional_stability: float
    stress_sensitivity: float
    recovery_speed: float
    positivity_bias: float
    volatility: float
    confidence: float
    total_sessions_processed: int
    last_updated: Optional[str] = None
    emotional_stability_history: Optional[List[float]] = None
    stress_sensitivity_history: Optional[List[float]] = None
    recovery_speed_history: Optional[List[float]] = None
    positivity_bias_history: Optional[List[float]] = None
    volatility_history: Optional[List[float]] = None
