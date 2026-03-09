// Shared TypeScript types mirroring the Pydantic schemas

export interface UserProfile {
  user_id: string
  name: string
  created_at: string
  last_active: string
  total_sessions: number
}

export interface PersonalityData {
  emotional_reactivity: number
  stress_tolerance: number
  emotional_stability: number
  baseline_mood: string
  confidence: number
  data_days: number
}

export interface DeviationAlert {
  type: string
  severity: number
  description: string
}

export interface EmotionState {
  face_emotion: string
  hidden_emotion?: string
  mental_state: string
  confidence: number
  risk_level: string
  adjusted_risk?: string
  stability_score: number
  explanations: string[]
  timestamp: number
  fps?: number
  voice_emotion?: string
  voice_confidence?: number
  stress_level?: string
  personality?: PersonalityData
  deviations?: DeviationAlert[]
  risk_adjustment_reason?: string
  // status heartbeat
  status?: string
}

export interface PSVData {
  emotional_stability: number
  stress_sensitivity: number
  recovery_speed: number
  positivity_bias: number
  volatility: number
  confidence: number
  total_sessions_processed: number
  last_updated?: string
  emotional_stability_history?: number[]
  stress_sensitivity_history?: number[]
  recovery_speed_history?: number[]
  positivity_bias_history?: number[]
  volatility_history?: number[]
}

export interface DailyProfile {
  date: string
  total_sessions: number
  total_duration_minutes: number
  avg_stress_ratio: number
  avg_high_stress_ratio: number
  avg_stress_intensity: number
  dominant_mental_states: [string, number][]
  dominant_emotions?: [string, number][]
  avg_confidence: number
  avg_stability: number
  confidence_trend?: string
  stability_trend?: string
  avg_risk_level: number
  high_risk_duration_ratio: number
  total_risk_escalations: number
  total_masking_events?: number
  avg_masking_frequency?: number
  masking_duration_ratio?: number
  positive_ratio: number
  negative_ratio: number
  neutral_ratio: number
  state_switches_per_minute?: number
  emotional_volatility: number
}

export type RiskLevel = 'low' | 'moderate' | 'high' | 'critical'

export interface SessionSummary {
  session_id?: string
  session_start: number
  session_end?: number
  session_duration: number          // seconds
  dominant_mental_states: [string, number][]
  dominant_emotions?: [string, number][]
  stress_duration_ratio: number
  high_stress_duration_ratio: number
  avg_stress_intensity: number
  avg_confidence: number
  avg_stability: number
  avg_risk_level: number
  high_risk_duration_ratio: number
  risk_escalations: number
  total_masking_events: number
  masking_duration_ratio: number
  positive_emotion_ratio: number
  negative_emotion_ratio: number
  neutral_emotion_ratio: number
  mental_state_switches: number
  confidence_variance: number
}
