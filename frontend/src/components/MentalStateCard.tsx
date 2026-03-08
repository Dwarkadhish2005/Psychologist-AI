import type { EmotionState } from '../types'
import RiskBadge from './RiskBadge'

interface Props {
  state: EmotionState | null
}

const EMOTION_EMOJI: Record<string, string> = {
  happy:    '😊', sad:     '😢', angry:   '😠',
  fear:     '😨', neutral: '😐', disgust: '🤢',
  surprise: '😲',
}

const MENTAL_STATE_COLORS: Record<string, string> = {
  calm:                'text-green-400',
  joyful:              'text-green-300',
  stable_positive:     'text-emerald-400',
  stressed:            'text-yellow-400',
  happy_under_stress:  'text-yellow-300',
  anxious:             'text-orange-400',
  overwhelmed:         'text-orange-500',
  emotionally_masked:  'text-purple-400',
  emotionally_flat:    'text-slate-400',
  fearful:             'text-red-400',
  angry_stressed:      'text-red-500',
  sad_depressed:       'text-blue-400',
  confused:            'text-cyan-400',
  stable_negative:     'text-slate-500',
  emotionally_unstable:'text-pink-400',
}

export default function MentalStateCard({ state }: Props) {
  if (!state || state.status === 'waiting') {
    return (
      <div className="rounded-xl bg-slate-800/60 p-5 flex items-center justify-center h-full min-h-[140px] border border-slate-700">
        <span className="text-slate-500 text-sm">Waiting for session to start…</span>
      </div>
    )
  }

  const displayRisk = state.adjusted_risk ?? state.risk_level
  const emoji = EMOTION_EMOJI[state.face_emotion] ?? '🧠'
  const mentalColor = MENTAL_STATE_COLORS[state.mental_state] ?? 'text-white'
  const mentalLabel = state.mental_state.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

  return (
    <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5 space-y-3">
      {/* Top row */}
      <div className="flex items-start justify-between">
        <div className="flex flex-col gap-1">
          <span className="text-3xl">{emoji}</span>
          <span className="text-lg font-bold capitalize text-white">{state.face_emotion}</span>
          {state.hidden_emotion && (
            <span className="text-xs text-purple-400">
              Hidden: {state.hidden_emotion}
            </span>
          )}
        </div>
        <RiskBadge risk={displayRisk} size="md" />
      </div>

      {/* Mental state */}
      <div>
        <span className="text-xs text-slate-400 uppercase tracking-wider">Mental State</span>
        <p className={`text-base font-semibold ${mentalColor}`}>{mentalLabel}</p>
      </div>

      {/* Meters */}
      <div className="space-y-2">
        <Meter label="Confidence" value={state.confidence} color="bg-indigo-500" />
        <Meter label="Stability"  value={state.stability_score} color="bg-emerald-500" />
      </div>

      {/* Voice */}
      {state.voice_emotion && (
        <div className="flex gap-3 text-xs text-slate-400">
          <span>Voice: <span className="text-white capitalize">{state.voice_emotion}</span></span>
          {state.stress_level && (
            <span>Stress: <span className="text-orange-300 capitalize">{state.stress_level}</span></span>
          )}
        </div>
      )}

      {/* FPS */}
      {state.fps !== undefined && (
        <div className="text-xs text-slate-600">FPS: {state.fps}</div>
      )}
    </div>
  )
}

function Meter({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-xs text-slate-400 mb-0.5">
        <span>{label}</span>
        <span>{Math.round(value * 100)}%</span>
      </div>
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${color}`}
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
    </div>
  )
}
