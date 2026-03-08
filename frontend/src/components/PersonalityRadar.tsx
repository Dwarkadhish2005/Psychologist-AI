import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'
import type { PSVData } from '../types'

interface Props {
  psv: PSVData
}

export default function PersonalityRadar({ psv }: Props) {
  const data = [
    { trait: 'Stability',     value: Math.round(psv.emotional_stability  * 100) },
    { trait: 'Positivity',    value: Math.round(psv.positivity_bias       * 100) },
    { trait: 'Recovery',      value: Math.round(psv.recovery_speed        * 100) },
    { trait: 'Tolerance',     value: Math.round((1 - psv.stress_sensitivity) * 100) },
    { trait: 'Consistency',   value: Math.round((1 - psv.volatility)       * 100) },
  ]

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-400">
          Confidence: {Math.round(psv.confidence * 100)}% · {psv.total_sessions_processed} sessions
        </span>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <RadarChart data={data} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
          <PolarGrid stroke="#334155" />
          <PolarAngleAxis
            dataKey="trait"
            tick={{ fill: '#94a3b8', fontSize: 12 }}
          />
          <Radar
            name="PSV"
            dataKey="value"
            stroke="#6366f1"
            fill="#6366f1"
            fillOpacity={0.35}
          />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            formatter={(v: number) => [`${v}%`, 'Score']}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}
