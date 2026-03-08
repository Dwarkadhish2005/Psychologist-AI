import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, CartesianGrid, Legend,
} from 'recharts'
import type { UserProfile, PSVData, DailyProfile } from '../types'
import PersonalityRadar from '../components/PersonalityRadar'

export default function Analytics() {
  const [searchParams] = useSearchParams()
  const [users, setUsers] = useState<UserProfile[]>([])
  const [selectedUser, setSelectedUser] = useState(searchParams.get('user') ?? '')
  const [history, setHistory] = useState<Record<string, DailyProfile>>({})
  const [psv, setPsv] = useState<PSVData | null>(null)
  const [loading, setLoading] = useState(false)
  const [psvError, setPsvError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/users/').then(r => r.json()).then(setUsers).catch(console.error)
  }, [])

  useEffect(() => {
    if (!selectedUser) return
    setLoading(true)
    setPsvError(null)

    Promise.all([
      fetch(`/api/analytics/${selectedUser}/history?days=30`)
        .then(r => r.ok ? r.json() : {})
        .catch(() => ({})),
      fetch(`/api/analytics/${selectedUser}/psv`)
        .then(r => r.ok ? r.json() : null)
        .catch(() => null),
    ]).then(([hist, psvData]) => {
      setHistory(hist)
      if (psvData) {
        setPsv(psvData)
      } else {
        setPsv(null)
        setPsvError('No personality data yet — complete more sessions to unlock PSV')
      }
    }).finally(() => setLoading(false))
  }, [selectedUser])

  // Transform daily history into chart-ready arrays
  const dailyData = Object.entries(history)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, d]) => ({
      date: date.slice(5), // MM-DD
      stress: Math.round((d.avg_stress_ratio ?? 0) * 100),
      risk: Math.round((d.avg_risk_level ?? 0) * 33), // scale 0-3 → 0-100
      positive: Math.round((d.positive_ratio ?? 0) * 100),
      negative: Math.round((d.negative_ratio ?? 0) * 100),
      sessions: d.total_sessions ?? 0,
    }))

  // PSV history charts
  const psvHistory = psv?.emotional_stability_history?.map((v, i) => ({
    session: `S${i + 1}`,
    stability:    Math.round(v * 100),
    sensitivity:  Math.round((psv.stress_sensitivity_history?.[i] ?? 0) * 100),
    recovery:     Math.round((psv.recovery_speed_history?.[i] ?? 0) * 100),
    positivity:   Math.round((psv.positivity_bias_history?.[i] ?? 0) * 100),
  })) ?? []

  const tooltipStyle = {
    contentStyle: { background: '#1e293b', border: '1px solid #334155', borderRadius: 8 },
    labelStyle: { color: '#94a3b8' },
  }

  return (
    <div className="p-8 space-y-8">
      {/* Header + user select */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Analytics</h1>
          <p className="text-slate-400 text-sm mt-1">Historical mood, risk, and personality trends</p>
        </div>
        <select
          className="bg-slate-800 border border-slate-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          value={selectedUser}
          onChange={e => setSelectedUser(e.target.value)}
        >
          <option value="">— Select user —</option>
          {users.map(u => (
            <option key={u.user_id} value={u.user_id}>{u.name}</option>
          ))}
        </select>
      </div>

      {!selectedUser && (
        <div className="rounded-xl bg-slate-800/40 border border-slate-700 p-12 text-center text-slate-500 text-sm">
          Select a user above to view their analytics
        </div>
      )}

      {selectedUser && loading && (
        <p className="text-slate-500 text-sm">Loading data…</p>
      )}

      {selectedUser && !loading && (
        <div className="space-y-8">
          {/* Stress & Risk over time */}
          {dailyData.length > 0 && (
            <ChartCard title="Stress & Risk Level (last 30 days)">
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={dailyData}>
                  <CartesianGrid stroke="#1e293b" />
                  <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 11 }} domain={[0, 100]} unit="%" />
                  <Tooltip {...tooltipStyle} formatter={(v: number) => [`${v}%`]} />
                  <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
                  <Line type="monotone" dataKey="stress"   name="Stress"    stroke="#f97316" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="risk"     name="Risk"      stroke="#ef4444" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="positive" name="Positive"  stroke="#22c55e" strokeWidth={2} dot={false} strokeDasharray="4 2" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          )}

          {/* Sessions per day */}
          {dailyData.length > 0 && (
            <ChartCard title="Sessions per Day">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={dailyData}>
                  <CartesianGrid stroke="#1e293b" />
                  <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 11 }} allowDecimals={false} />
                  <Tooltip {...tooltipStyle} />
                  <Bar dataKey="sessions" name="Sessions" fill="#6366f1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          )}

          {/* Personality section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Radar */}
            <ChartCard title={`Personality State Vector ${psv ? `(${psv.total_sessions_processed} sessions)` : ''}`}>
              {psv ? (
                <PersonalityRadar psv={psv} />
              ) : (
                <div className="flex items-center justify-center h-40 text-slate-500 text-sm text-center px-4">
                  {psvError}
                </div>
              )}
            </ChartCard>

            {/* PSV trait bars */}
            {psv && (
              <ChartCard title="Personality Traits">
                <div className="space-y-3 py-2">
                  <TraitBar label="Emotional Stability"  value={psv.emotional_stability} color="bg-indigo-500" />
                  <TraitBar label="Stress Sensitivity"   value={psv.stress_sensitivity}  color="bg-orange-500" invert />
                  <TraitBar label="Recovery Speed"       value={psv.recovery_speed}      color="bg-emerald-500" />
                  <TraitBar label="Positivity Bias"      value={psv.positivity_bias}     color="bg-green-500" />
                  <TraitBar label="Consistency"          value={1 - psv.volatility}      color="bg-sky-500" />
                </div>
                <p className="text-xs text-slate-500 mt-3">
                  Data confidence: {Math.round(psv.confidence * 100)}%
                </p>
              </ChartCard>
            )}
          </div>

          {/* PSV history trends */}
          {psvHistory.length > 1 && (
            <ChartCard title="Personality Trait Evolution">
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={psvHistory}>
                  <CartesianGrid stroke="#1e293b" />
                  <XAxis dataKey="session" tick={{ fill: '#64748b', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 11 }} domain={[0, 100]} unit="%" />
                  <Tooltip {...tooltipStyle} formatter={(v: number) => [`${v}%`]} />
                  <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
                  <Line type="monotone" dataKey="stability"   name="Stability"   stroke="#6366f1" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="recovery"    name="Recovery"    stroke="#22c55e" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="positivity"  name="Positivity"  stroke="#eab308" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="sensitivity" name="Sensitivity" stroke="#f97316" strokeWidth={2} dot={false} strokeDasharray="4 2" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          )}

          {dailyData.length === 0 && !loading && (
            <div className="rounded-xl bg-slate-800/40 border border-slate-700 p-8 text-center text-slate-500 text-sm">
              No session history yet for this user. Start a live session to generate data.
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5">
      <h3 className="text-sm font-semibold text-white mb-4">{title}</h3>
      {children}
    </div>
  )
}

function TraitBar({
  label, value, color, invert = false,
}: { label: string; value: number; color: string; invert?: boolean }) {
  const display = invert ? 1 - value : value
  return (
    <div>
      <div className="flex justify-between text-xs text-slate-400 mb-1">
        <span>{label}</span>
        <span>{Math.round(display * 100)}%</span>
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${Math.round(display * 100)}%` }}
        />
      </div>
    </div>
  )
}
