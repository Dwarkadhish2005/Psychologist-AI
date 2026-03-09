import { useEffect, useState } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, CartesianGrid, Legend,
} from 'recharts'
import {
  Calendar, ChevronDown, ChevronUp, Clock, TrendingUp,
  Activity, Brain, AlertTriangle, Smile, Download, FileText,
} from 'lucide-react'
import type { UserProfile, DailyProfile } from '../types'
import RiskBadge from '../components/RiskBadge'

// ─── helpers ────────────────────────────────────────────────────────────────

function riskLabel(val: number): 'low' | 'moderate' | 'high' | 'critical' {
  if (val < 1) return 'low'
  if (val < 2) return 'moderate'
  if (val < 3) return 'high'
  return 'critical'
}

function fmtDuration(minutes: number) {
  const h = Math.floor(minutes / 60)
  const m = Math.round(minutes % 60)
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

function pct(val: number) {
  return `${Math.round(val * 100)}%`
}

const TOOLTIP_STYLE = {
  contentStyle: { background: '#1e293b', border: '1px solid #334155', borderRadius: 8 },
  labelStyle: { color: '#94a3b8' },
}

// ─── day detail row ──────────────────────────────────────────────────────────

function DayRow({ date, profile }: { date: string; profile: DailyProfile }) {
  const [open, setOpen] = useState(false)

  const topState = profile.dominant_mental_states?.[0]?.[0] ?? '—'
  const topEmotion = profile.dominant_emotions?.[0]?.[0] ?? '—'
  const riskStr = riskLabel(profile.avg_risk_level ?? 0)

  return (
    <div className="rounded-xl bg-slate-800/60 border border-slate-700 overflow-hidden">
      {/* Summary row */}
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-4 px-5 py-4 text-left hover:bg-slate-700/30 transition-colors"
      >
        {/* Date */}
        <div className="w-24 flex-shrink-0">
          <p className="text-sm font-semibold text-white">{date}</p>
          <p className="text-xs text-slate-500">{profile.total_sessions} session{profile.total_sessions !== 1 ? 's' : ''}</p>
        </div>

        {/* Duration */}
        <div className="w-20 flex-shrink-0 text-xs text-slate-400 hidden sm:block">
          <Clock size={12} className="inline mr-1" />
          {fmtDuration(profile.total_duration_minutes ?? 0)}
        </div>

        {/* Top mental state */}
        <div className="flex-1 min-w-0">
          <p className="text-xs text-slate-500">Mental state</p>
          <p className="text-sm text-white capitalize truncate">{topState.replace(/_/g, ' ')}</p>
        </div>

        {/* Stress bar */}
        <div className="w-28 flex-shrink-0 hidden md:block">
          <p className="text-xs text-slate-500 mb-1">Stress</p>
          <div className="h-2 rounded-full bg-slate-700 overflow-hidden">
            <div
              className="h-full rounded-full bg-amber-500"
              style={{ width: `${Math.round((profile.avg_stress_ratio ?? 0) * 100)}%` }}
            />
          </div>
          <p className="text-xs text-slate-400 mt-0.5">{pct(profile.avg_stress_ratio ?? 0)}</p>
        </div>

        {/* Risk */}
        <div className="flex-shrink-0 mx-2">
          <RiskBadge risk={riskStr} size="sm" />
        </div>

        {/* Expand toggle */}
        <div className="text-slate-500 flex-shrink-0">
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </button>

      {/* Expanded detail */}
      {open && (
        <div className="border-t border-slate-700 px-5 py-5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Emotional breakdown */}
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1">
              <Smile size={12} /> Emotion split
            </p>
            <MiniBar label="Positive" value={profile.positive_ratio ?? 0} color="bg-emerald-500" />
            <MiniBar label="Neutral"  value={profile.neutral_ratio  ?? 0} color="bg-slate-500" />
            <MiniBar label="Negative" value={profile.negative_ratio ?? 0} color="bg-rose-500" />
          </div>

          {/* Dominant mental states */}
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1">
              <Brain size={12} /> Mental states
            </p>
            <ul className="space-y-1">
              {(profile.dominant_mental_states ?? []).slice(0, 4).map(([state, freq]) => (
                <li key={state} className="flex items-center justify-between text-xs">
                  <span className="text-slate-300 capitalize">{state.replace(/_/g, ' ')}</span>
                  <span className="text-slate-500">{pct(freq)}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Stability & confidence */}
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1">
              <Activity size={12} /> Stability
            </p>
            <GaugeItem label="Avg Confidence" value={profile.avg_confidence ?? 0} color="text-sky-400" />
            <GaugeItem label="Avg Stability"  value={profile.avg_stability  ?? 0} color="text-indigo-400" />
            <GaugeItem label="Emotional Volatility" value={profile.emotional_volatility ?? 0} color="text-amber-400" invert />
          </div>

          {/* Risk & masking */}
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1">
              <AlertTriangle size={12} /> Risk & masking
            </p>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-400">High-risk time</span>
                <span className="text-white">{pct(profile.high_risk_duration_ratio ?? 0)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Risk escalations</span>
                <span className="text-white">{profile.total_risk_escalations ?? 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Masking events</span>
                <span className="text-white">{profile.total_masking_events ?? 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Masking time</span>
                <span className="text-white">{pct(profile.masking_duration_ratio ?? 0)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Top emotion</span>
                <span className="text-white capitalize">{topEmotion}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function MiniBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="mb-1.5">
      <div className="flex justify-between text-xs mb-0.5">
        <span className="text-slate-400">{label}</span>
        <span className="text-slate-300">{pct(value)}</span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-700 overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
    </div>
  )
}

function GaugeItem({ label, value, color, invert = false }: { label: string; value: number; color: string; invert?: boolean }) {
  const display = invert ? (1 - value) : value
  return (
    <div className="flex items-center justify-between text-xs mb-1">
      <span className="text-slate-400">{label}</span>
      <span className={`${color} font-semibold`}>{Math.round(display * 100)}%</span>
    </div>
  )
}

// ─── main page ───────────────────────────────────────────────────────────────

export default function SessionHistory() {
  const [searchParams] = useSearchParams()
  const [users, setUsers] = useState<UserProfile[]>([])
  const [selectedUser, setSelectedUser] = useState(searchParams.get('user') ?? '')
  const [history, setHistory] = useState<Record<string, DailyProfile>>({})
  const [days, setDays] = useState(30)
  const [loading, setLoading] = useState(false)

  // Load users
  useEffect(() => {
    fetch('/api/users/').then(r => r.json()).then(setUsers).catch(console.error)
  }, [])

  // Load history whenever user or days changes
  useEffect(() => {
    if (!selectedUser) return
    setLoading(true)
    fetch(`/api/analytics/${selectedUser}/history?days=${days}`)
      .then(r => r.ok ? r.json() : {})
      .then(setHistory)
      .catch(() => setHistory({}))
      .finally(() => setLoading(false))
  }, [selectedUser, days])

  // Sorted entries (newest first for table, oldest first for chart)
  const sortedEntries = Object.entries(history).sort(([a], [b]) => b.localeCompare(a))
  const chartData = [...sortedEntries].reverse().map(([date, d]) => ({
    date: date.slice(5),
    stress:   Math.round((d.avg_stress_ratio   ?? 0) * 100),
    positive: Math.round((d.positive_ratio     ?? 0) * 100),
    negative: Math.round((d.negative_ratio     ?? 0) * 100),
    risk:     Math.round((d.avg_risk_level      ?? 0) * 33),
    sessions: d.total_sessions ?? 0,
    duration: Math.round(d.total_duration_minutes ?? 0),
  }))

  // Aggregate totals
  const totalSessions = sortedEntries.reduce((a, [, d]) => a + (d.total_sessions ?? 0), 0)
  const totalMinutes  = sortedEntries.reduce((a, [, d]) => a + (d.total_duration_minutes ?? 0), 0)
  const avgStress     = sortedEntries.length
    ? sortedEntries.reduce((a, [, d]) => a + (d.avg_stress_ratio ?? 0), 0) / sortedEntries.length
    : 0

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Calendar size={22} className="text-indigo-400" />
            Session History
          </h1>
          <p className="text-slate-400 text-sm mt-1">Daily session breakdown — stress, risk, and emotion trends</p>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          {/* User select */}
          <select
            value={selectedUser}
            onChange={e => setSelectedUser(e.target.value)}
            className="bg-slate-800 border border-slate-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          >
            <option value="">— Select user —</option>
            {users.map(u => (
              <option key={u.user_id} value={u.user_id}>{u.name}</option>
            ))}
          </select>

          {/* Days filter */}
          <select
            value={days}
            onChange={e => setDays(Number(e.target.value))}
            className="bg-slate-800 border border-slate-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          >
            <option value={7}>Last 7 days</option>
            <option value={14}>Last 14 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
          </select>

          {selectedUser && (
            <Link
              to={`/analytics?user=${selectedUser}`}
              className="flex items-center gap-1.5 text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 px-3 py-2 rounded-lg transition-colors"
            >
              <TrendingUp size={13} />
              Personality
            </Link>
          )}
          {selectedUser && (
            <>
              <a
                href={`/api/analytics/${selectedUser}/export/csv`}
                download
                className="flex items-center gap-1.5 text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 px-3 py-2 rounded-lg transition-colors"
              >
                <Download size={13} /> CSV
              </a>
              <a
                href={`/api/analytics/${selectedUser}/report`}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-1.5 text-xs bg-indigo-700 hover:bg-indigo-600 text-white px-3 py-2 rounded-lg transition-colors"
              >
                <FileText size={13} /> Report
              </a>
            </>
          )}
        </div>
      </div>

      {!selectedUser ? (
        <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-12 text-center">
          <Calendar size={40} className="text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400">Select a user to view their session history</p>
        </div>
      ) : loading ? (
        <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-12 text-center">
          <p className="text-slate-500 text-sm">Loading history…</p>
        </div>
      ) : sortedEntries.length === 0 ? (
        <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-12 text-center">
          <p className="text-slate-400">No sessions recorded in this period.</p>
          <Link to={`/live?user=${selectedUser}`} className="mt-3 inline-block text-indigo-400 text-sm hover:underline">
            Start a live session →
          </Link>
        </div>
      ) : (
        <>
          {/* Summary stats */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <MiniStat label="Days with data" value={sortedEntries.length} unit="" color="text-indigo-400" />
            <MiniStat label="Total sessions" value={totalSessions} unit="" color="text-emerald-400" />
            <MiniStat label="Total time" value={Math.round(totalMinutes)} unit=" min" color="text-sky-400" />
            <MiniStat label="Avg stress" value={Math.round(avgStress * 100)} unit="%" color="text-amber-400" />
          </div>

          {/* Trend charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Stress & risk */}
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5">
              <p className="text-sm font-semibold text-white mb-4">Stress & Risk Trend</p>
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="gStress" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#f59e0b" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="gRisk" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#ef4444" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} unit="%" domain={[0, 100]} />
                  <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => [`${v}%`]} />
                  <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
                  <Area type="monotone" dataKey="stress" name="Stress" stroke="#f59e0b" fill="url(#gStress)" strokeWidth={2} dot={false} />
                  <Area type="monotone" dataKey="risk"   name="Risk"   stroke="#ef4444" fill="url(#gRisk)"   strokeWidth={2} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Sessions & duration */}
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5">
              <p className="text-sm font-semibold text-white mb-4">Sessions &amp; Duration per Day</p>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis yAxisId="left"  tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} unit="m" />
                  <Tooltip {...TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
                  <Bar yAxisId="left"  dataKey="sessions" name="Sessions" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  <Bar yAxisId="right" dataKey="duration" name="Duration (min)" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Emotion polarity */}
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5 lg:col-span-2">
              <p className="text-sm font-semibold text-white mb-4">Daily Emotion Polarity</p>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={chartData} stackOffset="expand" barSize={18}>
                  <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => [`${Math.round(v)}%`]} />
                  <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
                  <Bar dataKey="positive" name="Positive" stackId="a" fill="#10b981" />
                  <Bar dataKey="negative" name="Negative" stackId="a" fill="#ef4444" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Day-by-day table */}
          <div>
            <h2 className="text-lg font-semibold text-white mb-4">Daily Breakdown</h2>
            <div className="space-y-3">
              {sortedEntries.map(([date, profile]) => (
                <DayRow key={date} date={date} profile={profile} />
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function MiniStat({ label, value, unit, color }: { label: string; value: number; unit: string; color: string }) {
  return (
    <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-4">
      <p className={`text-xl font-bold ${color}`}>{value}{unit}</p>
      <p className="text-xs text-slate-400 mt-0.5">{label}</p>
    </div>
  )
}
