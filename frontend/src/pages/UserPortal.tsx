import { useEffect, useState } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import {
  UserCircle, Calendar, Clock, TrendingUp, Activity,
  Brain, ChevronDown, ChevronUp, Monitor, BarChart2, FileText,
} from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts'
import type { DailyProfile, PSVData } from '../types'
import RiskBadge from '../components/RiskBadge'
import PersonalityRadar from '../components/PersonalityRadar'
import { useAuth } from '../context/AuthContext'

type UserTab = 'sessions' | 'analytics' | 'reports'

function fmtDuration(minutes: number) {
  const h = Math.floor(minutes / 60)
  const m = Math.round(minutes % 60)
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

function riskLabel(val: number): 'low' | 'moderate' | 'high' | 'critical' {
  if (val < 1) return 'low'
  if (val < 2) return 'moderate'
  if (val < 3) return 'high'
  return 'critical'
}

function pct(v: number) { return `${Math.round(v * 100)}%` }

const TOOLTIP = {
  contentStyle: { background: '#1e293b', border: '1px solid #334155', borderRadius: 8 },
  labelStyle: { color: '#94a3b8' },
}

const TABS: { id: UserTab; label: string; icon: React.ElementType }[] = [
  { id: 'sessions',  label: 'Own Sessions',  icon: Monitor   },
  { id: 'analytics', label: 'Own Analytics', icon: BarChart2  },
  { id: 'reports',   label: 'Daily Reports', icon: FileText   },
]

export default function UserPortal() {
  const [searchParams, setSearchParams] = useSearchParams()
  const { user: authUser } = useAuth()
  const selectedUser = authUser?.userId ?? ''
  const [history, setHistory] = useState<Record<string, DailyProfile>>({})
  const [psv, setPsv] = useState<PSVData | null>(null)
  const [loading, setLoading] = useState(false)
  const [expandedDay, setExpandedDay] = useState<string | null>(null)

  const activeTab: UserTab = (searchParams.get('tab') as UserTab) ?? 'sessions'

  function setTab(t: UserTab) {
    const next = new URLSearchParams(searchParams)
    next.set('tab', t)
    setSearchParams(next)
  }

  useEffect(() => {
    if (!selectedUser) return
    setLoading(true)
    Promise.all([
      fetch(`/api/analytics/${selectedUser}/history?days=30`).then(r => r.ok ? r.json() : {}).catch(() => ({})),
      fetch(`/api/analytics/${selectedUser}/psv`).then(r => r.ok ? r.json() : null).catch(() => null),
    ]).then(([hist, psvData]) => {
      setHistory(hist)
      setPsv(psvData)
    }).finally(() => setLoading(false))
  }, [selectedUser])
  const sortedEntries = Object.entries(history).sort(([a], [b]) => b.localeCompare(a))

  const chartData = [...sortedEntries].reverse().map(([date, d]) => ({
    date:     date.slice(5),
    stress:   Math.round((d.avg_stress_ratio ?? 0) * 100),
    positive: Math.round((d.positive_ratio   ?? 0) * 100),
    risk:     Math.round((d.avg_risk_level    ?? 0) * 33),
  }))

  const totalSessions = sortedEntries.reduce((a, [, d]) => a + (d.total_sessions ?? 0), 0)
  const totalMinutes  = sortedEntries.reduce((a, [, d]) => a + (d.total_duration_minutes ?? 0), 0)
  const avgStress     = sortedEntries.length
    ? sortedEntries.reduce((a, [, d]) => a + (d.avg_stress_ratio ?? 0), 0) / sortedEntries.length
    : 0

  const today = new Date().toISOString().slice(0, 10)
  const todayProfile = history[today]

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-emerald-600/20 flex items-center justify-center">
          <UserCircle size={20} className="text-emerald-400" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-white">
            Hello, {authUser?.name ?? 'User'} 👋
          </h1>
          <p className="text-slate-400 text-sm">Your sessions, wellness trends, and daily reports</p>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 bg-slate-800/60 border border-slate-700 rounded-xl p-1 w-fit">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              activeTab === id ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      {/* Empty state — no userId in auth (should not normally happen) */}
      {!selectedUser && (
        <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-14 text-center">
          <UserCircle size={48} className="text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400">Your session has expired. Please sign in again.</p>
        </div>
      )}

      {selectedUser && loading && (
        <p className="text-slate-500 text-sm">Loading your data…</p>
      )}

      {/* ── TAB: OWN SESSIONS ─────────────────────────────────── */}
      {selectedUser && !loading && activeTab === 'sessions' && (
        <div className="space-y-5">
          {/* Stats overview */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <StatBox label="Total Sessions"    value={`${totalSessions}`}       icon={<Calendar   size={16} />} color="text-indigo-400" />
            <StatBox label="Total Time"        value={fmtDuration(totalMinutes)} icon={<Clock      size={16} />} color="text-sky-400"    />
            <StatBox label="Avg Stress (30d)"  value={pct(avgStress)}            icon={<Activity   size={16} />} color="text-amber-400"  />
            <StatBox label="Data Points"       value={psv ? `${psv.total_sessions_processed}` : '—'} icon={<Brain size={16} />} color="text-purple-400" />
          </div>

          {/* Session list */}
          {sortedEntries.length > 0 ? (
            <div className="space-y-2">
              {sortedEntries.map(([date, d]) => {
                const open = expandedDay === date
                return (
                  <div key={date} className="rounded-xl bg-slate-800/60 border border-slate-700 overflow-hidden">
                    <button
                      onClick={() => setExpandedDay(open ? null : date)}
                      className="w-full flex items-center gap-4 px-5 py-3 text-left hover:bg-slate-700/30 transition-colors"
                    >
                      <div className="flex-1">
                        <p className="text-white text-sm font-semibold">{date}</p>
                        <p className="text-xs text-slate-500">{d.total_sessions} session{d.total_sessions !== 1 ? 's' : ''} · {fmtDuration(d.total_duration_minutes ?? 0)}</p>
                      </div>
                      <div className="flex items-center gap-3 text-xs">
                        <span className="text-slate-400 hidden sm:block">Stress: <span className="text-amber-400 font-semibold">{pct(d.avg_stress_ratio ?? 0)}</span></span>
                        <RiskBadge risk={riskLabel(d.avg_risk_level ?? 0)} size="sm" />
                      </div>
                      {open ? <ChevronUp size={15} className="text-slate-500 flex-shrink-0" /> : <ChevronDown size={15} className="text-slate-500 flex-shrink-0" />}
                    </button>

                    {open && (
                      <div className="border-t border-slate-700 px-5 py-4 grid grid-cols-1 sm:grid-cols-3 gap-5">
                        <div>
                          <p className="text-xs text-slate-400 uppercase tracking-wider mb-2">Emotions</p>
                          {[
                            { label: 'Positive', value: d.positive_ratio ?? 0, color: 'bg-emerald-500' },
                            { label: 'Neutral',  value: d.neutral_ratio  ?? 0, color: 'bg-slate-500'   },
                            { label: 'Negative', value: d.negative_ratio ?? 0, color: 'bg-rose-500'    },
                          ].map(({ label, value, color }) => (
                            <div key={label} className="mb-1.5">
                              <div className="flex justify-between text-xs mb-0.5">
                                <span className="text-slate-400">{label}</span>
                                <span className="text-slate-300">{pct(value)}</span>
                              </div>
                              <div className="h-1.5 rounded-full bg-slate-700 overflow-hidden">
                                <div className={`h-full rounded-full ${color}`} style={{ width: `${Math.round(value * 100)}%` }} />
                              </div>
                            </div>
                          ))}
                        </div>
                        <div>
                          <p className="text-xs text-slate-400 uppercase tracking-wider mb-2">Mental States</p>
                          <ul className="space-y-1">
                            {(d.dominant_mental_states ?? []).slice(0, 4).map(([state, freq]) => (
                              <li key={state} className="flex justify-between text-xs">
                                <span className="text-slate-300 capitalize">{state.replace(/_/g, ' ')}</span>
                                <span className="text-slate-500">{pct(freq)}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <p className="text-xs text-slate-400 uppercase tracking-wider mb-2">Indicators</p>
                          <div className="space-y-1 text-xs">
                            <div className="flex justify-between"><span className="text-slate-400">Confidence</span><span className="text-sky-400 font-semibold">{pct(d.avg_confidence ?? 0)}</span></div>
                            <div className="flex justify-between"><span className="text-slate-400">Stability</span><span className="text-indigo-400 font-semibold">{pct(d.avg_stability ?? 0)}</span></div>
                            <div className="flex justify-between"><span className="text-slate-400">High-risk time</span><span className="text-red-400 font-semibold">{pct(d.high_risk_duration_ratio ?? 0)}</span></div>
                            <div className="flex justify-between"><span className="text-slate-400">Escalations</span><span className="text-white">{d.total_risk_escalations ?? 0}</span></div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-12 text-center">
              <Brain size={40} className="text-slate-600 mx-auto mb-3" />
              <p className="text-slate-400">No sessions recorded yet.</p>
              <Link to={`/live?user=${selectedUser}`} className="mt-3 inline-block text-indigo-400 text-sm hover:underline">
                Start your first session →
              </Link>
            </div>
          )}
        </div>
      )}

      {/* ── TAB: OWN ANALYTICS ────────────────────────────────── */}
      {selectedUser && !loading && activeTab === 'analytics' && (
        <div className="space-y-6">
          {/* Wellness trend chart */}
          {chartData.length > 0 ? (
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5">
              <p className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                <TrendingUp size={14} className="text-indigo-400" /> Wellness Trend (30 days)
              </p>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="gS" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="gP" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#1e293b" />
                  <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} unit="%" domain={[0, 100]} />
                  <Tooltip {...TOOLTIP} formatter={(v: number) => [`${v}%`]} />
                  <Area type="monotone" dataKey="stress"   name="Stress"        stroke="#f59e0b" fill="url(#gS)" strokeWidth={2} dot={false} />
                  <Area type="monotone" dataKey="positive" name="Positive Mood" stroke="#10b981" fill="url(#gP)" strokeWidth={2} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-12 text-center">
              <BarChart2 size={40} className="text-slate-600 mx-auto mb-3" />
              <p className="text-slate-400">Not enough data yet. Complete a few sessions to see trends.</p>
            </div>
          )}

          {/* Personality profile */}
          {psv ? (
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5">
              <p className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                <Brain size={14} className="text-purple-400" /> Your Personality Profile
              </p>
              <div className="grid md:grid-cols-2 gap-6 items-center">
                <PersonalityRadar psv={psv} />
                <div className="space-y-3">
                  {[
                    { label: 'Emotional Stability', value: psv.emotional_stability, color: 'bg-indigo-500',  desc: 'How balanced you feel across sessions' },
                    { label: 'Recovery Speed',      value: psv.recovery_speed,      color: 'bg-emerald-500', desc: 'How quickly you bounce back from stress' },
                    { label: 'Positivity',          value: psv.positivity_bias,      color: 'bg-green-500',  desc: 'Tendency toward positive emotions'      },
                  ].map(({ label, value, color, desc }) => (
                    <div key={label}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-300 font-medium">{label}</span>
                        <span className="text-slate-400">{Math.round(value * 100)}%</span>
                      </div>
                      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div className={`h-full rounded-full ${color}`} style={{ width: `${Math.round(value * 100)}%` }} />
                      </div>
                      <p className="text-xs text-slate-600 mt-0.5">{desc}</p>
                    </div>
                  ))}
                  <p className="text-xs text-slate-600 pt-1">
                    Based on {psv.total_sessions_processed} sessions · {Math.round(psv.confidence * 100)}% confidence
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-8 text-center">
              <Brain size={36} className="text-slate-600 mx-auto mb-2" />
              <p className="text-slate-400 text-sm">Personality profile builds over time. Keep recording sessions.</p>
            </div>
          )}
        </div>
      )}

      {/* ── TAB: DAILY REPORTS ────────────────────────────────── */}
      {selectedUser && !loading && activeTab === 'reports' && (
        <div className="space-y-5">
          {/* Today's highlight */}
          {todayProfile ? (
            <div className="rounded-xl bg-emerald-900/20 border border-emerald-700/50 p-5">
              <div className="flex items-center gap-2 mb-4">
                <Calendar size={15} className="text-emerald-400" />
                <h2 className="font-semibold text-white">Today's Report</h2>
                <span className="text-xs text-slate-500 ml-auto">{today}</span>
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <InfoCard label="Sessions Today" value={`${todayProfile.total_sessions}`}                                           color="text-emerald-300" />
                <InfoCard label="Time Spent"     value={fmtDuration(todayProfile.total_duration_minutes ?? 0)}                    color="text-sky-300"    />
                <InfoCard label="Stress Level"   value={pct(todayProfile.avg_stress_ratio ?? 0)}                                  color="text-amber-300"  />
                <InfoCard label="Top Mood"       value={(todayProfile.dominant_mental_states?.[0]?.[0] ?? '—').replace(/_/g, ' ')} color="text-white"      />
              </div>
            </div>
          ) : (
            <div className="rounded-xl bg-slate-800/40 border border-slate-700 p-5 flex items-center justify-between flex-wrap gap-3">
              <div>
                <p className="text-white font-semibold">No session today yet</p>
                <p className="text-slate-400 text-sm mt-0.5">Start a session to track today's wellness</p>
              </div>
              <Link to={`/live?user=${selectedUser}`}
                className="flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 text-white text-sm px-4 py-2 rounded-lg transition-colors">
                <Activity size={14} /> Start Session
              </Link>
            </div>
          )}

          {/* All daily reports */}
          {sortedEntries.length > 0 ? (
            <div>
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Past Reports (30 days)</h3>
              <div className="space-y-2">
                {sortedEntries.filter(([date]) => date !== today).map(([date, d]) => (
                  <div key={date} className="rounded-xl bg-slate-800/60 border border-slate-700 p-4 flex items-center flex-wrap gap-4">
                    <div className="flex-1 min-w-0">
                      <p className="text-white text-sm font-semibold">{date}</p>
                      <p className="text-xs text-slate-500">{d.total_sessions} session{d.total_sessions !== 1 ? 's' : ''} · {fmtDuration(d.total_duration_minutes ?? 0)}</p>
                    </div>
                    <div className="flex items-center gap-6 text-xs">
                      <div className="text-center">
                        <p className="text-amber-400 font-semibold">{pct(d.avg_stress_ratio ?? 0)}</p>
                        <p className="text-slate-500">Stress</p>
                      </div>
                      <div className="text-center">
                        <p className="text-emerald-400 font-semibold">{pct(d.positive_ratio ?? 0)}</p>
                        <p className="text-slate-500">Positive</p>
                      </div>
                      <div className="text-center">
                        <p className="text-sky-400 font-semibold capitalize">{(d.dominant_mental_states?.[0]?.[0] ?? '—').replace(/_/g, ' ')}</p>
                        <p className="text-slate-500">Top State</p>
                      </div>
                      <RiskBadge risk={riskLabel(d.avg_risk_level ?? 0)} size="sm" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-12 text-center">
              <FileText size={40} className="text-slate-600 mx-auto mb-3" />
              <p className="text-slate-400">No daily reports yet. Complete sessions to generate reports.</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function StatBox({ label, value, icon, color }: { label: string; value: string; icon: React.ReactNode; color: string }) {
  return (
    <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-4 flex items-center gap-3">
      <div className="w-9 h-9 rounded-lg bg-slate-700 flex items-center justify-center flex-shrink-0">
        <span className={color}>{icon}</span>
      </div>
      <div>
        <p className={`text-lg font-bold ${color}`}>{value}</p>
        <p className="text-xs text-slate-400">{label}</p>
      </div>
    </div>
  )
}

function InfoCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="rounded-lg bg-emerald-900/20 border border-emerald-700/30 p-3">
      <p className={`text-lg font-bold capitalize ${color}`}>{value}</p>
      <p className="text-xs text-slate-400 mt-0.5">{label}</p>
    </div>
  )
}