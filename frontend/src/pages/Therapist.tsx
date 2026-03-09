import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  ClipboardList, Search, Video, BarChart3, AlertTriangle,
  StickyNote, Plus, Trash2, User2, Download,
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts'
import type { UserProfile, DailyProfile, PSVData } from '../types'
import PersonalityRadar from '../components/PersonalityRadar'

type PatientTab = 'overview' | 'sessions' | 'analytics' | 'alerts' | 'notes'

interface AlertItem {
  timestamp: string
  session_date: string
  type: string
  severity: number
  description: string
}

interface Note {
  id: string
  text: string
  createdAt: string
}

// ── localStorage helpers ─────────────────────────────────────────────────────
function loadNotes(userId: string): Note[] {
  try { return JSON.parse(localStorage.getItem(`psych_notes_${userId}`) ?? '[]') }
  catch { return [] }
}
function saveNotes(userId: string, notes: Note[]) {
  localStorage.setItem(`psych_notes_${userId}`, JSON.stringify(notes))
}

// ── helpers ──────────────────────────────────────────────────────────────────
function timeAgo(iso: string) {
  const diff = Date.now() - new Date(iso).getTime()
  const m = Math.floor(diff / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  return h < 24 ? `${h}h ago` : `${Math.floor(h / 24)}d ago`
}

function fmtDuration(minutes: number) {
  const h = Math.floor(minutes / 60)
  const m = Math.round(minutes % 60)
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

function pct(v: number) { return `${Math.round(v * 100)}%` }

const TOOLTIP = {
  contentStyle: { background: '#1e293b', border: '1px solid #334155', borderRadius: 8 },
  labelStyle: { color: '#94a3b8' },
}

// ── main component ────────────────────────────────────────────────────────────
export default function Therapist() {
  const [patients, setPatients] = useState<UserProfile[]>([])
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<UserProfile | null>(null)
  const [patientTab, setPatientTab] = useState<PatientTab>('overview')

  // per-patient data
  const [history, setHistory] = useState<Record<string, DailyProfile>>({})
  const [psv, setPsv] = useState<PSVData | null>(null)
  const [alerts, setAlerts] = useState<AlertItem[]>([])
  const [notes, setNotes] = useState<Note[]>([])
  const [noteText, setNoteText] = useState('')
  const [loadingData, setLoadingData] = useState(false)

  // alert badge counts per patient
  const [alertCounts, setAlertCounts] = useState<Record<string, number>>({})

  // load patient list once
  useEffect(() => {
    fetch('/api/users/')
      .then(r => r.json())
      .then((users: UserProfile[]) => {
        setPatients(users)
        users.forEach(u => {
          fetch(`/api/analytics/${u.user_id}/alerts?limit=1`)
            .then(r => r.json())
            .then(d => setAlertCounts(prev => ({ ...prev, [u.user_id]: d.total ?? 0 })))
            .catch(() => {})
        })
      })
      .catch(console.error)
  }, [])

  // load selected patient data
  useEffect(() => {
    if (!selected) return
    setLoadingData(true)
    setHistory({}); setPsv(null); setAlerts([])
    setNotes(loadNotes(selected.user_id))
    Promise.all([
      fetch(`/api/analytics/${selected.user_id}/history?days=30`).then(r => r.ok ? r.json() : {}).catch(() => ({})),
      fetch(`/api/analytics/${selected.user_id}/psv`).then(r => r.ok ? r.json() : null).catch(() => null),
      fetch(`/api/analytics/${selected.user_id}/alerts?limit=50`).then(r => r.ok ? r.json() : { alerts: [] }).catch(() => ({ alerts: [] })),
    ]).then(([hist, psvData, alertData]) => {
      setHistory(hist)
      setPsv(psvData)
      setAlerts(alertData.alerts ?? [])
    }).finally(() => setLoadingData(false))
  }, [selected])

  function addNote() {
    if (!selected || !noteText.trim()) return
    const note: Note = { id: Date.now().toString(), text: noteText.trim(), createdAt: new Date().toISOString() }
    const updated = [note, ...notes]
    setNotes(updated)
    saveNotes(selected.user_id, updated)
    setNoteText('')
  }

  function deleteNote(id: string) {
    if (!selected) return
    const updated = notes.filter(n => n.id !== id)
    setNotes(updated)
    saveNotes(selected.user_id, updated)
  }

  const filteredPatients = patients.filter(u =>
    u.name.toLowerCase().includes(search.toLowerCase())
  )

  const sortedHistory = Object.entries(history).sort(([a], [b]) => b.localeCompare(a))
  const chartData = [...sortedHistory].reverse().map(([date, d]) => ({
    date: date.slice(5),
    stress: Math.round((d.avg_stress_ratio ?? 0) * 100),
    risk:   Math.round((d.avg_risk_level   ?? 0) * 33),
  }))

  const totalSessions = sortedHistory.reduce((a, [, d]) => a + (d.total_sessions ?? 0), 0)
  const totalMinutes  = sortedHistory.reduce((a, [, d]) => a + (d.total_duration_minutes ?? 0), 0)
  const avgStress     = sortedHistory.length
    ? sortedHistory.reduce((a, [, d]) => a + (d.avg_stress_ratio ?? 0), 0) / sortedHistory.length
    : 0
  const criticalAlerts = alerts.filter(a => a.severity >= 0.8).length

  const PATIENT_TABS: { id: PatientTab; label: string }[] = [
    { id: 'overview',  label: 'Overview'                                               },
    { id: 'sessions',  label: 'Sessions'                                               },
    { id: 'analytics', label: 'Analytics'                                              },
    { id: 'alerts',    label: criticalAlerts > 0 ? `Alerts (${criticalAlerts} ⚠)` : 'Alerts' },
    { id: 'notes',     label: notes.length > 0 ? `Notes (${notes.length})` : 'Notes'  },
  ]

  return (
    <div className="flex">
      {/* ── Patient sidebar ──────────────────────────────────────── */}
      <div className="sticky top-0 h-screen w-72 flex-shrink-0 bg-[#161b27] border-r border-slate-800 flex flex-col overflow-hidden">
        <div className="px-4 py-5 border-b border-slate-800 flex-shrink-0">
          <div className="flex items-center gap-2 mb-4">
            <ClipboardList size={18} className="text-indigo-400" />
            <h1 className="font-bold text-white">Therapist Portal</h1>
          </div>
          <div className="flex items-center gap-2 bg-slate-800 rounded-lg px-3 py-2 border border-slate-700">
            <Search size={13} className="text-slate-500" />
            <input
              type="text" placeholder="Search patients…" value={search}
              onChange={e => setSearch(e.target.value)}
              className="flex-1 bg-transparent text-white placeholder-slate-500 text-sm focus:outline-none"
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto py-2">
          {filteredPatients.length === 0 && (
            <p className="text-slate-500 text-xs text-center mt-8 px-4">No patients registered.</p>
          )}
          {filteredPatients.map(u => (
            <button
              key={u.user_id}
              onClick={() => { setSelected(u); setPatientTab('overview') }}
              className={`w-full flex items-center gap-3 px-4 py-3 transition-colors text-left ${
                selected?.user_id === u.user_id
                  ? 'bg-indigo-600/20 border-r-2 border-indigo-500'
                  : 'hover:bg-slate-800'
              }`}
            >
              <div className="w-9 h-9 rounded-full bg-indigo-600/30 flex items-center justify-center flex-shrink-0">
                <span className="text-indigo-300 font-semibold text-sm">{u.name[0].toUpperCase()}</span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-white text-sm font-medium truncate">{u.name}</p>
                <p className="text-slate-500 text-xs">{u.total_sessions} sessions</p>
              </div>
              {(alertCounts[u.user_id] ?? 0) > 0 && (
                <span className="w-5 h-5 rounded-full bg-amber-500 text-black text-xs font-bold flex items-center justify-center flex-shrink-0">
                  {alertCounts[u.user_id] > 9 ? '9+' : alertCounts[u.user_id]}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* ── Main panel ───────────────────────────────────────────── */}
      <div className="flex-1 min-h-screen bg-[#0f1117]">
        {!selected ? (
          <div className="flex items-center justify-center h-full min-h-[60vh]">
            <div className="text-center">
              <User2 size={48} className="text-slate-700 mx-auto mb-3" />
              <p className="text-slate-400 font-medium">Select a patient</p>
              <p className="text-slate-600 text-sm mt-1">Choose from the list on the left</p>
            </div>
          </div>
        ) : (
          <div className="p-6 space-y-5">
            {/* Patient header */}
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-indigo-600/30 flex items-center justify-center">
                  <span className="text-indigo-300 font-bold text-xl">{selected.name[0].toUpperCase()}</span>
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">{selected.name}</h2>
                  <p className="text-xs text-slate-500 font-mono">{selected.user_id}</p>
                </div>
              </div>
              <div className="flex gap-2">
                <Link to={`/live?user=${selected.user_id}`}
                  className="flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs px-3 py-2 rounded-lg transition-colors">
                  <Video size={12} /> Start Session
                </Link>
                <a href={`/api/analytics/${selected.user_id}/report`} target="_blank" rel="noreferrer"
                  className="flex items-center gap-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs px-3 py-2 rounded-lg transition-colors">
                  <BarChart3 size={12} /> Full Report
                </a>
              </div>
            </div>

            {/* Sub-tabs */}
            <div className="flex gap-1 bg-slate-800/60 border border-slate-700 rounded-xl p-1 overflow-x-auto">
              {PATIENT_TABS.map(t => (
                <button key={t.id} onClick={() => setPatientTab(t.id)}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${
                    patientTab === t.id ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
                  }`}>
                  {t.label}
                </button>
              ))}
            </div>

            {loadingData && <p className="text-slate-500 text-sm">Loading patient data…</p>}

            {/* ── Overview ── */}
            {patientTab === 'overview' && !loadingData && (
              <div className="space-y-5">
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <MiniStat label="Total Sessions"  value={`${totalSessions}`}           color="text-indigo-400" />
                  <MiniStat label="Total Time"      value={fmtDuration(totalMinutes)}    color="text-sky-400"    />
                  <MiniStat label="Avg Stress"      value={pct(avgStress)}               color="text-amber-400"  />
                  <MiniStat label="Critical Alerts" value={`${criticalAlerts}`}          color="text-red-400"    />
                </div>

                {chartData.length > 0 && (
                  <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5">
                    <p className="text-sm font-semibold text-white mb-4">Stress & Risk Trend (30 days)</p>
                    <ResponsiveContainer width="100%" height={160}>
                      <LineChart data={chartData}>
                        <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} unit="%" domain={[0, 100]} />
                        <Tooltip {...TOOLTIP} formatter={(v: number) => [`${v}%`]} />
                        <Line type="monotone" dataKey="stress" name="Stress" stroke="#f59e0b" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="risk"   name="Risk"   stroke="#ef4444" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {psv && (
                  <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5">
                    <p className="text-sm font-semibold text-white mb-4">Personality Profile</p>
                    <div className="grid md:grid-cols-2 gap-4">
                      <PersonalityRadar psv={psv} />
                      <div className="space-y-3 py-2">
                        {[
                          { label: 'Emotional Stability', value: psv.emotional_stability, color: 'bg-indigo-500' },
                          { label: 'Recovery Speed',      value: psv.recovery_speed,      color: 'bg-emerald-500' },
                          { label: 'Positivity Bias',     value: psv.positivity_bias,      color: 'bg-green-500' },
                          { label: 'Stress Sensitivity',  value: psv.stress_sensitivity,   color: 'bg-orange-500', invert: true },
                        ].map(({ label, value, color, invert }) => {
                          const display = invert ? 1 - value : value
                          return (
                            <div key={label}>
                              <div className="flex justify-between text-xs text-slate-400 mb-1">
                                <span>{label}</span><span>{Math.round(display * 100)}%</span>
                              </div>
                              <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                <div className={`h-full rounded-full ${color}`} style={{ width: `${Math.round(display * 100)}%` }} />
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  </div>
                )}

                {sortedHistory.length === 0 && (
                  <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-8 text-center text-slate-500 text-sm">
                    No session data yet. <Link to={`/live?user=${selected.user_id}`} className="text-indigo-400 hover:underline">Start a session →</Link>
                  </div>
                )}
              </div>
            )}

            {/* ── Sessions ── */}
            {patientTab === 'sessions' && !loadingData && (
              <div className="space-y-3">
                {sortedHistory.length === 0 ? (
                  <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-8 text-center text-slate-500 text-sm">No sessions recorded.</div>
                ) : sortedHistory.map(([date, d]) => (
                  <div key={date} className="rounded-xl bg-slate-800/60 border border-slate-700 px-5 py-4">
                    <div className="flex items-center justify-between flex-wrap gap-3">
                      <div>
                        <p className="text-white font-semibold">{date}</p>
                        <p className="text-xs text-slate-500">{d.total_sessions} session{d.total_sessions !== 1 ? 's' : ''} · {fmtDuration(d.total_duration_minutes ?? 0)}</p>
                      </div>
                      <div className="flex items-center gap-4 text-xs">
                        <span className="text-slate-400">Stress: <span className="text-amber-400 font-semibold">{pct(d.avg_stress_ratio ?? 0)}</span></span>
                        <span className="text-slate-400">Top state: <span className="text-white capitalize">{(d.dominant_mental_states?.[0]?.[0] ?? '—').replace(/_/g, ' ')}</span></span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* ── Analytics ── */}
            {patientTab === 'analytics' && !loadingData && (
              <div className="space-y-5">
                <div className="flex justify-end gap-2">
                  <a href={`/api/analytics/${selected.user_id}/export/json`} download
                    className="flex items-center gap-1 text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 px-3 py-1.5 rounded-lg transition-colors">
                    <Download size={12} /> JSON
                  </a>
                  <a href={`/api/analytics/${selected.user_id}/export/csv`} download
                    className="flex items-center gap-1 text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 px-3 py-1.5 rounded-lg transition-colors">
                    <Download size={12} /> CSV
                  </a>
                </div>

                {chartData.length > 0 ? (
                  <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5">
                    <p className="text-sm font-semibold text-white mb-4">30-Day Trend</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={chartData}>
                        <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} unit="%" domain={[0, 100]} />
                        <Tooltip {...TOOLTIP} formatter={(v: number) => [`${v}%`]} />
                        <Line type="monotone" dataKey="stress" name="Stress" stroke="#f59e0b" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="risk"   name="Risk"   stroke="#ef4444" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-8 text-center text-slate-500 text-sm">No data.</div>
                )}

                {psv && (
                  <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5">
                    <p className="text-sm font-semibold text-white mb-4">Personality State Vector ({psv.total_sessions_processed} sessions)</p>
                    <PersonalityRadar psv={psv} />
                  </div>
                )}
              </div>
            )}

            {/* ── Alerts ── */}
            {patientTab === 'alerts' && !loadingData && (
              <div className="space-y-2">
                {alerts.length === 0 ? (
                  <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-8 text-center text-slate-500 text-sm">
                    No alerts for this patient.
                  </div>
                ) : alerts.map((a, i) => (
                  <div key={i} className="rounded-xl bg-slate-800/60 border border-slate-700 px-4 py-3 flex items-start gap-3">
                    <AlertTriangle size={14} className={`mt-0.5 flex-shrink-0 ${a.severity >= 0.8 ? 'text-red-400' : a.severity >= 0.6 ? 'text-orange-400' : 'text-amber-400'}`} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-white text-sm font-medium capitalize">{a.type.replace(/_/g, ' ')}</span>
                        <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${a.severity >= 0.8 ? 'text-red-400 bg-red-900/20' : 'text-amber-400 bg-amber-900/20'}`}>
                          {Math.round(a.severity * 100)}%
                        </span>
                        <span className="text-xs text-slate-500">{a.session_date}</span>
                      </div>
                      <p className="text-slate-400 text-xs mt-0.5">{a.description}</p>
                    </div>
                    <span className="text-xs text-slate-600 flex-shrink-0">{timeAgo(a.timestamp)}</span>
                  </div>
                ))}
              </div>
            )}

            {/* ── Notes ── */}
            {patientTab === 'notes' && (
              <div className="space-y-4">
                <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-4">
                  <textarea
                    placeholder="Write a clinical observation or note…"
                    value={noteText}
                    onChange={e => setNoteText(e.target.value)}
                    rows={3}
                    className="w-full bg-slate-900 border border-slate-600 text-white placeholder-slate-500 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  />
                  <div className="flex justify-end mt-2">
                    <button onClick={addNote} disabled={!noteText.trim()}
                      className="flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-sm px-4 py-1.5 rounded-lg">
                      <Plus size={14} /> Add Note
                    </button>
                  </div>
                </div>

                {notes.length === 0 ? (
                  <div className="rounded-xl bg-slate-800/40 border border-slate-700 p-8 text-center text-slate-500 text-sm">
                    No notes yet. Add clinical observations above.
                  </div>
                ) : (
                  <div className="space-y-2">
                    {notes.map(n => (
                      <div key={n.id} className="rounded-xl bg-slate-800/60 border border-slate-700 px-4 py-3 flex gap-3">
                        <StickyNote size={14} className="text-yellow-400 mt-1 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <p className="text-white text-sm whitespace-pre-wrap">{n.text}</p>
                          <p className="text-xs text-slate-500 mt-1">{new Date(n.createdAt).toLocaleString()}</p>
                        </div>
                        <button onClick={() => deleteNote(n.id)} className="text-slate-600 hover:text-red-400 flex-shrink-0 mt-0.5">
                          <Trash2 size={13} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function MiniStat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-4">
      <p className={`text-xl font-bold ${color}`}>{value}</p>
      <p className="text-xs text-slate-400 mt-0.5">{label}</p>
    </div>
  )
}
