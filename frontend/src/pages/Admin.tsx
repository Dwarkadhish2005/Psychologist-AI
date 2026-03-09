import { Fragment, useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  Activity, Video, BarChart3, Calendar, Clock, Layers,
  CheckCircle2, XCircle, Bell, Shield, AlertTriangle,
  Trash2, UserPlus, Search, RefreshCw, Users, ClipboardList,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import type { UserProfile } from '../types'

type Tab = 'overview' | 'users' | 'therapists' | 'sessions' | 'alerts'

interface SessionStatus {
  is_running: boolean
  active_user_id: string | null
  session_duration_seconds: number | null
  frame_count: number | null
}

interface AlertItem {
  timestamp: string
  session_date: string
  type: string
  severity: number
  description: string
}

function fmtSeconds(s: number) {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`
}

function severityColor(s: number) {
  if (s >= 0.8) return 'text-red-400 bg-red-900/20'
  if (s >= 0.6) return 'text-orange-400 bg-orange-900/20'
  return 'text-amber-400 bg-amber-900/20'
}

function timeAgo(iso: string) {
  const diff = Date.now() - new Date(iso).getTime()
  const m = Math.floor(diff / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  return h < 24 ? `${h}h ago` : `${Math.floor(h / 24)}d ago`
}

export default function Admin() {
  const [searchParams, setSearchParams] = useSearchParams()
  const urlTab = (searchParams.get('tab') as Tab) ?? 'overview'
  const [tab, setTabState] = useState<Tab>(urlTab)

  function setTab(t: Tab) {
    setTabState(t)
    setSearchParams({ tab: t })
  }
  const [status, setStatus] = useState<SessionStatus | null>(null)
  const [users, setUsers] = useState<UserProfile[]>([])
  const [search, setSearch] = useState('')
  const [tick, setTick] = useState(0)
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)
  const [allAlerts, setAllAlerts] = useState<Array<AlertItem & { userName: string }>>([])
  const [alertsLoading, setAlertsLoading] = useState(false)
  // ── Therapist management state ─────────────────────────────────────────────
  const { therapists, addTherapist, toggleTherapist, deleteTherapist, resetTherapistPassword } = useAuth()
  const [tName, setTName] = useState('')
  const [tEmail, setTEmail] = useState('')
  const [tPw, setTPw] = useState('')
  const [tPwConfirm, setTPwConfirm] = useState('')
  const [tError, setTError] = useState('')
  const [tSuccess, setTSuccess] = useState('')
  const [tAdding, setTAdding] = useState(false)
  const [resetingId, setResetingId] = useState<string | null>(null)
  const [resetNewPw, setResetNewPw] = useState('')
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)
  useEffect(() => {
    const t = (searchParams.get('tab') as Tab) ?? 'overview'
    setTabState(t)
  }, [searchParams])

  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 2000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    const ctrl = new AbortController()
    fetch('/api/sessions/status', { signal: ctrl.signal }).then(r => r.json()).then(setStatus).catch(() => {})
    return () => ctrl.abort()
  }, [tick])

  useEffect(() => {
    const ctrl = new AbortController()
    fetch('/api/users/', { signal: ctrl.signal }).then(r => r.json()).then(setUsers).catch(() => {})
    return () => ctrl.abort()
  }, [tick])

  useEffect(() => {
    if (tab !== 'alerts' || users.length === 0) return
    setAlertsLoading(true)
    Promise.all(
      users.map(u =>
        fetch(`/api/analytics/${u.user_id}/alerts?limit=20`)
          .then(r => r.json())
          .then(d => (d.alerts || []).map((a: AlertItem) => ({ ...a, userName: u.name })))
          .catch(() => [])
      )
    ).then(nested => {
      const flat = nested.flat().sort((a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      )
      setAllAlerts(flat)
    }).finally(() => setAlertsLoading(false))
  // NOTE: depends on users.length not the array reference — avoids re-fetch on every 2s tick
  }, [tab, users.length])

  const activeUser = users.find(u => u.user_id === status?.active_user_id) ?? null
  const filteredUsers = users.filter(u =>
    u.name.toLowerCase().includes(search.toLowerCase()) ||
    u.user_id.toLowerCase().includes(search.toLowerCase())
  )

  async function stopSession() {
    if (!confirm('Force stop the current session?')) return
    await fetch('/api/sessions/stop', { method: 'POST' })
    setTick(t => t + 1)
  }

  async function deleteUser(u: UserProfile) {
    if (!confirm(`Delete "${u.name}" and all their data?`)) return
    await fetch(`/api/users/${u.user_id}`, { method: 'DELETE' })
    setTick(t => t + 1)
  }

  async function registerUser(e: React.FormEvent) {
    e.preventDefault()
    if (!newName.trim()) return
    setCreating(true)
    await fetch('/api/users/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName.trim() }),
    }).catch(console.error)
    setNewName('')
    setCreating(false)
    setTick(t => t + 1)
  }

  async function handleAddTherapist(e: React.FormEvent) {
    e.preventDefault()
    setTError(''); setTSuccess('')
    if (tPw !== tPwConfirm) { setTError('Passwords do not match.'); return }
    setTAdding(true)
    const res = await addTherapist(tName, tEmail, tPw)
    if (res.ok) {
      setTSuccess(`Therapist "${tName.trim()}" created successfully.`)
      setTName(''); setTEmail(''); setTPw(''); setTPwConfirm('')
    } else {
      setTError(res.error ?? 'Failed to create account.')
    }
    setTAdding(false)
  }

  const TABS: { id: Tab; label: string }[] = [
    { id: 'overview',   label: 'Overview'                          },
    { id: 'users',      label: `Users (${users.length})`           },
    { id: 'therapists', label: `Therapists (${therapists.length})` },
    { id: 'sessions',   label: 'Sessions'                         },
    { id: 'alerts',     label: 'Alerts'                           },
  ]

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-indigo-600/20 flex items-center justify-center">
          <Shield size={20} className="text-indigo-400" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-white">Admin Portal</h1>
          <p className="text-slate-400 text-sm">System health · User management · Session monitoring · Alerts</p>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 bg-slate-800/60 border border-slate-700 rounded-xl p-1 w-fit">
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              tab === t.id ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard label="Registered Users" value={users.length} color="text-indigo-400" icon={<Users size={16} />} />
            <StatCard label="Active Sessions"  value={status?.is_running ? 1 : 0} color="text-emerald-400" icon={<Activity size={16} />} />
            <StatCard label="Total Sessions"   value={users.reduce((a, u) => a + u.total_sessions, 0)} color="text-sky-400" icon={<Calendar size={16} />} />
            <StatCard label="AI Phases Active" value={5} color="text-purple-400" icon={<Layers size={16} />} />
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5">
              <h2 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                <Activity size={14} className="text-emerald-400" /> API Health
              </h2>
              <div className="space-y-2.5">
                {[
                  { label: 'Users',    path: '/api/users/'          },
                  { label: 'Sessions', path: '/api/sessions/status' },
                  { label: 'Video',    path: '/api/video/feed'      },
                  { label: 'Health',   path: '/api/health'          },
                ].map(e => <EndpointRow key={e.label} {...e} />)}
              </div>
            </div>

            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5">
              <h2 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                <Layers size={14} className="text-indigo-400" /> AI Pipeline
              </h2>
              <div className="space-y-2.5">
                {[
                  'Face Emotion CNN',
                  'Voice Emotion + Stress',
                  'Multi-Modal Fusion',
                  'Cognitive Layer',
                  'Personality State Vector',
                ].map((label, i) => (
                  <div key={label} className="flex items-center justify-between text-sm">
                    <span className="text-slate-400">
                      <span className="text-indigo-400 font-mono text-xs mr-2">Phase {i + 1}</span>
                      {label}
                    </span>
                    <CheckCircle2 size={14} className="text-emerald-400" />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── USERS ── */}
      {tab === 'users' && (
        <div className="space-y-4">
          <div className="grid md:grid-cols-2 gap-4">
            <form onSubmit={registerUser} className="rounded-xl bg-slate-800/60 border border-slate-700 p-4 flex gap-2">
              <input
                type="text" placeholder="New user name…" value={newName}
                onChange={e => setNewName(e.target.value)} required
                className="flex-1 bg-slate-900 border border-slate-600 text-white placeholder-slate-500 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
              <button type="submit" disabled={creating}
                className="flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white px-4 py-2 rounded-lg text-sm font-medium">
                <UserPlus size={15} /> {creating ? '…' : 'Add'}
              </button>
            </form>
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-4 flex items-center gap-2">
              <Search size={15} className="text-slate-500 flex-shrink-0" />
              <input type="text" placeholder="Search users…" value={search}
                onChange={e => setSearch(e.target.value)}
                className="flex-1 bg-transparent text-white placeholder-slate-500 text-sm focus:outline-none" />
            </div>
          </div>

          <div className="rounded-xl bg-slate-800/60 border border-slate-700 overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">User</th>
                  <th className="text-center px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider hidden md:table-cell">Sessions</th>
                  <th className="text-center px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider hidden lg:table-cell">Last Active</th>
                  <th className="text-right px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredUsers.map(u => (
                  <tr key={u.user_id} className="border-b border-slate-700/50 last:border-0 hover:bg-slate-700/20">
                    <td className="px-4 py-3">
                      <p className="text-white font-medium">{u.name}</p>
                      <p className="text-xs text-slate-500 font-mono">{u.user_id.split('_').slice(-1)[0]}</p>
                    </td>
                    <td className="px-4 py-3 text-center text-slate-300 hidden md:table-cell">{u.total_sessions}</td>
                    <td className="px-4 py-3 text-center text-slate-400 text-xs hidden lg:table-cell">{new Date(u.last_active).toLocaleDateString()}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center justify-end gap-1">
                        <Link to={`/live?user=${u.user_id}`}      className="p-1.5 rounded-lg text-slate-400 hover:text-indigo-400 hover:bg-indigo-600/10 transition-colors" title="Live"><Video    size={13} /></Link>
                        <Link to={`/analytics?user=${u.user_id}`} className="p-1.5 rounded-lg text-slate-400 hover:text-sky-400   hover:bg-sky-600/10   transition-colors" title="Analytics"><BarChart3 size={13} /></Link>
                        <Link to={`/history?user=${u.user_id}`}   className="p-1.5 rounded-lg text-slate-400 hover:text-emerald-400 hover:bg-emerald-600/10 transition-colors" title="History"><Calendar  size={13} /></Link>
                        <Link to={`/alerts?user=${u.user_id}`}    className="p-1.5 rounded-lg text-slate-400 hover:text-amber-400  hover:bg-amber-600/10  transition-colors" title="Alerts"><Bell      size={13} /></Link>
                        <button onClick={() => deleteUser(u)} title="Delete"
                          className="p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-600/10 transition-colors">
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
                {filteredUsers.length === 0 && (
                  <tr><td colSpan={4} className="px-4 py-10 text-center text-slate-500 text-sm">No users found.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── SESSIONS ── */}
      {tab === 'sessions' && (
        <div className="space-y-5">
          <div className={`rounded-xl border p-6 ${status?.is_running ? 'bg-emerald-900/10 border-emerald-800' : 'bg-slate-800/60 border-slate-700'}`}>
            <div className="flex items-start justify-between flex-wrap gap-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <div className={`w-2.5 h-2.5 rounded-full ${status?.is_running ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`} />
                  <span className="text-white font-semibold text-lg">
                    {status?.is_running ? 'Session Active' : 'No Active Session'}
                  </span>
                </div>
                {activeUser && (
                  <p className="text-slate-400 text-sm">
                    User: <span className="text-white font-medium">{activeUser.name}</span>
                    <span className="text-slate-600 text-xs ml-2 font-mono">{activeUser.user_id}</span>
                  </p>
                )}
              </div>
              <div className="flex items-center gap-6">
                {status?.session_duration_seconds != null && (
                  <div className="text-center">
                    <p className="text-2xl font-bold text-white">{fmtSeconds(status.session_duration_seconds)}</p>
                    <p className="text-xs text-slate-400 flex items-center gap-1 justify-center"><Clock size={11} /> Duration</p>
                  </div>
                )}
                {status?.frame_count != null && (
                  <div className="text-center">
                    <p className="text-2xl font-bold text-white">{status.frame_count.toLocaleString()}</p>
                    <p className="text-xs text-slate-400">Frames</p>
                  </div>
                )}
              </div>
            </div>
            {status?.is_running && (
              <div className="mt-5 flex gap-3 flex-wrap">
                {activeUser && (
                  <Link to={`/live?user=${activeUser.user_id}`}
                    className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-sm transition-colors">
                    <Video size={14} /> View Live
                  </Link>
                )}
                <button onClick={stopSession}
                  className="flex items-center gap-2 bg-red-900/40 hover:bg-red-900/70 text-red-400 border border-red-800 px-4 py-2 rounded-lg text-sm transition-colors">
                  <XCircle size={14} /> Force Stop
                </button>
              </div>
            )}
          </div>

          <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5">
            <h2 className="text-sm font-semibold text-white mb-4">Session Stats by User</h2>
            {users.length === 0 ? (
              <p className="text-slate-500 text-sm">No users registered.</p>
            ) : (
              <div className="space-y-2">
                {users.map(u => (
                  <div key={u.user_id} className="flex items-center justify-between py-1.5 border-b border-slate-700/40 last:border-0">
                    <span className="text-white text-sm">{u.name}</span>
                    <div className="flex items-center gap-4 text-xs text-slate-400">
                      <span>{u.total_sessions} sessions</span>
                      <span className="hidden sm:block text-slate-600">{new Date(u.last_active).toLocaleDateString()}</span>
                      <Link to={`/history?user=${u.user_id}`} className="text-indigo-400 hover:underline">view history</Link>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── ALERTS ── */}
      {tab === 'alerts' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-slate-400 text-sm">{allAlerts.length} alerts across all users</p>
            <button onClick={() => setTick(t => t + 1)}
              className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-lg">
              <RefreshCw size={12} /> Refresh
            </button>
          </div>

          {alertsLoading && <p className="text-slate-500 text-sm">Loading…</p>}
          {!alertsLoading && allAlerts.length === 0 && (
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-10 text-center text-slate-500 text-sm">
              No alerts found across any user.
            </div>
          )}

          <div className="space-y-2">
            {allAlerts.slice(0, 60).map((a, i) => (
              <div key={i} className="rounded-xl bg-slate-800/60 border border-slate-700 px-4 py-3 flex items-start gap-3">
                <AlertTriangle size={15} className={`mt-0.5 flex-shrink-0 ${a.severity >= 0.8 ? 'text-red-400' : a.severity >= 0.6 ? 'text-orange-400' : 'text-amber-400'}`} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-white text-sm font-medium capitalize">{a.type.replace(/_/g, ' ')}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${severityColor(a.severity)}`}>{Math.round(a.severity * 100)}%</span>
                    <span className="text-xs text-indigo-400 bg-indigo-900/20 px-2 py-0.5 rounded-full">{a.userName}</span>
                  </div>
                  <p className="text-slate-400 text-xs mt-0.5">{a.description}</p>
                </div>
                <span className="text-xs text-slate-600 flex-shrink-0 mt-0.5">{timeAgo(a.timestamp)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── THERAPISTS ── */}
      {tab === 'therapists' && (
        <div className="space-y-5">
          {/* Add therapist form */}
          <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5">
            <h2 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
              <ClipboardList size={14} className="text-indigo-400" /> Create Therapist Account
            </h2>
            <form onSubmit={handleAddTherapist} className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-slate-400 mb-1">Full Name</label>
                <input type="text" value={tName} onChange={e => { setTName(e.target.value); setTError(''); setTSuccess('') }}
                  placeholder="Dr. Jane Smith" required
                  className="w-full bg-slate-900 border border-slate-600 text-white placeholder-slate-500 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500" />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">Email Address</label>
                <input type="email" value={tEmail} onChange={e => { setTEmail(e.target.value); setTError(''); setTSuccess('') }}
                  placeholder="therapist@clinic.com" required
                  className="w-full bg-slate-900 border border-slate-600 text-white placeholder-slate-500 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500" />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">Password</label>
                <input type="password" value={tPw} onChange={e => { setTPw(e.target.value); setTError('') }}
                  placeholder="Min. 6 characters" required minLength={6}
                  className="w-full bg-slate-900 border border-slate-600 text-white placeholder-slate-500 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500" />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">Confirm Password</label>
                <input type="password" value={tPwConfirm} onChange={e => { setTPwConfirm(e.target.value); setTError('') }}
                  placeholder="Repeat password" required
                  className="w-full bg-slate-900 border border-slate-600 text-white placeholder-slate-500 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500" />
              </div>
              <div className="md:col-span-2 flex items-center gap-3 flex-wrap">
                <button type="submit" disabled={tAdding}
                  className="flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                  <UserPlus size={15} /> {tAdding ? 'Creating…' : 'Create Account'}
                </button>
                {tError && (
                  <span className="flex items-center gap-1.5 text-sm text-red-400">
                    <AlertTriangle size={13} /> {tError}
                  </span>
                )}
                {tSuccess && (
                  <span className="flex items-center gap-1.5 text-sm text-emerald-400">
                    <CheckCircle2 size={13} /> {tSuccess}
                  </span>
                )}
              </div>
            </form>
          </div>

          {/* Therapist accounts table */}
          <div className="rounded-xl bg-slate-800/60 border border-slate-700 overflow-hidden">
            {therapists.length === 0 ? (
              <div className="p-10 text-center text-slate-500 text-sm">
                No therapist accounts yet. Create one above.
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Therapist</th>
                    <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider hidden md:table-cell">Email</th>
                    <th className="text-center px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Status</th>
                    <th className="text-center px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider hidden lg:table-cell">Created</th>
                    <th className="text-right px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {therapists.map(t => (
                    <Fragment key={t.id}>
                      <tr className="border-b border-slate-700/50 hover:bg-slate-700/20">
                        <td className="px-4 py-3">
                          <p className="text-white font-medium">{t.name}</p>
                          <p className="text-xs text-slate-500 md:hidden">{t.email}</p>
                        </td>
                        <td className="px-4 py-3 text-slate-300 text-sm hidden md:table-cell">{t.email}</td>
                        <td className="px-4 py-3 text-center">
                          <span className={`inline-flex items-center gap-1 text-xs px-2.5 py-0.5 rounded-full font-semibold ${t.isActive ? 'bg-emerald-900/30 text-emerald-400' : 'bg-red-900/30 text-red-400'}`}>
                            {t.isActive ? <CheckCircle2 size={11} /> : <XCircle size={11} />}
                            {t.isActive ? 'Active' : 'Inactive'}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-center text-xs text-slate-400 hidden lg:table-cell">
                          {new Date(t.createdAt).toLocaleDateString()}
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center justify-end gap-1">
                            <button onClick={() => void toggleTherapist(t.id)}
                              title={t.isActive ? 'Deactivate account' : 'Activate account'}
                              className={`p-1.5 rounded-lg transition-colors ${t.isActive ? 'text-slate-400 hover:text-amber-400 hover:bg-amber-600/10' : 'text-slate-400 hover:text-emerald-400 hover:bg-emerald-600/10'}`}>
                              {t.isActive ? <XCircle size={13} /> : <CheckCircle2 size={13} />}
                            </button>
                            <button onClick={() => { setResetingId(resetingId === t.id ? null : t.id); setResetNewPw('') }}
                              title="Reset password"
                              className="p-1.5 rounded-lg text-slate-400 hover:text-sky-400 hover:bg-sky-600/10 transition-colors">
                              <RefreshCw size={13} />
                            </button>
                            <button onClick={() => setDeleteConfirmId(deleteConfirmId === t.id ? null : t.id)}
                              title="Delete therapist"
                              className="p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-600/10 transition-colors">
                              <Trash2 size={13} />
                            </button>
                          </div>
                        </td>
                      </tr>

                      {/* Reset password inline row */}
                      {resetingId === t.id && (
                        <tr className="border-b border-slate-700/50 bg-sky-950/20">
                          <td colSpan={5} className="px-4 py-3">
                            <div className="flex items-center gap-3 flex-wrap">
                              <p className="text-xs text-sky-300 font-medium">Reset password for <span className="text-white">{t.name}</span></p>
                              <input type="password" value={resetNewPw} onChange={e => setResetNewPw(e.target.value)}
                                placeholder="New password (min. 6 chars)" minLength={6}
                                className="bg-slate-900 border border-slate-600 text-white placeholder-slate-500 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500 flex-1 max-w-xs" />
                              <button
                                onClick={() => { if (resetNewPw.length >= 6) { void resetTherapistPassword(t.id, resetNewPw); setResetingId(null); setResetNewPw('') } }}
                                disabled={resetNewPw.length < 6}
                                className="bg-sky-700 hover:bg-sky-600 disabled:opacity-50 text-white px-3 py-1.5 rounded-lg text-xs font-medium transition-colors">
                                Save New Password
                              </button>
                              <button onClick={() => setResetingId(null)}
                                className="text-slate-500 hover:text-white text-xs transition-colors">Cancel</button>
                            </div>
                          </td>
                        </tr>
                      )}

                      {/* Delete confirmation inline row */}
                      {deleteConfirmId === t.id && (
                        <tr className="border-b border-slate-700/50 bg-red-950/20">
                          <td colSpan={5} className="px-4 py-3">
                            <div className="flex items-center gap-3 flex-wrap">
                              <AlertTriangle size={14} className="text-red-400 flex-shrink-0" />
                              <p className="text-sm text-red-300 flex-1">
                                Delete <strong className="text-white">{t.name}</strong>'s account permanently? They will no longer be able to log in.
                              </p>
                              <button
                                onClick={() => { void deleteTherapist(t.id); setDeleteConfirmId(null) }}
                                className="bg-red-700 hover:bg-red-600 text-white px-3 py-1.5 rounded-lg text-xs font-medium transition-colors">
                                Confirm Delete
                              </button>
                              <button onClick={() => setDeleteConfirmId(null)}
                                className="text-slate-500 hover:text-white text-xs transition-colors">Cancel</button>
                            </div>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function StatCard({ label, value, color, icon }: { label: string; value: number; color: string; icon: React.ReactNode }) {
  return (
    <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-4 flex items-center gap-3">
      <div className="w-9 h-9 rounded-lg bg-slate-700 flex items-center justify-center flex-shrink-0">
        <span className={color}>{icon}</span>
      </div>
      <div>
        <p className={`text-xl font-bold ${color}`}>{value}</p>
        <p className="text-xs text-slate-400">{label}</p>
      </div>
    </div>
  )
}

function EndpointRow({ label, path }: { label: string; path: string }) {
  const [ok, setOk] = useState<boolean | null>(null)
  useEffect(() => {
    fetch(path).then(r => setOk(r.ok)).catch(() => setOk(false))
  }, [path])
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-slate-400">{label}</span>
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-600 font-mono hidden sm:block">{path}</span>
        {ok === null
          ? <div className="w-2 h-2 rounded-full bg-slate-600 animate-pulse" />
          : ok ? <CheckCircle2 size={14} className="text-emerald-400" />
               : <XCircle size={14} className="text-red-400" />
        }
      </div>
    </div>
  )
}
