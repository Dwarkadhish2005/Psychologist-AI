import { useEffect, useState, useCallback } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import { Bell, Trash2, AlertTriangle, CheckCircle, Activity, TrendingUp } from 'lucide-react'
import type { UserProfile } from '../types'

interface Alert {
  timestamp: number
  session_date: string
  type: string
  severity: number
  description: string
  context: { mental_state: string; risk_level: string }
}

function severityLabel(s: number): { label: string; color: string; bg: string } {
  if (s >= 0.75) return { label: 'Critical', color: 'text-red-400',    bg: 'bg-red-900/30 border-red-800' }
  if (s >= 0.5)  return { label: 'High',     color: 'text-orange-400', bg: 'bg-orange-900/30 border-orange-800' }
  return              { label: 'Moderate', color: 'text-amber-400',  bg: 'bg-amber-900/20 border-amber-800' }
}

function timeAgo(ts: number): string {
  const diff = (Date.now() / 1000) - ts
  if (diff < 60)   return 'Just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return new Date(ts * 1000).toLocaleDateString()
}

export default function Alerts() {
  const [searchParams] = useSearchParams()
  const [users, setUsers] = useState<UserProfile[]>([])
  const [selectedUser, setSelectedUser] = useState(searchParams.get('user') ?? '')
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch('/api/users/').then(r => r.json()).then(setUsers).catch(console.error)
  }, [])

  const loadAlerts = useCallback(() => {
    if (!selectedUser) return
    setLoading(true)
    fetch(`/api/analytics/${selectedUser}/alerts?limit=100`)
      .then(r => r.ok ? r.json() : { alerts: [], total: 0 })
      .then(d => { setAlerts(d.alerts ?? []); setTotal(d.total ?? 0) })
      .catch(() => setAlerts([]))
      .finally(() => setLoading(false))
  }, [selectedUser])

  useEffect(() => { loadAlerts() }, [loadAlerts])

  async function clearAlerts() {
    if (!selectedUser || !confirm('Clear all alerts for this user?')) return
    await fetch(`/api/analytics/${selectedUser}/alerts`, { method: 'DELETE' })
    setAlerts([])
    setTotal(0)
  }

  // Group by date
  const grouped = alerts.reduce<Record<string, Alert[]>>((acc, a) => {
    const d = a.session_date
    if (!acc[d]) acc[d] = []
    acc[d].push(a)
    return acc
  }, {})

  const criticalCount = alerts.filter(a => a.severity >= 0.75).length
  const highCount     = alerts.filter(a => a.severity >= 0.5 && a.severity < 0.75).length
  const modCount      = alerts.filter(a => a.severity < 0.5).length

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Bell size={22} className="text-amber-400" />
            Deviation Alerts
          </h1>
          <p className="text-slate-400 text-sm mt-1">Significant behavioral deviations detected during sessions</p>
        </div>

        <div className="flex items-center gap-3">
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

          {selectedUser && alerts.length > 0 && (
            <button
              onClick={clearAlerts}
              className="flex items-center gap-2 bg-red-900/30 hover:bg-red-900/60 text-red-400 border border-red-800 px-3 py-2 rounded-lg text-sm transition-colors"
            >
              <Trash2 size={14} />
              Clear all
            </button>
          )}

          {selectedUser && (
            <Link
              to={`/analytics?user=${selectedUser}`}
              className="flex items-center gap-1.5 text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 px-3 py-2 rounded-lg transition-colors"
            >
              <TrendingUp size={13} />
              Analytics
            </Link>
          )}
        </div>
      </div>

      {!selectedUser ? (
        <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-12 text-center">
          <Bell size={40} className="text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400">Select a user to view their deviation alerts</p>
        </div>
      ) : loading ? (
        <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-12 text-center">
          <p className="text-slate-500 text-sm">Loading alerts…</p>
        </div>
      ) : alerts.length === 0 ? (
        <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-12 text-center">
          <CheckCircle size={40} className="text-emerald-500 mx-auto mb-3" />
          <p className="text-white font-medium">No deviation alerts</p>
          <p className="text-slate-400 text-sm mt-1">
            Alerts are generated during live sessions when significant behavioral changes are detected.
          </p>
          <Link
            to={`/live?user=${selectedUser}`}
            className="mt-4 inline-block text-indigo-400 text-sm hover:underline"
          >
            Start a session →
          </Link>
        </div>
      ) : (
        <>
          {/* Summary stats */}
          <div className="grid grid-cols-3 gap-4">
            <div className="rounded-xl bg-red-900/20 border border-red-800 p-4 text-center">
              <p className="text-2xl font-bold text-red-400">{criticalCount}</p>
              <p className="text-xs text-slate-400 mt-0.5">Critical</p>
            </div>
            <div className="rounded-xl bg-orange-900/20 border border-orange-800 p-4 text-center">
              <p className="text-2xl font-bold text-orange-400">{highCount}</p>
              <p className="text-xs text-slate-400 mt-0.5">High</p>
            </div>
            <div className="rounded-xl bg-amber-900/20 border border-amber-800 p-4 text-center">
              <p className="text-2xl font-bold text-amber-400">{modCount}</p>
              <p className="text-xs text-slate-400 mt-0.5">Moderate</p>
            </div>
          </div>

          {/* Alerts grouped by date */}
          <div className="space-y-6">
            {Object.entries(grouped).map(([date, dayAlerts]) => (
              <div key={date}>
                <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                  {date}
                </h3>
                <div className="space-y-2">
                  {dayAlerts.map((alert, i) => {
                    const sev = severityLabel(alert.severity)
                    return (
                      <div key={i} className={`rounded-xl border p-4 ${sev.bg}`}>
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex items-start gap-3 min-w-0">
                            <AlertTriangle size={16} className={`${sev.color} flex-shrink-0 mt-0.5`} />
                            <div className="min-w-0">
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className={`font-semibold text-sm ${sev.color}`}>
                                  {alert.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                                </span>
                                <span className={`text-xs px-2 py-0.5 rounded-full border ${sev.bg} ${sev.color}`}>
                                  {sev.label} · {Math.round(alert.severity * 100)}%
                                </span>
                              </div>
                              <p className="text-slate-300 text-sm mt-1">{alert.description || '—'}</p>
                              {alert.context && (
                                <div className="flex gap-3 mt-2 text-xs text-slate-500">
                                  <span className="flex items-center gap-1">
                                    <Activity size={11} />
                                    {alert.context.mental_state?.replace(/_/g, ' ')}
                                  </span>
                                  <span className="capitalize">
                                    Risk: {alert.context.risk_level}
                                  </span>
                                </div>
                              )}
                            </div>
                          </div>
                          <span className="text-xs text-slate-500 flex-shrink-0">{timeAgo(alert.timestamp)}</span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>

          <p className="text-xs text-slate-600 text-center">
            Showing {alerts.length} of {total} total alerts
          </p>
        </>
      )}
    </div>
  )
}
