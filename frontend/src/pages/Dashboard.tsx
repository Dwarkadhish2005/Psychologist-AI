import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Activity, Users, Calendar, TrendingUp, Video } from 'lucide-react'
import type { UserProfile } from '../types'
import RiskBadge from '../components/RiskBadge'

export default function Dashboard() {
  const [users, setUsers] = useState<UserProfile[]>([])
  const [summaries, setSummaries] = useState<Record<string, any>>({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/users/')
      .then(r => r.json())
      .then(async (data: UserProfile[]) => {
        setUsers(data)
        // Load summaries for each user in parallel
        const entries = await Promise.all(
          data.map(async u => {
            try {
              const res = await fetch(`/api/analytics/${u.user_id}/summary`)
              if (res.ok) return [u.user_id, await res.json()]
            } catch { /* ignore */ }
            return [u.user_id, null]
          })
        )
        setSummaries(Object.fromEntries(entries))
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  const totalSessions = users.reduce((a, u) => a + u.total_sessions, 0)
  const activeDays = Object.values(summaries).reduce(
    (a: number, s: any) => a + (s?.days_tracked ?? 0), 0
  )

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-slate-400 text-sm mt-1">Psychologist AI · Multi-modal behavioral analysis</p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Users}    label="Users"           value={users.length}    color="text-indigo-400" />
        <StatCard icon={Activity} label="Total Sessions"  value={totalSessions}   color="text-emerald-400" />
        <StatCard icon={Calendar} label="Days Tracked"    value={activeDays}      color="text-sky-400" />
        <StatCard icon={TrendingUp} label="Phases Active" value={5}               color="text-purple-400" />
      </div>

      {/* Quick actions */}
      <div className="flex flex-wrap gap-3">
        <Link
          to="/live"
          className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
        >
          <Video size={16} />
          Start Live Analysis
        </Link>
        <Link
          to="/users"
          className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
        >
          <Users size={16} />
          Manage Users
        </Link>
        <Link
          to="/analytics"
          className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
        >
          <TrendingUp size={16} />
          View Analytics
        </Link>
      </div>

      {/* User cards */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-4">Registered Users</h2>
        {loading ? (
          <p className="text-slate-500 text-sm">Loading…</p>
        ) : users.length === 0 ? (
          <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-8 text-center">
            <p className="text-slate-400">No users yet. Go to <Link to="/users" className="text-indigo-400 underline">Users</Link> to register one.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {users.map(u => (
              <UserCard key={u.user_id} user={u} summary={summaries[u.user_id]} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function StatCard({
  icon: Icon, label, value, color,
}: { icon: any; label: string; value: number; color: string }) {
  return (
    <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5 flex items-center gap-4">
      <div className={`${color} bg-slate-700/60 rounded-lg p-2`}>
        <Icon size={22} />
      </div>
      <div>
        <p className="text-2xl font-bold text-white">{value}</p>
        <p className="text-xs text-slate-400">{label}</p>
      </div>
    </div>
  )
}

function UserCard({ user, summary }: { user: UserProfile; summary: any }) {
  const lastActive = new Date(user.last_active).toLocaleDateString()
  const risk = summary?.latest_risk_avg

  const riskStr = risk == null
    ? 'low'
    : risk < 1 ? 'low' : risk < 2 ? 'moderate' : risk < 3 ? 'high' : 'critical'

  return (
    <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5 space-y-3">
      <div className="flex items-start justify-between">
        <div>
          <p className="font-semibold text-white">{user.name}</p>
          <p className="text-xs text-slate-500 font-mono mt-0.5">{user.user_id.split('_').slice(-1)[0]}</p>
        </div>
        {summary && <RiskBadge risk={riskStr} size="sm" />}
      </div>
      <div className="flex gap-4 text-xs text-slate-400">
        <span>{user.total_sessions} sessions</span>
        <span>{summary?.days_tracked ?? 0} days</span>
        <span>Last: {lastActive}</span>
      </div>
      <div className="flex gap-2">
        <Link
          to={`/live?user=${user.user_id}`}
          className="flex-1 text-center text-xs bg-indigo-600/20 text-indigo-300 hover:bg-indigo-600/40 px-2 py-1.5 rounded-lg transition-colors"
        >
          Live
        </Link>
        <Link
          to={`/analytics?user=${user.user_id}`}
          className="flex-1 text-center text-xs bg-slate-700 text-slate-300 hover:bg-slate-600 px-2 py-1.5 rounded-lg transition-colors"
        >
          Analytics
        </Link>
      </div>
    </div>
  )
}
