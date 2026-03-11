import { useEffect, useState } from 'react'
import { UserPlus, Trash2, Video, BarChart3, Info } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { UserProfile } from '../types'

// ─── main page ───────────────────────────────────────────────────────────────

export default function Users() {
  const [users, setUsers] = useState<UserProfile[]>([])
  const [newName, setNewName] = useState('')
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadUsers = () =>
    fetch('/api/users/')
      .then(r => r.json())
      .then(setUsers)
      .catch(console.error)
      .finally(() => setLoading(false))

  useEffect(() => { loadUsers() }, [])

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault()
    if (!newName.trim()) return
    setCreating(true)
    setError(null)
    try {
      const res = await fetch('/api/users/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newName.trim() }),
      })
      if (!res.ok) throw new Error(await res.text())
      setNewName('')
      await loadUsers()
    } catch (e: any) {
      setError(e.message)
    } finally {
      setCreating(false)
    }
  }

  async function handleDelete(user: UserProfile) {
    if (!confirm(`Delete "${user.name}" and all their data?`)) return
    await fetch(`/api/users/${user.user_id}`, { method: 'DELETE' })
    await loadUsers()
  }

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">User Management</h1>
        <p className="text-slate-400 text-sm mt-1">Register and manage analysis profiles</p>
      </div>

      {/* Register form */}
      <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-6">
        <h2 className="text-base font-semibold text-white mb-4">Register New User</h2>
        <form onSubmit={handleRegister} className="flex gap-3">
          <input
            type="text"
            placeholder="Full name…"
            value={newName}
            onChange={e => setNewName(e.target.value)}
            className="flex-1 bg-slate-900 border border-slate-600 text-white placeholder-slate-500 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
            required
          />
          <button
            type="submit"
            disabled={creating}
            className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            <UserPlus size={16} />
            {creating ? 'Registering…' : 'Register'}
          </button>
        </form>
        {error && <p className="text-red-400 text-xs mt-2">{error}</p>}
      </div>

      {/* Onboarding banner for users with 0 sessions */}
      {users.some(u => u.total_sessions === 0) && (
        <div className="flex items-start gap-3 bg-indigo-900/30 border border-indigo-700/50 rounded-xl px-4 py-3 text-sm text-indigo-300">
          <Info size={16} className="mt-0.5 flex-shrink-0 text-indigo-400" />
          <span>
            <strong>New users detected.</strong> Click the{' '}
            <span className="inline-flex items-center gap-0.5 font-semibold"><Video size={12} /> Live</span>{' '}
            button to run their first session — the AI will calibrate a baseline personality profile.
          </span>
        </div>
      )}

      {/* User list */}
      <div>
        <h2 className="text-base font-semibold text-white mb-4">
          All Users ({users.length})
        </h2>
        {loading ? (
          <p className="text-slate-500 text-sm">Loading…</p>
        ) : users.length === 0 ? (
          <div className="rounded-xl bg-slate-800/40 border border-slate-700 p-8 text-center text-slate-500 text-sm">
            No users registered yet.
          </div>
        ) : (
          <div className="space-y-3">
            {users.map(u => (
              <div
                key={u.user_id}
                className="rounded-xl bg-slate-800/60 border border-slate-700 p-4 flex items-center justify-between gap-4"
              >
                <div className="min-w-0">
                  <p className="font-semibold text-white">{u.name}</p>
                  <p className="text-xs text-slate-500 font-mono truncate">{u.user_id}</p>
                  <p className="text-xs text-slate-400 mt-1">
                    {u.total_sessions} sessions · Last active: {new Date(u.last_active).toLocaleDateString()}
                  </p>
                </div>

                <div className="flex items-center gap-2 flex-shrink-0">
                  <Link
                    to={`/live?user=${u.user_id}`}
                    title="Start live analysis"
                    className="p-2 rounded-lg bg-indigo-600/20 text-indigo-400 hover:bg-indigo-600/40 transition-colors"
                  >
                    <Video size={16} />
                  </Link>
                  <Link
                    to={`/analytics?user=${u.user_id}`}
                    title="View analytics"
                    className="p-2 rounded-lg bg-slate-700 text-slate-300 hover:bg-slate-600 transition-colors"
                  >
                    <BarChart3 size={16} />
                  </Link>
                  <button
                    onClick={() => handleDelete(u)}
                    title="Delete user"
                    className="p-2 rounded-lg bg-red-900/30 text-red-400 hover:bg-red-900/60 transition-colors"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
