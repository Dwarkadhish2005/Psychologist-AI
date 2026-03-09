import { useEffect, useState } from 'react'
import { UserPlus, Trash2, Video, BarChart3, Lock, Unlock, KeyRound, X, Info } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { UserProfile } from '../types'

// ─── per-user PIN manager ────────────────────────────────────────────────────

function PinManager({ userId }: { userId: string }) {
  const [hasPin, setHasPin] = useState<boolean | null>(null)
  const [open, setOpen] = useState(false)
  const [pin, setPin] = useState('')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)

  useEffect(() => {
    fetch(`/api/users/${userId}/pin/status`)
      .then(r => r.json())
      .then(d => setHasPin(d.has_pin))
      .catch(() => setHasPin(false))
  }, [userId])

  async function handleSet(e: React.FormEvent) {
    e.preventDefault()
    if (pin.length < 4) { setMsg('PIN must be 4–8 digits'); return }
    setBusy(true); setMsg(null)
    const res = await fetch(`/api/users/${userId}/pin`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pin }),
    })
    const data = await res.json()
    if (res.ok) { setHasPin(true); setOpen(false); setPin(''); setMsg(null) }
    else setMsg(data.detail ?? 'Error setting PIN')
    setBusy(false)
  }

  async function handleRemove() {
    if (!confirm('Remove PIN for this user?')) return
    setBusy(true)
    await fetch(`/api/users/${userId}/pin`, { method: 'DELETE' })
    setHasPin(false); setOpen(false); setMsg(null); setBusy(false)
  }

  if (hasPin === null) return null

  return (
    <div className="relative">
      <button
        onClick={() => { setOpen(o => !o); setMsg(null) }}
        title={hasPin ? 'PIN is set — click to manage' : 'Set a PIN lock'}
        className={`p-2 rounded-lg transition-colors ${
          hasPin
            ? 'bg-amber-900/30 text-amber-400 hover:bg-amber-900/60'
            : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
        }`}
      >
        {hasPin ? <Lock size={16} /> : <Unlock size={16} />}
      </button>

      {open && (
        <div className="absolute right-0 top-10 z-20 w-64 rounded-xl bg-slate-800 border border-slate-700 shadow-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-semibold text-white flex items-center gap-1.5">
              <KeyRound size={14} className="text-indigo-400" />
              {hasPin ? 'Change / Remove PIN' : 'Set PIN Lock'}
            </p>
            <button onClick={() => setOpen(false)} className="text-slate-500 hover:text-slate-300">
              <X size={14} />
            </button>
          </div>

          <form onSubmit={handleSet} className="space-y-2">
            <input
              type="password"
              inputMode="numeric"
              pattern="[0-9]{4,8}"
              maxLength={8}
              placeholder="4–8 digit PIN"
              value={pin}
              onChange={e => setPin(e.target.value.replace(/\D/g, ''))}
              className="w-full bg-slate-900 border border-slate-600 text-white placeholder-slate-500 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
            {msg && <p className="text-red-400 text-xs">{msg}</p>}
            <div className="flex gap-2">
              <button
                type="submit"
                disabled={busy}
                className="flex-1 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white text-xs py-1.5 rounded-lg font-medium"
              >
                {hasPin ? 'Update' : 'Set PIN'}
              </button>
              {hasPin && (
                <button
                  type="button"
                  disabled={busy}
                  onClick={handleRemove}
                  className="flex-1 bg-red-700/60 hover:bg-red-700 disabled:opacity-60 text-red-300 text-xs py-1.5 rounded-lg font-medium"
                >
                  Remove
                </button>
              )}
            </div>
          </form>
        </div>
      )}
    </div>
  )
}

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
                  <PinManager userId={u.user_id} />
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
