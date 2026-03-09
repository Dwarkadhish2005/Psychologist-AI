import { createContext, useContext, useState, useEffect, useCallback } from 'react'

// ── Hardcoded Admin credentials ───────────────────────────────────────────────
const ADMIN_CREDS = {
  email: 'dwarkadhish@admin.com',
  password: 'dwarka@1234',
  name: 'Dwarkadhish',
}

// ── localStorage key (session only — no therapist data here anymore) ──────────
const S_KEY = 'psych_auth_session_v1'

// ── Public types ─────────────────────────────────────────────────────────────
export interface TherapistAccount {
  id: string
  name: string
  email: string
  createdAt: string
  isActive: boolean
  // password is never sent by the API — only used for create/login forms
}

export interface AuthUser {
  role: 'admin' | 'therapist' | 'user'
  name: string
  email?: string
  therapistId?: string
  userId?: string
}

interface AuthContextValue {
  user: AuthUser | null
  loginAdmin(email: string, pw: string): { ok: boolean; error?: string }
  loginTherapist(email: string, pw: string): Promise<{ ok: boolean; error?: string }>
  loginUser(userId: string, name: string): void
  logout(): void
  // Therapist management (admin-only)
  therapists: TherapistAccount[]
  therapistsLoading: boolean
  refreshTherapists(): Promise<void>
  addTherapist(name: string, email: string, pw: string): Promise<{ ok: boolean; error?: string }>
  toggleTherapist(id: string): Promise<void>
  deleteTherapist(id: string): Promise<void>
  resetTherapistPassword(id: string, pw: string): Promise<void>
}

const AuthContext = createContext<AuthContextValue>({} as AuthContextValue)

function safeLoad<T>(key: string, fallback: T): T {
  try { return JSON.parse(localStorage.getItem(key) ?? 'null') ?? fallback }
  catch { return fallback }
}

// ── Provider ──────────────────────────────────────────────────────────────────
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(() =>
    safeLoad<AuthUser | null>(S_KEY, null)
  )
  const [therapists, setTherapists] = useState<TherapistAccount[]>([])
  const [therapistsLoading, setTherapistsLoading] = useState(false)

  function persistUser(u: AuthUser | null) {
    setUser(u)
    if (u) localStorage.setItem(S_KEY, JSON.stringify(u))
    else localStorage.removeItem(S_KEY)
  }

  // ── Load therapists from backend ──────────────────────────────────────────
  const refreshTherapists = useCallback(async () => {
    setTherapistsLoading(true)
    try {
      const res = await fetch('/api/therapists/')
      if (res.ok) setTherapists(await res.json())
    } catch { /* API offline — keep current list */ }
    finally { setTherapistsLoading(false) }
  }, [])

  useEffect(() => { refreshTherapists() }, [refreshTherapists])

  // ── Login ─────────────────────────────────────────────────────────────────
  function loginAdmin(email: string, pw: string) {
    if (
      email.trim().toLowerCase() !== ADMIN_CREDS.email.toLowerCase() ||
      pw !== ADMIN_CREDS.password
    ) return { ok: false, error: 'Invalid admin credentials.' }
    persistUser({ role: 'admin', name: ADMIN_CREDS.name, email: ADMIN_CREDS.email })
    return { ok: true }
  }

  async function loginTherapist(email: string, pw: string): Promise<{ ok: boolean; error?: string }> {
    try {
      const res = await fetch('/api/therapists/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: '', email: email.trim().toLowerCase(), password: pw }),
      })
      if (res.ok) {
        const t: TherapistAccount = await res.json()
        persistUser({ role: 'therapist', name: t.name, email: t.email, therapistId: t.id })
        return { ok: true }
      }
      const err = await res.json()
      return { ok: false, error: err.detail ?? 'Login failed.' }
    } catch {
      return { ok: false, error: 'Cannot reach server. Is the API running?' }
    }
  }

  function loginUser(userId: string, name: string) {
    persistUser({ role: 'user', name, userId })
  }

  function logout() { persistUser(null) }

  // ── Therapist CRUD (all persisted to disk via API) ─────────────────────────
  async function addTherapist(name: string, email: string, pw: string): Promise<{ ok: boolean; error?: string }> {
    try {
      const res = await fetch('/api/therapists/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim(), email: email.trim().toLowerCase(), password: pw }),
      })
      if (res.ok) {
        await refreshTherapists()
        return { ok: true }
      }
      const err = await res.json()
      return { ok: false, error: err.detail ?? 'Failed to create account.' }
    } catch {
      return { ok: false, error: 'Cannot reach server. Is the API running?' }
    }
  }

  async function toggleTherapist(id: string) {
    await fetch(`/api/therapists/${id}/toggle`, { method: 'PATCH' }).catch(() => {})
    await refreshTherapists()
  }

  async function deleteTherapist(id: string) {
    await fetch(`/api/therapists/${id}`, { method: 'DELETE' }).catch(() => {})
    await refreshTherapists()
  }

  async function resetTherapistPassword(id: string, pw: string) {
    await fetch(`/api/therapists/${id}/password`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password: pw }),
    }).catch(() => {})
  }

  return (
    <AuthContext.Provider value={{
      user, loginAdmin, loginTherapist, loginUser, logout,
      therapists, therapistsLoading, refreshTherapists,
      addTherapist, toggleTherapist, deleteTherapist, resetTherapistPassword,
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
