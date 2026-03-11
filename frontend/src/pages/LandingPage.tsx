import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Brain, Shield, ClipboardList, UserCircle,
  Eye, EyeOff, AlertCircle, ArrowRight, RefreshCw,
  Activity, Users, Monitor, Bell, StopCircle,
  Calendar, BarChart2, FileText, TrendingUp, UserPlus, X, CheckCircle,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'

// â”€â”€ Portal definitions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
type Role = 'admin' | 'therapist' | 'user'

interface PortalDef {
  role: Role
  title: string
  subtitle: string
  Icon: React.ElementType
  gradient: string
  ring: string
  activeRing: string
  iconBg: string
  iconColor: string
  btnClass: string
  features: string[]
}

const PORTALS: PortalDef[] = [
  {
    role: 'admin',
    title: 'Admin Portal',
    subtitle: 'Full system control & oversight',
    Icon: Shield,
    gradient: 'from-indigo-950/70 via-slate-900/80 to-slate-950',
    ring: 'border-indigo-800/40 hover:border-indigo-600/70',
    activeRing: 'border-indigo-500 ring-2 ring-indigo-500/20',
    iconBg: 'bg-indigo-600/20',
    iconColor: 'text-indigo-400',
    btnClass: 'bg-indigo-600 hover:bg-indigo-500',
    features: ['System Health', 'User Management', 'Therapist Management', 'Session Monitoring', 'Alerts Overview'],
  },
  {
    role: 'therapist',
    title: 'Therapist Portal',
    subtitle: 'Patient management & insights',
    Icon: ClipboardList,
    gradient: 'from-teal-950/70 via-slate-900/80 to-slate-950',
    ring: 'border-teal-800/40 hover:border-teal-600/70',
    activeRing: 'border-teal-500 ring-2 ring-teal-500/20',
    iconBg: 'bg-teal-600/20',
    iconColor: 'text-teal-400',
    btnClass: 'bg-teal-600 hover:bg-teal-500',
    features: ['Patient List', 'Session History', 'Analytics', 'Notes', 'Alerts'],
  },
  {
    role: 'user',
    title: 'User Portal',
    subtitle: 'Your personal wellness dashboard',
    Icon: UserCircle,
    gradient: 'from-emerald-950/70 via-slate-900/80 to-slate-950',
    ring: 'border-emerald-800/40 hover:border-emerald-600/70',
    activeRing: 'border-emerald-500 ring-2 ring-emerald-500/20',
    iconBg: 'bg-emerald-600/20',
    iconColor: 'text-emerald-400',
    btnClass: 'bg-emerald-600 hover:bg-emerald-500',
    features: ['Own Sessions', 'Own Analytics', 'Daily Reports'],
  },
]

// â”€â”€ Feature icon map â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const FEAT_ICONS: Record<string, React.ElementType> = {
  'System Health': Activity, 'User Management': Users, 'Therapist Management': ClipboardList,
  'Session Monitoring': Monitor, 'Force Stop Sessions': StopCircle, 'Alerts Overview': Bell,
  'Patient List': Users, 'Session History': Calendar, 'Analytics': BarChart2,
  'Notes': FileText, 'Alerts': Bell, 'Own Sessions': Monitor,
  'Own Analytics': TrendingUp, 'Daily Reports': Calendar,
}

// â”€â”€ Component â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export default function LandingPage() {
  const { user, loginAdmin, loginTherapist, loginUserWithCredentials, registerNewUser } = useAuth()
  const navigate = useNavigate()

  const [selectedRole, setSelectedRole] = useState<Role | null>(null)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  // Register modal state
  const [showRegister, setShowRegister] = useState(false)
  const [regName, setRegName] = useState('')
  const [regEmail, setRegEmail] = useState('')
  const [regPassword, setRegPassword] = useState('')
  const [regShowPw, setRegShowPw] = useState(false)
  const [regError, setRegError] = useState('')
  const [regLoading, setRegLoading] = useState(false)
  const [regSuccess, setRegSuccess] = useState(false)

  // Redirect if already authenticated
  useEffect(() => {
    if (!user) return
    if (user.role === 'admin') navigate('/admin', { replace: true })
    else if (user.role === 'therapist') navigate('/therapist', { replace: true })
    else navigate('/user', { replace: true })
  }, [user, navigate])

  function selectRole(role: Role) {
    setSelectedRole(role)
    setEmail(''); setPassword(''); setError('')
  }

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)
    await new Promise(r => setTimeout(r, 150))

    if (selectedRole === 'admin') {
      const res = loginAdmin(email, password)
      if (!res.ok) setError(res.error ?? 'Login failed.')
    } else if (selectedRole === 'therapist') {
      const res = await loginTherapist(email, password)
      if (!res.ok) setError(res.error ?? 'Login failed.')
    } else if (selectedRole === 'user') {
      const res = await loginUserWithCredentials(email, password)
      if (!res.ok) setError(res.error ?? 'Login failed.')
    }
    setLoading(false)
  }

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault()
    setRegError('')
    setRegLoading(true)
    const res = await registerNewUser(regName, regEmail, regPassword)
    if (res.ok) {
      setRegSuccess(true)
    } else {
      setRegError(res.error ?? 'Registration failed.')
    }
    setRegLoading(false)
  }

  function openRegister() {
    setRegName(''); setRegEmail(''); setRegPassword('')
    setRegError(''); setRegSuccess(false); setRegShowPw(false)
    setShowRegister(true)
  }

  const portal = selectedRole ? PORTALS.find(p => p.role === selectedRole)! : null

  return (
    <div className="min-h-screen bg-[#0a0d14] flex flex-col">
      {/* Background glows */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 left-1/2 -translate-x-1/2 w-[900px] h-[600px] rounded-full bg-indigo-900/20 blur-[120px]" />
        <div className="absolute bottom-0 -right-40 w-[600px] h-[400px] rounded-full bg-teal-900/15 blur-[100px]" />
        <div className="absolute bottom-0 -left-40 w-[600px] h-[400px] rounded-full bg-emerald-900/15 blur-[100px]" />
      </div>

      {/* Header */}
      <header className="relative z-10 flex items-center gap-3 px-8 py-5 border-b border-white/5">
        <div className="w-9 h-9 rounded-xl bg-indigo-600/20 flex items-center justify-center">
          <Brain size={20} className="text-indigo-400" />
        </div>
        <div>
          <span className="text-white font-bold text-lg leading-none">Psychologist AI</span>
          <span className="block text-indigo-400 text-xs">Phase 6 Â· Multi-modal Behavioral Analysis</span>
        </div>
        <span className="ml-auto text-xs text-slate-700 font-mono">v6.0.0</span>
      </header>

      {/* Main */}
      <div className="relative z-10 flex-1 flex flex-col items-center px-4 pt-10 pb-12">
        {/* Hero text */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 bg-indigo-500/10 border border-indigo-500/20 rounded-full px-4 py-1.5 mb-5 text-xs text-indigo-300 font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
            {selectedRole ? `Logging in as ${portal?.title}` : 'Select your role to continue'}
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold text-white tracking-tight mb-3">
            {selectedRole
              ? <>Sign in to <span className={`text-transparent bg-clip-text bg-gradient-to-r ${selectedRole === 'admin' ? 'from-indigo-400 to-purple-400' : selectedRole === 'therapist' ? 'from-teal-400 to-cyan-400' : 'from-emerald-400 to-green-400'}`}>{portal?.title}</span></>
              : <>Who are you <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-teal-400">logging in as?</span></>
            }
          </h1>
          <p className="text-slate-400 text-base max-w-lg mx-auto">
            {selectedRole ? 'Enter your credentials to access the portal.' : 'Choose your role to access the features designed for you.'}
          </p>
        </div>

        {/* Role selection cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full max-w-4xl mb-8">
          {PORTALS.map(p => {
            const isActive = selectedRole === p.role
            const isDimmed = selectedRole && !isActive
            return (
              <div key={p.role} className="flex flex-col gap-2">
                <button
                  onClick={() => selectRole(p.role)}
                  className={`text-left rounded-2xl border bg-gradient-to-b ${p.gradient} p-5 transition-all duration-200 ${isActive ? p.activeRing : p.ring} ${isDimmed ? 'opacity-40' : ''}`}
                >
                  <div className={`w-10 h-10 rounded-xl ${p.iconBg} flex items-center justify-center mb-3 transition-transform ${isActive ? 'scale-110' : ''}`}>
                    <p.Icon size={20} className={p.iconColor} />
                  </div>
                  <p className={`font-bold text-sm mb-0.5 ${isActive ? 'text-white' : 'text-slate-200'}`}>{p.title}</p>
                  <p className="text-slate-400 text-xs mb-3">{p.subtitle}</p>
                  <ul className="space-y-1.5">
                    {p.features.slice(0, 3).map(f => {
                      const FIcon = FEAT_ICONS[f] ?? Activity
                      return (
                        <li key={f} className="flex items-center gap-2 text-xs text-slate-500">
                          <FIcon size={11} className={isActive ? p.iconColor : 'text-slate-600'} />
                          {f}
                        </li>
                      )
                    })}
                    {p.features.length > 3 && <li className="text-xs text-slate-700">+{p.features.length - 3} more</li>}
                  </ul>
                </button>
                {p.role === 'user' && (
                  <button
                    onClick={openRegister}
                    className="flex items-center justify-center gap-2 rounded-xl border border-emerald-800/40 hover:border-emerald-600/70 bg-emerald-950/30 hover:bg-emerald-900/30 text-emerald-400 hover:text-emerald-300 text-xs font-medium py-2.5 px-4 transition-all duration-200"
                  >
                    <UserPlus size={13} />
                    Register New User
                  </button>
                )}
              </div>
            )
          })}
        </div>

        {/* Login form */}
        {selectedRole && portal && (
          <div className={`w-full max-w-md rounded-2xl border ${portal.activeRing} bg-[#13192a] p-7 shadow-2xl`}>
            <div className="flex items-center gap-3 mb-6">
              <div className={`w-10 h-10 rounded-xl ${portal.iconBg} flex items-center justify-center`}>
                <portal.Icon size={20} className={portal.iconColor} />
              </div>
              <div>
                <p className="text-white font-bold">{portal.title}</p>
                <p className="text-slate-400 text-xs">Enter your credentials below</p>
              </div>
            </div>

            <form onSubmit={handleLogin} className="space-y-4">
              <div>
                <label className="block text-xs text-slate-400 mb-1.5 font-medium">Email address</label>
                <input
                  type="email" value={email} autoFocus
                  onChange={e => { setEmail(e.target.value); setError('') }}
                  placeholder={selectedRole === 'admin' ? 'Enter admin email' : selectedRole === 'therapist' ? 'therapist@clinic.com' : 'your@email.com'}
                  required
                  className="w-full bg-slate-900 border border-slate-700 text-white placeholder-slate-600 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500 transition-colors"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1.5 font-medium">Password</label>
                <div className="relative">
                  <input
                    type={showPw ? 'text' : 'password'} value={password}
                    onChange={e => { setPassword(e.target.value); setError('') }}
                    placeholder="Enter password"
                    required
                    className="w-full bg-slate-900 border border-slate-700 text-white placeholder-slate-600 rounded-xl px-4 py-3 pr-12 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500 transition-colors"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPw(v => !v)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
                  >
                    {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              {error && (
                <div className="flex items-center gap-2 text-sm text-red-400 bg-red-900/20 border border-red-800 rounded-xl px-4 py-3">
                  <AlertCircle size={15} className="flex-shrink-0" /> {error}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold text-white transition-all ${portal.btnClass} disabled:opacity-60`}
              >
                {loading
                  ? <><RefreshCw size={15} className="animate-spin" /> Signing in...</>
                  : <><ArrowRight size={15} /> Enter {portal.title}</>
                }
              </button>

              {selectedRole === 'user' && (
                <button
                  type="button"
                  onClick={openRegister}
                  className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-xs font-medium text-emerald-400 hover:text-emerald-300 border border-emerald-800/40 hover:border-emerald-600/70 bg-emerald-950/20 hover:bg-emerald-900/20 transition-all"
                >
                  <UserPlus size={13} /> New here? Register an account
                </button>
              )}
            </form>

            <button
              onClick={() => setSelectedRole(null)}
              className="w-full mt-4 text-xs text-slate-700 hover:text-slate-400 transition-colors"
            >
              Back to role selection
            </button>
          </div>
        )}

        {/* Register Modal */}
        {showRegister && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
            <div className="w-full max-w-md rounded-2xl border border-emerald-800/50 bg-[#0f1620] p-7 shadow-2xl">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-emerald-600/20 flex items-center justify-center">
                    <UserPlus size={20} className="text-emerald-400" />
                  </div>
                  <div>
                    <p className="text-white font-bold">Register New User</p>
                    <p className="text-slate-400 text-xs">Create a new patient account</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowRegister(false)}
                  className="text-slate-600 hover:text-slate-300 transition-colors"
                >
                  <X size={18} />
                </button>
              </div>

              {regSuccess ? (
                <div className="text-center py-6">
                  <div className="w-14 h-14 rounded-full bg-emerald-500/20 flex items-center justify-center mx-auto mb-4">
                    <CheckCircle size={28} className="text-emerald-400" />
                  </div>
                  <p className="text-white font-semibold mb-1">Account created!</p>
                  <p className="text-slate-400 text-sm mb-5">You can now log in with your email and password.</p>
                  <button
                    onClick={() => { setShowRegister(false); setSelectedRole('user') }}
                    className="flex items-center justify-center gap-2 mx-auto px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium transition-colors"
                  >
                    <ArrowRight size={14} /> Go to Login
                  </button>
                </div>
              ) : (
                <form onSubmit={handleRegister} className="space-y-4">
                  <div>
                    <label className="block text-xs text-slate-400 mb-1.5 font-medium">Full name</label>
                    <input
                      type="text" value={regName} autoFocus
                      onChange={e => { setRegName(e.target.value); setRegError('') }}
                      placeholder="Your name"
                      required
                      className="w-full bg-slate-900 border border-slate-700 text-white placeholder-slate-600 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/40 focus:border-emerald-500 transition-colors"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1.5 font-medium">Email address</label>
                    <input
                      type="email" value={regEmail}
                      onChange={e => { setRegEmail(e.target.value); setRegError('') }}
                      placeholder="you@example.com"
                      required
                      className="w-full bg-slate-900 border border-slate-700 text-white placeholder-slate-600 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/40 focus:border-emerald-500 transition-colors"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1.5 font-medium">Password</label>
                    <div className="relative">
                      <input
                        type={regShowPw ? 'text' : 'password'} value={regPassword}
                        onChange={e => { setRegPassword(e.target.value); setRegError('') }}
                        placeholder="Min. 6 characters"
                        required
                        className="w-full bg-slate-900 border border-slate-700 text-white placeholder-slate-600 rounded-xl px-4 py-3 pr-12 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/40 focus:border-emerald-500 transition-colors"
                      />
                      <button
                        type="button"
                        onClick={() => setRegShowPw(v => !v)}
                        className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
                      >
                        {regShowPw ? <EyeOff size={16} /> : <Eye size={16} />}
                      </button>
                    </div>
                  </div>

                  {regError && (
                    <div className="flex items-center gap-2 text-sm text-red-400 bg-red-900/20 border border-red-800 rounded-xl px-4 py-3">
                      <AlertCircle size={15} className="flex-shrink-0" /> {regError}
                    </div>
                  )}

                  <button
                    type="submit"
                    disabled={regLoading}
                    className="w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold text-white bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 transition-all"
                  >
                    {regLoading
                      ? <><RefreshCw size={15} className="animate-spin" /> Creating account...</>
                      : <><UserPlus size={15} /> Create Account</>
                    }
                  </button>
                </form>
              )}
            </div>
          </div>
        )}

      </div>

      <footer className="relative z-10 text-center pb-5 text-xs text-slate-800">
        Psychologist AI &middot; Confidential Clinical System &middot; All sessions monitored &amp; recorded
      </footer>
    </div>
  )
}
