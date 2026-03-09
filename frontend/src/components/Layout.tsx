import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import {
  Brain,
  Shield,
  ClipboardList,
  UserCircle,
  Activity,
  Users,
  Monitor,
  Bell,
  FileText,
  BarChart2,
  Calendar,
  TrendingUp,
  StopCircle,
  LogOut,
  Video,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'

// ── Per-role nav definitions ──────────────────────────────────────────────────
const ADMIN_NAV = [
  { to: '/admin?tab=overview',   label: 'System Health',       icon: Activity     },
  { to: '/admin?tab=users',      label: 'User Management',     icon: Users        },
  { to: '/admin?tab=therapists', label: 'Therapist Mgmt',      icon: ClipboardList },
  { to: '/admin?tab=sessions',   label: 'Session Monitoring',  icon: Monitor      },
  { to: '/admin?tab=sessions',   label: 'Force Stop Sessions', icon: StopCircle   },
  { to: '/admin?tab=alerts',     label: 'Alerts Overview',     icon: Bell         },
]

const THERAPIST_NAV = [
  { to: '/therapist',           label: 'Patient List',    icon: Users       },
  { to: '/therapist?tab=sessions', label: 'Session History', icon: Calendar  },
  { to: '/therapist?tab=analytics', label: 'Analytics',   icon: BarChart2   },
  { to: '/therapist?tab=notes', label: 'Notes',           icon: FileText    },
  { to: '/therapist?tab=alerts', label: 'Alerts',         icon: Bell        },
]

const USER_NAV = [
  { to: '/user?tab=sessions',   label: 'Own Sessions',  icon: Monitor    },
  { to: '/user?tab=analytics',  label: 'Own Analytics', icon: TrendingUp },
  { to: '/user?tab=reports',    label: 'Daily Reports', icon: Calendar   },
]

const ROLE_META = {
  admin:     { label: 'Admin Portal',     Icon: Shield,       accent: 'text-indigo-400', bg: 'bg-indigo-600/20', nav: ADMIN_NAV     },
  therapist: { label: 'Therapist Portal', Icon: ClipboardList, accent: 'text-teal-400',   bg: 'bg-teal-600/20',   nav: THERAPIST_NAV },
  user:      { label: 'User Portal',      Icon: UserCircle,   accent: 'text-emerald-400', bg: 'bg-emerald-600/20', nav: USER_NAV     },
}

function isNavActive(pathname: string, search: string, to: string) {
  const [toPath, toQuery] = to.split('?')
  if (pathname !== toPath) return false
  if (!toQuery) return true
  return search.includes(toQuery)
}

export default function Layout({ children }: { children: React.ReactNode }) {
  const { pathname, search } = useLocation()
  const navigate = useNavigate()
  const { user, logout } = useAuth()
  const [sessionActive, setSessionActive] = useState(false)

  useEffect(() => {
    const check = () =>
      fetch('/api/sessions/status')
        .then(r => r.json())
        .then(s => setSessionActive(s.is_running))
        .catch(() => {})
    check()
    const id = setInterval(check, 5000)
    return () => clearInterval(id)
  }, [])

  function handleSignOut() {
    logout()
    navigate('/')
  }

  const meta = user?.role ? ROLE_META[user.role] : null

  return (
    <div className="flex min-h-screen bg-[#0f1117]">
      {/* ── Sidebar ──────────────────────────────────────────────── */}
      <aside className="w-60 flex-shrink-0 bg-[#161b27] border-r border-slate-800 flex flex-col">

        {/* Logo + role badge */}
        <div className="flex items-center gap-2.5 px-4 py-5 border-b border-slate-800">
          <Brain className="text-indigo-400 flex-shrink-0" size={22} />
          <div className="flex-1 min-w-0">
            <span className="font-bold text-white text-sm block leading-tight">Psychologist AI</span>
            {meta ? (
              <span className={`text-xs font-medium ${meta.accent}`}>{meta.label}</span>
            ) : (
              <span className="text-indigo-400 text-xs">Phase 6</span>
            )}
          </div>
        </div>

        {/* Role icon + live indicator */}
        {meta && (
          <div className="px-4 py-3 border-b border-slate-800 flex items-center gap-3">
            <div className={`w-9 h-9 rounded-xl ${meta.bg} flex items-center justify-center flex-shrink-0`}>
              <meta.Icon size={18} className={meta.accent} />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-white text-xs font-semibold truncate">{user?.name ?? 'Unknown'}</p>
              <p className="text-slate-400 text-xs truncate">{meta.label}</p>
            </div>
            {sessionActive && (
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              </div>
            )}
          </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 py-4 px-2 overflow-y-auto">
          {meta ? (
            <div>
              <p className="px-3 mb-2 text-[10px] font-semibold tracking-widest text-slate-600 uppercase">
                Features
              </p>
              <div className="space-y-0.5">
                {meta.nav.map(({ to, label, icon: Icon }, i) => {
                  const active = isNavActive(pathname, search, to)
                  return (
                    <Link
                      key={`${to}-${i}`}
                      to={to}
                      className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors
                        ${active
                          ? `${meta.bg} ${meta.accent}`
                          : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                        }`}
                    >
                      <Icon size={16} />
                      {label}
                    </Link>
                  )
                })}
              </div>

              {/* Live analysis quick link */}
              <div className="mt-4">
                <p className="px-3 mb-2 text-[10px] font-semibold tracking-widest text-slate-600 uppercase">
                  Quick Actions
                </p>
                <Link
                  to="/live"
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors
                    ${pathname === '/live'
                      ? 'bg-indigo-600/20 text-indigo-300'
                      : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                    }`}
                >
                  <div className="relative">
                    <Video size={16} />
                    {sessionActive && (
                      <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-emerald-400 border border-[#161b27]" />
                    )}
                  </div>
                  Live Analysis
                </Link>
              </div>
            </div>
          ) : (
            <p className="px-3 text-slate-600 text-xs">No role selected</p>
          )}
        </nav>

        {/* Footer: change role */}
        <div className="px-3 py-3 border-t border-slate-800 space-y-1">
          <Link
            to="/"
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-slate-500 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <Brain size={13} />
            Change Portal
          </Link>
          <button
            onClick={handleSignOut}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-slate-500 hover:text-red-400 hover:bg-red-900/20 transition-colors"
          >
            <LogOut size={13} />
            Sign Out
          </button>
          <p className="px-3 pt-1 text-[10px] text-slate-700">API: localhost:8000</p>
        </div>
      </aside>

      {/* ── Main content ─────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto">
        {children}
      </main>
    </div>
  )
}

