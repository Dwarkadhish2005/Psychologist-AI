import { Link, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Video,
  Users,
  BarChart3,
  Brain,
} from 'lucide-react'

const NAV = [
  { to: '/',          label: 'Dashboard',   icon: LayoutDashboard },
  { to: '/live',      label: 'Live',        icon: Video           },
  { to: '/users',     label: 'Users',       icon: Users           },
  { to: '/analytics', label: 'Analytics',   icon: BarChart3       },
]

export default function Layout({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation()

  return (
    <div className="flex min-h-screen bg-[#0f1117]">
      {/* Sidebar */}
      <aside className="w-56 flex-shrink-0 bg-[#161b27] border-r border-slate-800 flex flex-col">
        {/* Logo */}
        <div className="flex items-center gap-2 px-4 py-5 border-b border-slate-800">
          <Brain className="text-indigo-400" size={24} />
          <span className="font-bold text-white text-base leading-tight">
            Psychologist<br />
            <span className="text-indigo-400 text-xs font-normal">AI · Phase 6</span>
          </span>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-4 space-y-1 px-2">
          {NAV.map(({ to, label, icon: Icon }) => {
            const active = pathname === to
            return (
              <Link
                key={to}
                to={to}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors
                  ${active
                    ? 'bg-indigo-600/20 text-indigo-300'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                  }`}
              >
                <Icon size={18} />
                {label}
              </Link>
            )
          })}
        </nav>

        {/* Footer hint */}
        <div className="px-4 py-3 border-t border-slate-800 text-xs text-slate-600">
          API: localhost:8000
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        {children}
      </main>
    </div>
  )
}
