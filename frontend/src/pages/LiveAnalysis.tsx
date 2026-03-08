import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Play, Square, AlertTriangle } from 'lucide-react'
import type { EmotionState, UserProfile, PSVData } from '../types'
import MentalStateCard from '../components/MentalStateCard'
import PersonalityRadar from '../components/PersonalityRadar'
import RiskBadge from '../components/RiskBadge'

export default function LiveAnalysis() {
  const [searchParams] = useSearchParams()
  const preselectedUser = searchParams.get('user') ?? ''

  const [users, setUsers] = useState<UserProfile[]>([])
  const [selectedUser, setSelectedUser] = useState(preselectedUser)
  const [isRunning, setIsRunning] = useState(false)
  const [state, setState] = useState<EmotionState | null>(null)
  const [psv, setPsv] = useState<PSVData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [explanations, setExplanations] = useState<string[]>([])

  const wsRef = useRef<WebSocket | null>(null)

  // Load users
  useEffect(() => {
    fetch('/api/users/')
      .then(r => r.json())
      .then(setUsers)
      .catch(console.error)
  }, [])

  // Check if there's already a running session
  useEffect(() => {
    fetch('/api/sessions/status')
      .then(r => r.json())
      .then(s => {
        if (s.is_running) {
          setIsRunning(true)
          if (s.active_user_id) setSelectedUser(s.active_user_id)
          connectWS()
        }
      })
      .catch(console.error)
  }, [])

  function connectWS() {
    if (wsRef.current) wsRef.current.close()
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/stream`)
    ws.onmessage = ev => {
      const data: EmotionState = JSON.parse(ev.data)
      setState(data)
      if (data.explanations?.length) {
        setExplanations(prev => {
          const combined = [...data.explanations, ...prev]
          return combined.slice(0, 12)
        })
      }
    }
    ws.onerror = () => setError('WebSocket connection failed')
    wsRef.current = ws
  }

  async function startSession() {
    if (!selectedUser) { setError('Please select a user first'); return }
    setError(null)
    setExplanations([])
    try {
      const res = await fetch('/api/sessions/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: selectedUser }),
      })
      if (!res.ok) throw new Error(await res.text())
      setIsRunning(true)
      connectWS()

      // Try to load PSV for personality radar
      fetch(`/api/analytics/${selectedUser}/psv`)
        .then(r => r.ok ? r.json() : null)
        .then(d => d && setPsv(d))
        .catch(() => {})
    } catch (e: any) {
      setError(e.message)
    }
  }

  async function stopSession() {
    await fetch('/api/sessions/stop', { method: 'POST' })
    setIsRunning(false)
    wsRef.current?.close()
    setState(null)
  }

  // Cleanup on unmount
  useEffect(() => () => { wsRef.current?.close() }, [])

  const displayRisk = state?.adjusted_risk ?? state?.risk_level ?? 'low'

  return (
    <div className="p-8 space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Live Analysis</h1>
          <p className="text-slate-400 text-sm mt-1">Real-time multi-modal psychological state detection</p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3">
          <select
            className="bg-slate-800 border border-slate-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            value={selectedUser}
            onChange={e => setSelectedUser(e.target.value)}
            disabled={isRunning}
          >
            <option value="">— Select user —</option>
            {users.map(u => (
              <option key={u.user_id} value={u.user_id}>{u.name}</option>
            ))}
          </select>

          {!isRunning ? (
            <button
              onClick={startSession}
              className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              <Play size={16} /> Start Session
            </button>
          ) : (
            <button
              onClick={stopSession}
              className="flex items-center gap-2 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              <Square size={16} /> Stop Session
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 text-red-300 text-sm">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {/* Main grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Video feed — 2 cols wide */}
        <div className="xl:col-span-2 space-y-4">
          <div className="rounded-xl overflow-hidden bg-slate-900 border border-slate-700 aspect-video relative flex items-center justify-center">
            {isRunning ? (
              <img
                src="http://localhost:8000/api/video/feed"
                alt="Live feed"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-center text-slate-500 space-y-2">
                <div className="text-5xl">📷</div>
                <p className="text-sm">Select a user and press <strong>Start Session</strong></p>
              </div>
            )}

            {/* Floating risk badge */}
            {isRunning && state && state.status !== 'waiting' && (
              <div className="absolute top-3 right-3">
                <RiskBadge risk={displayRisk} size="sm" />
              </div>
            )}
          </div>

          {/* Explanations log */}
          {explanations.length > 0 && (
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-4">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Reasoning Log
              </h3>
              <ul className="space-y-1">
                {explanations.slice(0, 8).map((e, i) => (
                  <li key={i} className="text-xs text-slate-300 flex items-start gap-1.5">
                    <span className="text-indigo-400 mt-0.5">›</span>
                    {e}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Right panel */}
        <div className="space-y-4">
          <MentalStateCard state={state} />

          {/* Personality radar */}
          {psv && (
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-4">
              <h3 className="text-sm font-semibold text-white mb-3">Personality Profile</h3>
              <PersonalityRadar psv={psv} />
            </div>
          )}

          {/* Deviations */}
          {state?.deviations && state.deviations.length > 0 && (
            <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-4 space-y-2">
              <h3 className="text-sm font-semibold text-white">⚠ Deviations Detected</h3>
              {state.deviations.map((d, i) => (
                <div key={i} className="flex items-center justify-between">
                  <span className="text-xs text-slate-300 capitalize">
                    {d.type.replace(/_/g, ' ')}
                  </span>
                  <span
                    className={`text-xs font-semibold ${
                      d.severity > 0.7 ? 'text-red-400' :
                      d.severity > 0.4 ? 'text-orange-400' : 'text-yellow-400'
                    }`}
                  >
                    {Math.round(d.severity * 100)}%
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Risk reason */}
          {state?.risk_adjustment_reason && (
            <div className="rounded-xl bg-slate-800/40 border border-slate-700 p-4">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
                Risk Adjustment
              </h3>
              <p className="text-xs text-slate-300">{state.risk_adjustment_reason}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
