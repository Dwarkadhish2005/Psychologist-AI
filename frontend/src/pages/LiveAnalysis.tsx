import { useEffect, useRef, useState } from 'react'
import { Play, Square, AlertTriangle } from 'lucide-react'
import type { EmotionState, PSVData } from '../types'
import MentalStateCard from '../components/MentalStateCard'
import PersonalityRadar from '../components/PersonalityRadar'
import RiskBadge from '../components/RiskBadge'
import { useAuth } from '../context/AuthContext'

function buildApiUrl(path: string): string {
  const rawBase = import.meta.env.VITE_API_BASE_URL as string | undefined
  const base = rawBase?.trim().replace(/\/$/, '')
  return base ? `${base}${path}` : path
}

function buildWsUrl(path: string): string {
  const rawBase = import.meta.env.VITE_API_BASE_URL as string | undefined
  const base = rawBase?.trim().replace(/\/$/, '')
  if (base) {
    const api = new URL(base)
    const wsProtocol = api.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${wsProtocol}//${api.host}${path}`
  }
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${wsProtocol}//${window.location.host}${path}`
}

export default function LiveAnalysis() {
  const { user: authUser } = useAuth()
  const selectedUser = authUser?.userId ?? ''

  const [isRunning, setIsRunning] = useState(false)
  const [state, setState] = useState<EmotionState | null>(null)
  const [psv, setPsv] = useState<PSVData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [explanations, setExplanations] = useState<string[]>([])

  const wsRef = useRef<WebSocket | null>(null)
  const pollingRef = useRef<number | null>(null)
  const lastWsUpdateRef = useRef(0)
  const isRunningRef = useRef(false)

  // Keep isRunningRef in sync for WS reconnect closure
  useEffect(() => { isRunningRef.current = isRunning }, [isRunning])

  // Check if there's already a running session
  useEffect(() => {
    fetch('/api/sessions/status')
      .then(r => r.json())
      .then(s => {
        if (s.is_running) {
          setIsRunning(true)
          connectWS()
        }
      })
      .catch(console.error)
  }, [])

  function stopPolling() {
    if (pollingRef.current !== null) {
      window.clearInterval(pollingRef.current)
      pollingRef.current = null
    }
  }

  function applyIncomingState(data: EmotionState) {
    setState(data)
    if (data.explanations?.length) {
      setExplanations(prev => [...data.explanations, ...prev].slice(0, 12))
    }
  }

  function startPollingFallback() {
    if (pollingRef.current !== null) return
    pollingRef.current = window.setInterval(async () => {
      try {
        const res = await fetch(buildApiUrl('/api/stream/state'))
        if (!res.ok) return
        const data: EmotionState = await res.json()
        applyIncomingState(data)
      } catch {
        // Keep retrying while session is running
      }
    }, 500)
  }

  function connectWS() {
    stopPolling()
    if (wsRef.current) wsRef.current.close()
    const wsUrl = buildWsUrl('/ws/stream')
    const ws = new WebSocket(wsUrl)
    ws.onmessage = ev => {
      // Throttle to max 5 updates/sec to prevent UI freeze
      const now = Date.now()
      if (now - lastWsUpdateRef.current < 200) return
      lastWsUpdateRef.current = now
      const data: EmotionState = JSON.parse(ev.data)
      applyIncomingState(data)
    }
    ws.onerror = () => {
      setError(`WebSocket connection failed (${wsUrl}), using HTTP fallback`)
      startPollingFallback()
    }
    ws.onclose = () => {
      // Auto-reconnect if session is still running
      if (isRunningRef.current) {
        startPollingFallback()
        setTimeout(connectWS, 1500)
      }
    }
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
    stopPolling()
    setState(null)
  }

  // Cleanup on unmount
  useEffect(() => () => {
    wsRef.current?.close()
    stopPolling()
  }, [])

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
                src={buildApiUrl('/api/video/feed')}
                alt="Live feed"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-center text-slate-500 space-y-2">
                <div className="text-5xl">📷</div>
                <p className="text-sm">Press <strong>Start Session</strong> to begin</p>
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
