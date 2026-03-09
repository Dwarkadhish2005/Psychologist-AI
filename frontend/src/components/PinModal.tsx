import { useState, useRef, useEffect } from 'react'
import { Lock, X, Eye, EyeOff } from 'lucide-react'

interface Props {
  userId: string
  userName?: string
  onVerified: () => void
  onCancel?: () => void
}

export default function PinModal({ userId, userName, onVerified, onCancel }: Props) {
  const [pin, setPin] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [showPin, setShowPin] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  async function handleVerify(e: React.FormEvent) {
    e.preventDefault()
    if (!pin) return
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`/api/users/${userId}/pin/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pin }),
      })
      const data = await res.json()
      if (data.valid) {
        // Remember verification in sessionStorage so we don't ask again this session
        sessionStorage.setItem(`pin_verified_${userId}`, '1')
        onVerified()
      } else {
        setError('Incorrect PIN. Please try again.')
        setPin('')
      }
    } catch {
      setError('Could not verify PIN. Check your connection.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-[#161b27] border border-slate-700 rounded-2xl p-8 w-full max-w-sm shadow-2xl">
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-600/20 p-2.5 rounded-xl">
              <Lock size={20} className="text-indigo-400" />
            </div>
            <div>
              <p className="text-white font-semibold">PIN Required</p>
              {userName && <p className="text-slate-400 text-xs mt-0.5">{userName}</p>}
            </div>
          </div>
          {onCancel && (
            <button onClick={onCancel} className="text-slate-500 hover:text-white transition-colors">
              <X size={18} />
            </button>
          )}
        </div>

        <form onSubmit={handleVerify} className="space-y-4">
          <div>
            <label className="text-xs text-slate-400 block mb-1.5">Enter PIN</label>
            <div className="relative">
              <input
                ref={inputRef}
                type={showPin ? 'text' : 'password'}
                inputMode="numeric"
                pattern="[0-9]*"
                maxLength={8}
                value={pin}
                onChange={e => { setPin(e.target.value.replace(/\D/g, '')); setError('') }}
                placeholder="••••"
                className="w-full bg-slate-900 border border-slate-600 text-white text-center text-xl tracking-[0.5em] font-mono rounded-lg px-4 py-3 pr-12 focus:outline-none focus:ring-1 focus:ring-indigo-500 placeholder-slate-700"
              />
              <button
                type="button"
                onClick={() => setShowPin(v => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white transition-colors"
              >
                {showPin ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            {error && <p className="text-red-400 text-xs mt-1.5">{error}</p>}
          </div>

          <button
            type="submit"
            disabled={loading || pin.length < 4}
            className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white py-2.5 rounded-lg font-medium transition-colors"
          >
            {loading ? 'Verifying…' : 'Unlock'}
          </button>
        </form>
      </div>
    </div>
  )
}

/** Returns true if the user is verified (no PIN or pin already verified this session) */
export async function checkPinRequired(userId: string): Promise<boolean> {
  if (sessionStorage.getItem(`pin_verified_${userId}`)) return false
  try {
    const res = await fetch(`/api/users/${userId}/pin/status`)
    const data = await res.json()
    return data.has_pin === true
  } catch {
    return false
  }
}
