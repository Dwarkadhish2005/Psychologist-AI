import { type RiskLevel } from '../types'

const RISK_CONFIG: Record<
  RiskLevel,
  { label: string; bg: string; text: string; dot: string; pulse: boolean }
> = {
  low:      { label: 'LOW',      bg: 'bg-green-900/60',  text: 'text-green-300',  dot: 'bg-green-400',  pulse: false },
  moderate: { label: 'MODERATE', bg: 'bg-yellow-900/60', text: 'text-yellow-300', dot: 'bg-yellow-400', pulse: false },
  high:     { label: 'HIGH',     bg: 'bg-orange-900/60', text: 'text-orange-300', dot: 'bg-orange-400', pulse: true  },
  critical: { label: 'CRITICAL', bg: 'bg-red-900/60',    text: 'text-red-300',    dot: 'bg-red-500',    pulse: true  },
}

interface Props {
  risk: string
  size?: 'sm' | 'md' | 'lg'
}

export default function RiskBadge({ risk, size = 'md' }: Props) {
  const cfg = RISK_CONFIG[(risk?.toLowerCase() as RiskLevel)] ?? RISK_CONFIG.low
  const sizeClass = size === 'sm' ? 'text-xs px-2 py-0.5' : size === 'lg' ? 'text-base px-4 py-1.5' : 'text-sm px-3 py-1'

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold tracking-wide ${cfg.bg} ${cfg.text} ${sizeClass} ${cfg.pulse ? 'risk-critical' : ''}`}
    >
      <span className={`h-2 w-2 rounded-full ${cfg.dot}`} />
      {cfg.label}
    </span>
  )
}
