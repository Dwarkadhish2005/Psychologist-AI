// Animated pulse skeleton blocks for loading states

function Block({ className }: { className?: string }) {
  return (
    <div className={`bg-slate-700/60 rounded-lg animate-pulse ${className ?? ''}`} />
  )
}

export function SkeletonCard() {
  return (
    <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5 space-y-3">
      <Block className="h-4 w-32" />
      <Block className="h-8 w-24" />
      <Block className="h-3 w-20" />
    </div>
  )
}

export function SkeletonUserCard() {
  return (
    <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5 space-y-3">
      <div className="flex justify-between">
        <div className="space-y-2">
          <Block className="h-4 w-36" />
          <Block className="h-3 w-24" />
        </div>
        <Block className="h-6 w-16 rounded-full" />
      </div>
      <div className="flex gap-4">
        <Block className="h-3 w-20" />
        <Block className="h-3 w-16" />
      </div>
      <div className="flex gap-2">
        <Block className="h-8 flex-1 rounded-lg" />
        <Block className="h-8 flex-1 rounded-lg" />
        <Block className="h-8 flex-1 rounded-lg" />
      </div>
    </div>
  )
}

export function SkeletonChart() {
  return (
    <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-5 space-y-4">
      <Block className="h-4 w-48" />
      <Block className="h-48 w-full rounded-xl" />
    </div>
  )
}

export function SkeletonRow() {
  return (
    <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-4 flex items-center gap-4">
      <Block className="h-10 w-24 flex-shrink-0" />
      <Block className="h-10 flex-1" />
      <Block className="h-10 w-28 flex-shrink-0" />
      <Block className="h-6 w-16 rounded-full flex-shrink-0" />
    </div>
  )
}

export function SkeletonStatRow() {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <SkeletonCard key={i} />
      ))}
    </div>
  )
}
