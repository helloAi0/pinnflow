import React from 'react'

interface MetricsFooterProps {
  latencyMs: number | null
  l2Error: number | null
}

export const MetricsFooter: React.FC<MetricsFooterProps> = ({ latencyMs, l2Error }) => {
  return (
    <footer className="grid grid-cols-3 gap-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4 font-mono text-xs shadow-md">
      <div>
        <span className="text-[10px] uppercase text-[var(--muted-foreground)]">Model Checkpoint</span>
        <p className="mt-0.5 font-bold text-[var(--foreground)]">fourier_constrained_seed8008.pth</p>
      </div>
      <div>
        <span className="text-[10px] uppercase text-[var(--muted-foreground)]">Inference Latency</span>
        <p className="mt-0.5 font-bold text-[var(--color-primary)]">
          {latencyMs !== null ? `${latencyMs.toFixed(1)} ms` : '—'}
        </p>
      </div>
      <div>
        <span className="text-[10px] uppercase text-[var(--muted-foreground)]">Relative L2 Field Error</span>
        <p className="mt-0.5 font-bold text-[var(--color-success)]">
          {l2Error !== null ? l2Error.toExponential(4) : '—'}
        </p>
      </div>
    </footer>
  )
}