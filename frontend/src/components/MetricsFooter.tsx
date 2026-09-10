import React from 'react'
import type { HealthResponse } from '../types/api'

interface MetricsFooterProps {
  latencyMs: number | null
  l2Error: number | null
  healthData: HealthResponse
}

export const MetricsFooter: React.FC<MetricsFooterProps> = ({ latencyMs, l2Error, healthData }) => {
  return (
    <footer className="grid grid-cols-3 gap-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4 font-mono text-xs shadow-md">
      <div>
        <span className="text-[10px] uppercase text-[var(--muted-foreground)]">Model Checkpoint</span>
        {healthData.checkpoint ? (
          <p className="mt-0.5 font-bold text-[var(--foreground)]">{healthData.checkpoint}</p>
        ) : (
          <p className="mt-0.5 font-bold text-[var(--color-danger)]">No trained weights — fallback mode</p>
        )}
      </div>
      <div>
        <span className="text-[10px] uppercase text-[var(--muted-foreground)]">Inference Latency</span>
        <p className="mt-0.5 font-bold text-[var(--color-primary)]">
          {latencyMs !== null ? `${latencyMs.toFixed(1)} ms` : '—'}
        </p>
      </div>
      <div>
        <span className="text-[10px] uppercase text-[var(--muted-foreground)]">Relative L2 Field Error (vs DNS)</span>
        {l2Error !== null ? (
          <p className="mt-0.5 font-bold text-[var(--color-success)]">{l2Error.toExponential(4)}</p>
        ) : compareModeHasNoData(healthData) ? (
          <p className="mt-0.5 font-bold text-[var(--muted-foreground)]">Reference unavailable</p>
        ) : (
          <p className="mt-0.5 font-bold text-[var(--muted-foreground)]">—</p>
        )}
      </div>
    </footer>
  )
}

function compareModeHasNoData(healthData: HealthResponse) {
  return healthData.status === 'ok' && !healthData.reference_dataset_available
}