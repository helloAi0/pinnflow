import React, { useState } from 'react'
import type { FieldRequest } from '../types/api'
import { AlertCircle, ChevronDown, ChevronUp, Loader2, Play } from 'lucide-react'

interface QueryControlsProps {
  params: FieldRequest
  setParams: React.Dispatch<React.SetStateAction<FieldRequest>>
  onEvaluate: () => void
  isLoading: boolean
  error: string | null
}

export const QueryControls: React.FC<QueryControlsProps> = ({
  params,
  setParams,
  onEvaluate,
  isLoading,
  error,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false)

  return (
    <div className="flex flex-col space-y-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-5 shadow-lg">
      <div className="border-b border-[var(--border)] pb-3">
        <h2 className="text-sm font-semibold tracking-wide uppercase text-[var(--foreground)]">
          Query Controls
        </h2>
        <p className="text-xs text-[var(--muted-foreground)]">
          Configure spatiotemporal domain parameters
        </p>
      </div>

      {/* Time Slider */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs font-mono">
          <span className="text-[var(--muted-foreground)]">Time Snapshot (t)</span>
          <span className="font-bold text-[var(--color-primary)]">{params.t.toFixed(1)}s</span>
        </div>
        <input
          type="range"
          min="0.0"
          max="20.0"
          step="0.2"
          value={params.t}
          onChange={(e) => setParams((prev) => ({ ...prev, t: parseFloat(e.target.value) }))}
          className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-[var(--slate-800)] accent-[var(--color-primary)]"
        />
        <div className="flex justify-between text-[10px] font-mono text-[var(--muted-foreground)]">
          <span>0.0s</span>
          <span>10.0s</span>
          <span>20.0s</span>
        </div>
      </div>

      {/* Vorticity Toggle */}
      <div className="flex items-center justify-between rounded-lg border border-[var(--border)] bg-[var(--slate-950)] p-3">
        <span className="text-xs font-medium text-[var(--foreground)]">Compute Vorticity Field</span>
        <button
          type="button"
          onClick={() => setParams((prev) => ({ ...prev, compute_vorticity: !prev.compute_vorticity }))}
          className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
            params.compute_vorticity ? 'bg-[var(--color-primary)]' : 'bg-[var(--slate-800)]'
          }`}
        >
          <span
            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
              params.compute_vorticity ? 'translate-x-4.5' : 'translate-x-1'
            }`}
          />
        </button>
      </div>

      {/* Advanced Domain Bounds */}
      <div className="rounded-lg border border-[var(--border)] bg-[var(--slate-950)]">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex w-full items-center justify-between p-3 text-xs font-medium text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
        >
          <span>Domain Bounds & Resolution</span>
          {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>

        {showAdvanced && (
          <div className="grid grid-cols-2 gap-3 border-t border-[var(--border)] p-3 text-xs">
            <div>
              <label className="text-[10px] font-mono text-[var(--muted-foreground)]">X Min / Max</label>
              <div className="flex space-x-1 mt-1">
                <input
                  type="number"
                  value={params.x_min}
                  onChange={(e) => setParams((prev) => ({ ...prev, x_min: parseFloat(e.target.value) || 0 }))}
                  className="w-full rounded border border-[var(--border)] bg-[var(--surface)] px-2 py-1 font-mono text-xs text-[var(--foreground)]"
                />
                <input
                  type="number"
                  value={params.x_max}
                  onChange={(e) => setParams((prev) => ({ ...prev, x_max: parseFloat(e.target.value) || 0 }))}
                  className="w-full rounded border border-[var(--border)] bg-[var(--surface)] px-2 py-1 font-mono text-xs text-[var(--foreground)]"
                />
              </div>
            </div>
            <div>
              <label className="text-[10px] font-mono text-[var(--muted-foreground)]">Y Min / Max</label>
              <div className="flex space-x-1 mt-1">
                <input
                  type="number"
                  value={params.y_min}
                  onChange={(e) => setParams((prev) => ({ ...prev, y_min: parseFloat(e.target.value) || 0 }))}
                  className="w-full rounded border border-[var(--border)] bg-[var(--surface)] px-2 py-1 font-mono text-xs text-[var(--foreground)]"
                />
                <input
                  type="number"
                  value={params.y_max}
                  onChange={(e) => setParams((prev) => ({ ...prev, y_max: parseFloat(e.target.value) || 0 }))}
                  className="w-full rounded border border-[var(--border)] bg-[var(--surface)] px-2 py-1 font-mono text-xs text-[var(--foreground)]"
                />
              </div>
            </div>
            <div className="col-span-2">
              <label className="text-[10px] font-mono text-[var(--muted-foreground)]">Grid Resolution (NX × NY)</label>
              <div className="flex space-x-1 mt-1">
                <input
                  type="number"
                  value={params.nx}
                  onChange={(e) => setParams((prev) => ({ ...prev, nx: parseInt(e.target.value) || 10 }))}
                  className="w-full rounded border border-[var(--border)] bg-[var(--surface)] px-2 py-1 font-mono text-xs text-[var(--foreground)]"
                />
                <input
                  type="number"
                  value={params.ny}
                  onChange={(e) => setParams((prev) => ({ ...prev, ny: parseInt(e.target.value) || 10 }))}
                  className="w-full rounded border border-[var(--border)] bg-[var(--surface)] px-2 py-1 font-mono text-xs text-[var(--foreground)]"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Evaluate Button */}
      <button
        type="button"
        onClick={onEvaluate}
        disabled={isLoading}
        className={`flex w-full items-center justify-center space-x-2 rounded-lg py-2.5 px-4 font-semibold text-xs transition-all shadow-md ${
          isLoading
            ? 'bg-[var(--slate-800)] text-[var(--muted-foreground)] cursor-not-allowed'
            : 'bg-[var(--color-primary)] text-white hover:bg-[var(--blue-600)] active:scale-[0.99]'
        }`}
      >
        {isLoading ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin text-[var(--color-primary)]" />
            <span>Evaluating network forward pass...</span>
          </>
        ) : (
          <>
            <Play className="h-4 w-4 fill-current" />
            <span>Evaluate Flow Field</span>
          </>
        )}
      </button>

      {/* Error Alert State */}
      {error && (
        <div className="flex items-start space-x-2.5 rounded-lg border border-red-900/50 bg-red-950/30 p-3 text-red-400">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          <p className="text-xs font-mono leading-tight">{error}</p>
        </div>
      )}
    </div>
  )
}