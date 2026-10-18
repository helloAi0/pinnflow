import React, { useState, useEffect } from 'react'
import { Play, Pause, Sliders, RefreshCw, Download, Sparkles, FastForward, Rewind, Server } from 'lucide-react'
import type { FieldRequest, HealthResponse } from '../types/api'

interface QueryControlsProps {
  params: FieldRequest
  setParams: React.Dispatch<React.SetStateAction<FieldRequest>>
  onEvaluate: (overrideParams?: FieldRequest) => void
  isLoading: boolean
  error: string | null
  compareMode: boolean
  setCompareMode: (c: boolean) => void
  onExportField: () => void
  onExportMetrics: () => void
  health?: HealthResponse
  isPlaying: boolean
  setIsPlaying: (p: boolean) => void
}

export const QueryControls: React.FC<QueryControlsProps> = ({
  params,
  setParams,
  onEvaluate,
  isLoading,
  error,
  compareMode,
  setCompareMode,
  onExportField,
  onExportMetrics,
  health,
  isPlaying,
  setIsPlaying
}) => {
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1)
  const [localNx, setLocalNx] = useState<string>(String(params.nx))
  const [localNy, setLocalNy] = useState<string>(String(params.ny))

  useEffect(() => {
    setLocalNx(String(params.nx))
  }, [params.nx])

  useEffect(() => {
    setLocalNy(String(params.ny))
  }, [params.ny])

  const commitNx = () => {
    const val = parseInt(localNx, 10)
    const clamped = isNaN(val) ? 100 : Math.max(10, Math.min(200, val))
    setLocalNx(String(clamped))
    if (clamped !== params.nx) {
      setParams((prev) => ({ ...prev, nx: clamped }))
    }
  }

  const commitNy = () => {
    const val = parseInt(localNy, 10)
    const clamped = isNaN(val) ? 50 : Math.max(10, Math.min(200, val))
    setLocalNy(String(clamped))
    if (clamped !== params.ny) {
      setParams((prev) => ({ ...prev, ny: clamped }))
    }
  }

  const applyPreset = (presetType: 'vortex' | 'inflow' | 'farwake' | 'highres') => {
    if (presetType === 'vortex') {
      setParams((prev) => ({ ...prev, t: 10.0, nx: 100, ny: 50, x_min: 1.0, x_max: 8.0, y_min: -2.0, y_max: 2.0 }))
    } else if (presetType === 'inflow') {
      setParams((prev) => ({ ...prev, t: 2.0, nx: 100, ny: 50, x_min: 1.0, x_max: 8.0, y_min: -2.0, y_max: 2.0 }))
    } else if (presetType === 'farwake') {
      setParams((prev) => ({ ...prev, t: 16.0, nx: 100, ny: 50, x_min: 2.0, x_max: 8.0, y_min: -2.0, y_max: 2.0 }))
    } else if (presetType === 'highres') {
      // 140x70 = 9,800 points (within 10,000 point cap with vorticity)
      setParams((prev) => ({ ...prev, nx: 140, ny: 70 }))
    }
  }

  return (
    <div className="flex flex-col space-y-5 rounded-xl border border-slate-800/90 bg-slate-900/70 p-5 backdrop-blur-xl shadow-2xl glass-panel">
      {/* Header with Quick Presets */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
        <div className="flex items-center space-x-2 text-sm font-bold text-slate-100">
          <Sliders className="h-4 w-4 text-cyan-400" />
          <span>Experiment Controls</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/80 px-2.5 py-0.5 rounded-md border border-cyan-800 font-bold">
            Re = {health?.reynolds_number ?? 100.0}
          </span>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-red-800/80 bg-red-950/70 p-3.5 text-xs text-red-200 shadow-lg space-y-2">
          <div className="flex items-center space-x-1.5 font-bold text-red-300">
            <span className="flex h-2 w-2 rounded-full bg-red-400 animate-ping" />
            <span>Backend Notice / Error</span>
          </div>
          <p className="font-mono text-[11px] text-red-200 leading-relaxed">{error}</p>
          {(error.includes('502') || error.includes('memory') || error.includes('OOM') || error.includes('timed out') || error.includes('limit') || error.includes('exceeds')) && (
            <div className="flex flex-wrap gap-2 pt-1 border-t border-red-900/60">
              <button
                type="button"
                onClick={() => {
                  const safeParams: FieldRequest = { ...params, nx: 60, ny: 30, compute_vorticity: false }
                  setParams(safeParams)
                  onEvaluate(safeParams)
                }}
                className="flex items-center space-x-1 rounded bg-red-900/80 hover:bg-red-800 px-2.5 py-1 text-[11px] font-mono text-white font-bold transition shadow-sm"
              >
                <span>⚡ Set Safe Grid (60x30) & Retry</span>
              </button>
            </div>
          )}
        </div>
      )}

      {/* Preset Quick Actions */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-[11px] text-slate-400">
          <span className="flex items-center space-x-1">
            <Sparkles className="h-3 w-3 text-cyan-400" />
            <span>Quick Presets:</span>
          </span>
        </div>
        <div className="grid grid-cols-2 gap-1.5">
          <button
            type="button"
            onClick={() => applyPreset('vortex')}
            className="px-2 py-1 text-[11px] rounded-md bg-slate-950 hover:bg-slate-800 border border-slate-800 hover:border-cyan-700 text-slate-300 hover:text-cyan-300 font-mono transition text-left"
          >
            🌊 Shedding (t=10s)
          </button>
          <button
            type="button"
            onClick={() => applyPreset('inflow')}
            className="px-2 py-1 text-[11px] rounded-md bg-slate-950 hover:bg-slate-800 border border-slate-800 hover:border-cyan-700 text-slate-300 hover:text-cyan-300 font-mono transition text-left"
          >
            ⚡ Transient (t=2s)
          </button>
          <button
            type="button"
            onClick={() => applyPreset('farwake')}
            className="px-2 py-1 text-[11px] rounded-md bg-slate-950 hover:bg-slate-800 border border-slate-800 hover:border-cyan-700 text-slate-300 hover:text-cyan-300 font-mono transition text-left"
          >
            🌀 Far Wake (t=16s)
          </button>
          <button
            type="button"
            onClick={() => applyPreset('highres')}
            className="px-2 py-1 text-[11px] rounded-md bg-slate-950 hover:bg-slate-800 border border-slate-800 hover:border-cyan-700 text-slate-300 hover:text-cyan-300 font-mono transition text-left"
          >
            🔍 High Res (140x70)
          </button>
        </div>
      </div>

      {/* READ-ONLY SERVED MODEL PANEL (Replaces the obsolete dropdowns) */}
      <div className="space-y-2 rounded-xl bg-slate-950/90 p-3.5 border border-cyan-900/60 shadow-inner">
        <div className="flex items-center justify-between text-xs font-semibold text-slate-200 border-b border-slate-800/80 pb-2">
          <div className="flex items-center space-x-1.5 text-cyan-400">
            <Server className="h-3.5 w-3.5" />
            <span>Served Model (Production Checkpoint)</span>
          </div>
          <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-800">
            {health?.trained ? 'Active' : 'Standby'}
          </span>
        </div>

        <div className="space-y-1.5 text-[11px] font-mono text-slate-300 pt-1">
          <div className="flex justify-between">
            <span className="text-slate-500">Architecture:</span>
            <span className="text-slate-200 font-medium">Fourier Hard-Constrained PINN</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Weights File:</span>
            <span className="text-cyan-300 truncate max-w-[170px]" title={health?.checkpoint_name || 'final_pinn_model.pth'}>
              {health?.checkpoint_name || 'final_pinn_model.pth'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Boundary Guard:</span>
            <span className="text-slate-200">Exact Solid-Wall No-Slip (R=0.5)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Execution Hardware:</span>
            <span className="text-slate-200">CPU Single Thread (512MB RAM Cap)</span>
          </div>
        </div>
      </div>

      {/* Temporal Snapshot Slider with Interactive Animation Player */}
      <div className="space-y-2.5 rounded-xl bg-slate-950/80 p-3.5 border border-slate-800 shadow-inner">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-slate-200">Temporal Coordinate (t)</span>
          <span className="font-mono text-cyan-300 font-bold bg-cyan-950/80 px-2 py-0.5 rounded border border-cyan-800">
            {params.t.toFixed(1)} s
          </span>
        </div>

        <input
          type="range"
          min="0.0"
          max="19.9"
          step="0.1"
          value={params.t}
          onChange={(e) => setParams((prev) => ({ ...prev, t: parseFloat(e.target.value) }))}
          className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg appearance-none"
        />

        <div className="flex justify-between text-[10px] text-slate-500 font-mono">
          <span>0.0s (Inlet)</span>
          <span>10.0s (Periodic)</span>
          <span>19.9s (Wake)</span>
        </div>

        {/* Animation Playback Controls */}
        <div className="flex items-center justify-between pt-1 border-t border-slate-800/80">
          <div className="flex items-center space-x-1.5">
            <button
              type="button"
              onClick={() => setParams((prev) => ({ ...prev, t: Math.max(0.0, Math.round((prev.t - 0.5) * 10) / 10) }))}
              className="p-1 rounded bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-400 hover:text-white"
              title="Step Back -0.5s"
            >
              <Rewind className="h-3.5 w-3.5" />
            </button>

            <button
              type="button"
              onClick={() => setIsPlaying(!isPlaying)}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded text-xs font-mono font-medium transition ${
                isPlaying ? 'bg-amber-600 text-white' : 'bg-cyan-950 border border-cyan-800 text-cyan-300 hover:bg-cyan-900'
              }`}
            >
              {isPlaying ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
              <span>{isPlaying ? 'Pause' : 'Animate'}</span>
            </button>

            <button
              type="button"
              onClick={() => setParams((prev) => ({ ...prev, t: Math.min(19.9, Math.round((prev.t + 0.5) * 10) / 10) }))}
              className="p-1 rounded bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-400 hover:text-white"
              title="Step Forward +0.5s"
            >
              <FastForward className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="flex items-center space-x-1 text-[10px] font-mono">
            <span className="text-slate-500">Speed:</span>
            {[0.5, 1, 2].map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setPlaybackSpeed(s)}
                className={`px-1.5 py-0.5 rounded ${
                  playbackSpeed === s ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Resolution Grid Inputs (with point cap limit indicator) */}
      <div className="space-y-1">
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <label className="text-xs text-slate-300 font-mono">nx (columns)</label>
            <input
              type="number"
              min="10"
              max="200"
              value={localNx}
              onChange={(e) => setLocalNx(e.target.value)}
              onBlur={commitNx}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  commitNx()
                  e.currentTarget.blur()
                }
              }}
              className="w-full rounded-lg border border-slate-700/80 bg-slate-950 px-3 py-1.5 text-xs text-slate-200 font-mono"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-slate-300 font-mono">ny (rows)</label>
            <input
              type="number"
              min="10"
              max="200"
              value={localNy}
              onChange={(e) => setLocalNy(e.target.value)}
              onBlur={commitNy}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  commitNy()
                  e.currentTarget.blur()
                }
              }}
              className="w-full rounded-lg border border-slate-700/80 bg-slate-950 px-3 py-1.5 text-xs text-slate-200 font-mono"
            />
          </div>
        </div>
        <div className="text-[10px] font-mono text-slate-500 text-right pt-0.5">
          Grid: {params.nx * params.ny} points (Cap: {params.compute_vorticity ? '10,000 pts' : '20,000 pts'})
        </div>
      </div>

      {/* Toggles */}
      <div className="flex flex-col space-y-2 pt-2 border-t border-slate-800/80">
        <label className="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer">
          <input
            type="checkbox"
            checked={compareMode}
            onChange={(e) => setCompareMode(e.target.checked)}
            className="rounded border-slate-700 bg-slate-950 text-cyan-500 focus:ring-0"
          />
          <span>Compare vs Reference DNS Ground Truth</span>
        </label>

        <label className="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer">
          <input
            type="checkbox"
            checked={params.compute_vorticity}
            onChange={(e) => setParams((prev) => ({ ...prev, compute_vorticity: e.target.checked }))}
            className="rounded border-slate-700 bg-slate-950 text-cyan-500 focus:ring-0"
          />
          <span>Compute Exact Autograd Vorticity (ω)</span>
        </label>
      </div>

      {/* Main Action Button */}
      <button
        type="button"
        onClick={() => onEvaluate()}
        disabled={isLoading}
        className="flex w-full items-center justify-center space-x-2 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 px-4 py-3 text-xs font-bold text-white transition-all hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-cyan-900/40 active:scale-[0.98]"
      >
        {isLoading ? (
          <>
            <RefreshCw className="h-4 w-4 animate-spin text-white" />
            <span>Evaluating Autograd Navier-Stokes...</span>
          </>
        ) : (
          <>
            <Play className="h-4 w-4 fill-current" />
            <span>Evaluate Physical Field</span>
          </>
        )}
      </button>

      {/* Export tools */}
      <div className="flex gap-2 pt-2 border-t border-slate-800/80">
        <button
          type="button"
          onClick={onExportField}
          className="flex-1 flex items-center justify-center space-x-1.5 rounded-lg border border-slate-700/80 bg-slate-950 px-2 py-2 text-[11px] font-mono text-slate-300 hover:bg-slate-800 hover:text-white transition shadow-sm"
        >
          <Download className="h-3.5 w-3.5 text-cyan-400" />
          <span>Export Field CSV</span>
        </button>
        <button
          type="button"
          onClick={onExportMetrics}
          className="flex-1 flex items-center justify-center space-x-1.5 rounded-lg border border-slate-700/80 bg-slate-950 px-2 py-2 text-[11px] font-mono text-slate-300 hover:bg-slate-800 hover:text-white transition shadow-sm"
        >
          <Download className="h-3.5 w-3.5 text-cyan-400" />
          <span>Export Metrics JSON</span>
        </button>
      </div>
    </div>
  )
}