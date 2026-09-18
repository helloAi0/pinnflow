import React, { useState, useEffect, useRef } from 'react'
import { Play, Pause, Sliders, RefreshCw, Download, Sparkles, FastForward, Rewind } from 'lucide-react'
import type { FieldRequest } from '../types/api'

interface QueryControlsProps {
  params: FieldRequest
  setParams: React.Dispatch<React.SetStateAction<FieldRequest>>
  selectedModel: string
  setSelectedModel: (m: string) => void
  selectedBudget: number
  setSelectedBudget: (b: number) => void
  selectedNoise: number
  setSelectedNoise: (n: number) => void
  selectedSplit: string
  setSelectedSplit: (s: string) => void
  selectedSeed: number
  setSelectedSeed: (s: number) => void
  onEvaluate: () => void
  isLoading: boolean
  error: string | null
  compareMode: boolean
  setCompareMode: (c: boolean) => void
  onExportField: () => void
  onExportMetrics: () => void
}

export const QueryControls: React.FC<QueryControlsProps> = ({
  params,
  setParams,
  selectedModel,
  setSelectedModel,
  selectedBudget,
  setSelectedBudget,
  selectedNoise,
  setSelectedNoise,
  selectedSplit,
  setSelectedSplit,
  selectedSeed,
  setSelectedSeed,
  onEvaluate,
  isLoading,
  error,
  compareMode,
  setCompareMode,
  onExportField,
  onExportMetrics
}) => {
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1)
  const animIntervalRef = useRef<number | null>(null)

  // Temporal animation playback loop
  useEffect(() => {
    if (isPlaying) {
      animIntervalRef.current = window.setInterval(() => {
        setParams((prev) => {
          let nextT = Math.round((prev.t + 0.2 * playbackSpeed) * 10) / 10
          if (nextT > 19.9) nextT = 0.0
          return { ...prev, t: nextT }
        })
      }, 300)
    } else if (animIntervalRef.current) {
      clearInterval(animIntervalRef.current)
      animIntervalRef.current = null
    }

    return () => {
      if (animIntervalRef.current) {
        clearInterval(animIntervalRef.current)
      }
    }
  }, [isPlaying, playbackSpeed, setParams])

  const applyPreset = (presetType: 'vortex' | 'inflow' | 'farwake' | 'highres') => {
    if (presetType === 'vortex') {
      setParams((prev) => ({ ...prev, t: 10.0, nx: 100, ny: 50, x_min: 1.0, x_max: 8.0, y_min: -2.0, y_max: 2.0 }))
    } else if (presetType === 'inflow') {
      setParams((prev) => ({ ...prev, t: 2.0, nx: 100, ny: 50, x_min: 1.0, x_max: 8.0, y_min: -2.0, y_max: 2.0 }))
    } else if (presetType === 'farwake') {
      setParams((prev) => ({ ...prev, t: 16.0, nx: 120, ny: 60, x_min: 2.0, x_max: 8.0, y_min: -2.0, y_max: 2.0 }))
    } else if (presetType === 'highres') {
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
            Re = 100.0
          </span>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-900/80 bg-red-950/60 p-3 text-xs text-red-200 shadow-md">
          <div className="flex items-center space-x-1.5 font-bold text-red-400 mb-1">
            <span>Query Error:</span>
          </div>
          <p className="font-mono text-[11px]">{error}</p>
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

      {/* Model Selection */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-slate-300 flex justify-between">
          <span>Surrogate Model Architecture</span>
          <span className="text-[10px] text-cyan-400 font-mono">PyTorch 2.x</span>
        </label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="w-full rounded-lg border border-slate-700/80 bg-slate-950 px-3 py-2 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500 font-mono shadow-inner"
        >
          <option value="hard_constrained_pinn">Fourier Hard-Constrained PINN (Canonical Production)</option>
          <option value="fourier_pinn">Fourier Feature PINN (Tancik / Wang 2021)</option>
          <option value="adaptive_pinn">Homoscedastic Adaptive PINN (Kendall 2018)</option>
          <option value="static_pinn">Standard Static PINN (Raissi 2019)</option>
          <option value="baseline_mlp">Data-Only MLP Baseline (No Physics)</option>
        </select>
      </div>

      {/* Observation Budget & Noise */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-slate-300">Observation Budget</label>
          <select
            value={selectedBudget}
            onChange={(e) => setSelectedBudget(Number(e.target.value))}
            className="w-full rounded-lg border border-slate-700/80 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none font-mono"
          >
            <option value={500}>500 pts (Extreme)</option>
            <option value={1000}>1,000 pts</option>
            <option value={2500}>2,500 pts</option>
            <option value={5000}>5,000 pts (Standard)</option>
            <option value={10000}>10,000 pts</option>
          </select>
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-slate-300">Observation Noise</label>
          <select
            value={selectedNoise}
            onChange={(e) => setSelectedNoise(Number(e.target.value))}
            className="w-full rounded-lg border border-slate-700/80 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none font-mono"
          >
            <option value={0.0}>0% (Clean DNS)</option>
            <option value={0.01}>1% Noise</option>
            <option value={0.05}>5% Noise</option>
            <option value={0.10}>10% Noise</option>
            <option value={0.20}>20% Noise</option>
          </select>
        </div>
      </div>

      {/* Split Protocol & Seed */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-slate-300">Evaluation Split</label>
          <select
            value={selectedSplit}
            onChange={(e) => setSelectedSplit(e.target.value)}
            className="w-full rounded-lg border border-slate-700/80 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none font-mono"
          >
            <option value="random">Random Spatio-Temporal</option>
            <option value="sensor">Unseen Sensor Locations</option>
            <option value="temporal">Temporal Future Holdout</option>
          </select>
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-slate-300">Training Seed</label>
          <select
            value={selectedSeed}
            onChange={(e) => setSelectedSeed(Number(e.target.value))}
            className="w-full rounded-lg border border-slate-700/80 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none font-mono"
          >
            <option value={0}>Seed 0</option>
            <option value={1}>Seed 1</option>
            <option value={2}>Seed 2</option>
            <option value={3}>Seed 3</option>
            <option value={4}>Seed 4</option>
            <option value={42}>Seed 42</option>
          </select>
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

      {/* Resolution Grid Inputs */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <label className="text-xs text-slate-300 font-mono">nx (columns)</label>
          <input
            type="number"
            min="10"
            max="200"
            value={params.nx}
            onChange={(e) => setParams((prev) => ({ ...prev, nx: parseInt(e.target.value) || 100 }))}
            className="w-full rounded-lg border border-slate-700/80 bg-slate-950 px-3 py-1.5 text-xs text-slate-200 font-mono"
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs text-slate-300 font-mono">ny (rows)</label>
          <input
            type="number"
            min="10"
            max="200"
            value={params.ny}
            onChange={(e) => setParams((prev) => ({ ...prev, ny: parseInt(e.target.value) || 50 }))}
            className="w-full rounded-lg border border-slate-700/80 bg-slate-950 px-3 py-1.5 text-xs text-slate-200 font-mono"
          />
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
        onClick={onEvaluate}
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