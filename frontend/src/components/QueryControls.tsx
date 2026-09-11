import React from 'react'
import { Play, Sliders, RefreshCw, Download } from 'lucide-react'
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
  return (
    <div className="flex flex-col space-y-5 rounded-xl border border-slate-800 bg-slate-900/70 p-5 backdrop-blur-md shadow-xl">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2 text-sm font-semibold text-slate-200">
          <Sliders className="h-4 w-4 text-cyan-400" />
          <span>Experiment Controls</span>
        </div>
        <span className="text-[11px] font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700">
          Re = 100.0
        </span>
      </div>

      {error && (
        <div className="rounded-lg border border-red-900/60 bg-red-950/40 p-3 text-xs text-red-300">
          <span className="font-semibold text-red-400">Error:</span> {error}
        </div>
      )}

      {/* Model Selection */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-slate-300">Surrogate Model Architecture</label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
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
            className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
          >
            <option value={500}>500 points (Extreme Scarcity)</option>
            <option value={1000}>1,000 points</option>
            <option value={2500}>2,500 points</option>
            <option value={5000}>5,000 points (Standard)</option>
            <option value={10000}>10,000 points</option>
          </select>
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-slate-300">Observation Noise</label>
          <select
            value={selectedNoise}
            onChange={(e) => setSelectedNoise(Number(e.target.value))}
            className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
          >
            <option value={0.0}>0% (Clean DNS)</option>
            <option value={0.01}>1% Gaussian Noise</option>
            <option value={0.05}>5% Gaussian Noise</option>
            <option value={0.10}>10% Gaussian Noise</option>
            <option value={0.20}>20% Extreme Noise</option>
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
            className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
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
            className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
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

      {/* Temporal Snapshot Slider */}
      <div className="space-y-2 rounded-lg bg-slate-950/60 p-3 border border-slate-800">
        <div className="flex justify-between text-xs">
          <span className="font-medium text-slate-300">Temporal Coordinate (t)</span>
          <span className="font-mono text-cyan-400 font-semibold">{params.t.toFixed(1)} s</span>
        </div>
        <input
          type="range"
          min="0.0"
          max="19.9"
          step="0.1"
          value={params.t}
          onChange={(e) => setParams((prev) => ({ ...prev, t: parseFloat(e.target.value) }))}
          className="w-full accent-cyan-500 cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-slate-500 font-mono">
          <span>0.0s (Inlet)</span>
          <span>10.0s (Vortex shedding)</span>
          <span>19.9s (End)</span>
        </div>
      </div>

      {/* Resolution */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <label className="text-xs text-slate-300">Grid nx (columns)</label>
          <input
            type="number"
            min="10"
            max="200"
            value={params.nx}
            onChange={(e) => setParams((prev) => ({ ...prev, nx: parseInt(e.target.value) || 100 }))}
            className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-slate-200"
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs text-slate-300">Grid ny (rows)</label>
          <input
            type="number"
            min="10"
            max="200"
            value={params.ny}
            onChange={(e) => setParams((prev) => ({ ...prev, ny: parseInt(e.target.value) || 50 }))}
            className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-slate-200"
          />
        </div>
      </div>

      {/* Toggles */}
      <div className="flex flex-col space-y-2 pt-1 border-t border-slate-800">
        <label className="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer">
          <input
            type="checkbox"
            checked={compareMode}
            onChange={(e) => setCompareMode(e.target.checked)}
            className="rounded border-slate-700 bg-slate-950 text-cyan-500 focus:ring-0"
          />
          <span>Compare against Reference DNS Ground Truth</span>
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

      {/* Action Button */}
      <button
        onClick={onEvaluate}
        disabled={isLoading}
        className="flex w-full items-center justify-center space-x-2 rounded-lg bg-cyan-600 px-4 py-2.5 text-xs font-semibold text-white transition-all hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-cyan-900/30"
      >
        {isLoading ? (
          <>
            <RefreshCw className="h-4 w-4 animate-spin" />
            <span>Evaluating Autograd Navier-Stokes...</span>
          </>
        ) : (
          <>
            <Play className="h-4 w-4" />
            <span>Evaluate Physical Field</span>
          </>
        )}
      </button>

      {/* Export tools */}
      <div className="flex gap-2 pt-2 border-t border-slate-800">
        <button
          onClick={onExportField}
          className="flex-1 flex items-center justify-center space-x-1.5 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-[11px] text-slate-300 hover:bg-slate-800 transition"
        >
          <Download className="h-3.5 w-3.5 text-cyan-400" />
          <span>Export Field CSV</span>
        </button>
        <button
          onClick={onExportMetrics}
          className="flex-1 flex items-center justify-center space-x-1.5 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-[11px] text-slate-300 hover:bg-slate-800 transition"
        >
          <Download className="h-3.5 w-3.5 text-cyan-400" />
          <span>Export Metrics JSON</span>
        </button>
      </div>
    </div>
  )
}