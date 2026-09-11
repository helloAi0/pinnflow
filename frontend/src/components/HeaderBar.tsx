import React from 'react'
import { Activity, CheckCircle2, AlertCircle, GitCommit, Layers } from 'lucide-react'
import type { HealthResponse } from '../types/api'

interface HeaderBarProps {
  health: HealthResponse
}

export const HeaderBar: React.FC<HeaderBarProps> = ({ health }) => {
  const isReady = health.status === 'ok' && health.trained

  return (
    <header className="border-b border-slate-800 bg-slate-950/80 px-6 py-4 backdrop-blur-md">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-600/20 text-cyan-400 ring-1 ring-cyan-500/30">
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-bold tracking-tight text-white font-mono">PINNFlow</h1>
              <span className="rounded bg-cyan-950 px-2 py-0.5 text-xs font-semibold text-cyan-400 border border-cyan-800">
                v{health.model_version}
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Physics-Informed Navier-Stokes Scientific Research Instrument
            </p>
          </div>
        </div>

        {/* Live System & Scientific Metadata */}
        <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
          <div className="flex items-center space-x-1.5 rounded-md bg-slate-900 px-3 py-1.5 border border-slate-800 text-slate-300">
            <Layers className="h-3.5 w-3.5 text-cyan-400" />
            <span className="text-slate-500">Model:</span>
            <span className="font-semibold text-slate-200">{health.architecture || 'standard_mlp'}</span>
          </div>

          <div className="flex items-center space-x-1.5 rounded-md bg-slate-900 px-3 py-1.5 border border-slate-800 text-slate-300">
            <span className="text-slate-500">Re:</span>
            <span className="font-semibold text-cyan-300">{health.reynolds_number.toFixed(1)}</span>
          </div>

          <div className="flex items-center space-x-1.5 rounded-md bg-slate-900 px-3 py-1.5 border border-slate-800 text-slate-300">
            <span className="text-slate-500">Seed:</span>
            <span className="font-semibold text-slate-200">{health.training_seed ?? 'N/A'}</span>
          </div>

          <div className="flex items-center space-x-1.5 rounded-md bg-slate-900 px-3 py-1.5 border border-slate-800 text-slate-300">
            <GitCommit className="h-3.5 w-3.5 text-slate-400" />
            <span className="text-slate-500">Commit:</span>
            <span className="text-slate-300">{health.git_commit ? health.git_commit.slice(0, 7) : 'head'}</span>
          </div>

          <div className={`flex items-center space-x-1.5 rounded-md px-3 py-1.5 border ${
            isReady
              ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
              : 'bg-amber-950/60 border-amber-800 text-amber-300'
          }`}>
            {isReady ? (
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
            ) : (
              <AlertCircle className="h-3.5 w-3.5 text-amber-400" />
            )}
            <span>{isReady ? 'Checkpoint Verified' : 'Service Not Ready'}</span>
          </div>
        </div>
      </div>
    </header>
  )
}