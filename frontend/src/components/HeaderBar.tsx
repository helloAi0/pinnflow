import React from 'react'
import { Activity, CheckCircle2, AlertCircle, GitCommit, Layers, ExternalLink, RefreshCw } from 'lucide-react'
import type { HealthResponse } from '../types/api'
import { getApiUrl } from '../services/api'

interface HeaderBarProps {
  health: HealthResponse
  onRefreshHealth?: () => void
  isRefreshing?: boolean
}

export const HeaderBar: React.FC<HeaderBarProps> = ({ health, onRefreshHealth, isRefreshing }) => {
  const isReady = health.status === 'ok' && health.trained
  const apiUrl = getApiUrl()

  return (
    <header className="border-b border-slate-800/80 bg-slate-950/90 px-6 py-4 backdrop-blur-xl sticky top-0 z-30 shadow-lg shadow-black/20">
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Brand & Project Identity */}
        <div className="flex items-center space-x-3.5">
          <div className="relative flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 text-cyan-400 ring-1 ring-cyan-500/40 shadow-inner">
            <Activity className="h-6 w-6 animate-pulse" />
            <span className="absolute -top-1 -right-1 flex h-3 w-3">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isReady ? 'bg-emerald-400' : 'bg-amber-400'}`} />
              <span className={`relative inline-flex rounded-full h-3 w-3 ${isReady ? 'bg-emerald-500' : 'bg-amber-500'}`} />
            </span>
          </div>
          <div>
            <div className="flex items-center space-x-2.5">
              <h1 className="text-xl font-extrabold tracking-tight text-white font-mono bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-teal-200 to-blue-400">
                PINNFlow
              </h1>
              <span className="rounded-full bg-cyan-950/80 px-2.5 py-0.5 text-[11px] font-semibold text-cyan-300 border border-cyan-700/60 shadow-sm font-mono">
                v{health.model_version}
              </span>
              <span className="hidden sm:inline-flex rounded-full bg-blue-950/60 px-2 py-0.5 text-[10px] font-medium text-blue-300 border border-blue-800/40 font-mono">
                Production
              </span>
            </div>
            <p className="text-xs text-slate-400 font-sans">
              Physics-Informed Navier-Stokes Flow Reconstruction Instrument
            </p>
          </div>
        </div>

        {/* Live Scientific & Runtime Telemetry */}
        <div className="flex flex-wrap items-center gap-2.5 font-mono text-xs">
          <div className="flex items-center space-x-1.5 rounded-lg bg-slate-900/90 px-3 py-1.5 border border-slate-800 text-slate-300 shadow-sm">
            <Layers className="h-3.5 w-3.5 text-cyan-400" />
            <span className="text-slate-500">Model:</span>
            <span className="font-semibold text-slate-200">{health.architecture || 'hard_constrained_pinn'}</span>
          </div>

          <div className="flex items-center space-x-1.5 rounded-lg bg-slate-900/90 px-3 py-1.5 border border-slate-800 text-slate-300 shadow-sm">
            <span className="text-slate-500">Re:</span>
            <span className="font-bold text-cyan-300">{health.reynolds_number.toFixed(1)}</span>
          </div>

          <div className="hidden md:flex items-center space-x-1.5 rounded-lg bg-slate-900/90 px-3 py-1.5 border border-slate-800 text-slate-300 shadow-sm">
            <span className="text-slate-500">Seed:</span>
            <span className="font-semibold text-slate-200">{health.training_seed ?? '0'}</span>
          </div>

          <div className="hidden lg:flex items-center space-x-1.5 rounded-lg bg-slate-900/90 px-3 py-1.5 border border-slate-800 text-slate-300 shadow-sm">
            <GitCommit className="h-3.5 w-3.5 text-slate-400" />
            <span className="text-slate-500">Commit:</span>
            <span className="text-slate-300">{health.git_commit ? health.git_commit.slice(0, 7) : 'head'}</span>
          </div>

          {/* Status Badge */}
          <div className={`flex items-center space-x-1.5 rounded-lg px-3 py-1.5 border font-semibold shadow-sm transition-colors ${
            isReady
              ? 'bg-emerald-950/70 border-emerald-700/80 text-emerald-300'
              : 'bg-amber-950/70 border-amber-700/80 text-amber-300'
          }`}>
            {isReady ? (
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
            ) : (
              <AlertCircle className="h-3.5 w-3.5 text-amber-400" />
            )}
            <span>{isReady ? 'Checkpoint Verified' : 'Service Not Ready'}</span>
          </div>

          {/* API Docs link */}
          <a
            href={`${apiUrl}/docs`}
            target="_blank"
            rel="noreferrer"
            className="flex items-center space-x-1 rounded-lg bg-slate-800/80 hover:bg-slate-700 px-2.5 py-1.5 border border-slate-700 text-slate-300 hover:text-white transition shadow-sm text-[11px]"
            title="Open OpenAPI Interactive Documentation"
          >
            <span>Swagger API</span>
            <ExternalLink className="h-3 w-3 text-cyan-400" />
          </a>

          {onRefreshHealth && (
            <button
              onClick={onRefreshHealth}
              disabled={isRefreshing}
              className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-400 hover:text-cyan-400 transition"
              title="Refresh System Health"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? 'animate-spin text-cyan-400' : ''}`} />
            </button>
          )}
        </div>
      </div>
    </header>
  )
}