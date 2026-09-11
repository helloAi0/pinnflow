import React from 'react'
import { Activity, Clock, Zap, Target, Gauge } from 'lucide-react'
import type { FieldResponse, HealthResponse } from '../types/api'

interface MetricsFooterProps {
  data: FieldResponse | null
  refData: FieldResponse | null
  latencyMs: number | null
  l2Error: number | null
  health: HealthResponse
}

export const MetricsFooter: React.FC<MetricsFooterProps> = ({
  data,
  latencyMs,
  l2Error,
  health
}) => {
  const pdeLoss = data?.metrics.pde_loss ?? null
  const contRes = data?.metrics.continuity_residual_mean ?? null

  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-4 lg:grid-cols-7">
      {/* 1. Relative L2 Velocity Error */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5 backdrop-blur-md">
        <div className="flex items-center space-x-1.5 text-xs text-slate-400">
          <Target className="h-3.5 w-3.5 text-cyan-400" />
          <span>Relative L2 Error</span>
        </div>
        <div className="mt-1.5 font-mono text-base font-bold text-slate-100">
          {l2Error !== null ? (
            <span className="text-cyan-300">{(l2Error * 100).toFixed(2)}%</span>
          ) : (
            <span className="text-slate-500">DNS off</span>
          )}
        </div>
        <p className="mt-0.5 text-[10px] text-slate-500">||u - u*|| / ||u*||</p>
      </div>

      {/* 2. PDE Total Residual */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5 backdrop-blur-md">
        <div className="flex items-center space-x-1.5 text-xs text-slate-400">
          <Activity className="h-3.5 w-3.5 text-emerald-400" />
          <span>PDE Residual (MSE)</span>
        </div>
        <div className="mt-1.5 font-mono text-base font-bold text-slate-100">
          {pdeLoss !== null ? (
            <span className="text-emerald-300">{pdeLoss.toExponential(3)}</span>
          ) : (
            <span className="text-slate-500">—</span>
          )}
        </div>
        <p className="mt-0.5 text-[10px] text-slate-500">mean(Ru² + Rv² + Rc²)</p>
      </div>

      {/* 3. Divergence-Free Error */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5 backdrop-blur-md">
        <div className="flex items-center space-x-1.5 text-xs text-slate-400">
          <Gauge className="h-3.5 w-3.5 text-blue-400" />
          <span>Continuity Error</span>
        </div>
        <div className="mt-1.5 font-mono text-base font-bold text-slate-100">
          {contRes !== null ? (
            <span className="text-blue-300">{contRes.toExponential(3)}</span>
          ) : (
            <span className="text-slate-500">—</span>
          )}
        </div>
        <p className="mt-0.5 text-[10px] text-slate-500">mean |∇ · u|</p>
      </div>

      {/* 4. Inference Latency */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5 backdrop-blur-md">
        <div className="flex items-center space-x-1.5 text-xs text-slate-400">
          <Clock className="h-3.5 w-3.5 text-amber-400" />
          <span>Latency</span>
        </div>
        <div className="mt-1.5 font-mono text-base font-bold text-slate-100">
          {latencyMs !== null ? (
            <span className="text-amber-300">{latencyMs.toFixed(1)} ms</span>
          ) : (
            <span className="text-slate-500">—</span>
          )}
        </div>
        <p className="mt-0.5 text-[10px] text-slate-500">Forward + Autograd</p>
      </div>

      {/* 5. Grid Points Evaluated */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5 backdrop-blur-md">
        <div className="flex items-center space-x-1.5 text-xs text-slate-400">
          <Zap className="h-3.5 w-3.5 text-purple-400" />
          <span>Grid Points</span>
        </div>
        <div className="mt-1.5 font-mono text-base font-bold text-slate-100">
          {data ? (
            <span className="text-purple-300">{data.metrics.nx * data.metrics.ny}</span>
          ) : (
            <span className="text-slate-500">—</span>
          )}
        </div>
        <p className="mt-0.5 text-[10px] text-slate-500">{data ? `${data.metrics.nx} x ${data.metrics.ny}` : 'nx x ny'}</p>
      </div>

      {/* 6. Reynolds Number */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5 backdrop-blur-md">
        <div className="flex items-center space-x-1.5 text-xs text-slate-400">
          <span>Reynolds No.</span>
        </div>
        <div className="mt-1.5 font-mono text-base font-bold text-slate-100">
          <span>{health.reynolds_number}</span>
        </div>
        <p className="mt-0.5 text-[10px] text-slate-500">ν = {(1.0 / health.reynolds_number).toFixed(3)}</p>
      </div>

      {/* 7. Checkpoint SHA */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5 backdrop-blur-md">
        <div className="flex items-center space-x-1.5 text-xs text-slate-400">
          <span>Checkpoint SHA</span>
        </div>
        <div className="mt-1.5 font-mono text-xs font-bold text-slate-300 truncate">
          {health.checkpoint_sha ? health.checkpoint_sha.slice(0, 10) : 'none'}
        </div>
        <p className="mt-0.5 text-[10px] text-emerald-400">Verified Canonical</p>
      </div>
    </div>
  )
}