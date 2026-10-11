import React, { useState } from 'react'
import { Award, Terminal, Copy, Check, ShieldCheck } from 'lucide-react'

export const BenchmarkTab: React.FC = () => {
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null)

  const copyToClipboard = (cmd: string) => {
    navigator.clipboard.writeText(cmd)
    setCopiedCmd(cmd)
    setTimeout(() => setCopiedCmd(null), 2500)
  }

  const benchmarkCommands = [
    {
      title: 'Multi-Seed Empirical Benchmark Suite',
      desc: 'Executes the full comparative benchmark across baseline and hard-constrained PINN models (Seeds 0, 1, 2) on authentic Karniadakis cylinder wake DNS data.',
      cmd: 'python scripts/run_experiments.py'
    },
    {
      title: 'Four-Variant Statistical Significance Test',
      desc: 'Calculates Welch t-statistic and Wilcoxon signed-rank p-values comparing Fourier Hard-Constrained PINN against baseline coordinate MLPs.',
      cmd: 'python scripts/benchmark_multi_seed.py'
    },
    {
      title: 'Automated Scientific Smoke Test',
      desc: 'Executes automated forward autograd differentiation, residual checks, and export validation parity tests.',
      cmd: 'python scripts/smoke_test.py'
    }
  ]

  const metricsFramework = [
    {
      metric: 'Relative L2 Velocity Error',
      symbol: '||u - u_DNS||_2 / ||u_DNS||_2',
      purpose: 'Measures global vector reconstruction fidelity across both u and v components over space and time.'
    },
    {
      metric: 'Continuity Divergence Loss',
      symbol: '||∇ · u||',
      purpose: 'Quantifies adherence to incompressible mass conservation (∂u/∂x + ∂v/∂y = 0).'
    },
    {
      metric: 'Navier-Stokes PDE Residual',
      symbol: '||R_u||^2 + ||R_v||^2',
      purpose: 'Measures momentum equation residual norms evaluated across thousands of interior collocation points.'
    },
    {
      metric: 'Solid-Wall No-Slip Condition',
      symbol: '||u(x ∈ ∂Ω)||_2',
      purpose: 'Analytically enforced to exactly 0.00 via distance transform masking D(x, y).'
    }
  ]

  return (
    <div className="space-y-6">
      {/* Integrity Notice Banner */}
      <div className="rounded-xl border border-cyan-800/80 bg-cyan-950/60 p-5 backdrop-blur-xl glass-panel shadow-lg flex items-start space-x-3.5">
        <ShieldCheck className="h-5 w-5 text-cyan-400 mt-0.5 shrink-0" />
        <div className="space-y-1 text-xs text-slate-300">
          <div className="font-bold text-white text-sm font-mono">
            Scientific Integrity Policy: No Fabricated Benchmark Metrics
          </div>
          <p className="leading-relaxed">
            In compliance with rigorous scientific reproduction standards, PINNFlow displays only empirically computed metrics.
            Hard-coded benchmark percentages have been removed. To generate fresh statistical benchmarks with verified confidence intervals,
            execute the empirical evaluation scripts below using authentic reference data.
          </p>
        </div>
      </div>

      {/* Evaluation Protocols Matrix */}
      <div className="rounded-2xl border border-slate-800/90 bg-slate-900/70 p-6 backdrop-blur-xl shadow-2xl glass-panel space-y-4">
        <div className="border-b border-slate-800/80 pb-3">
          <h2 className="text-base font-bold text-white flex items-center gap-2 font-mono">
            <Award className="h-5 w-5 text-cyan-400" />
            Empirical Benchmark Protocol & Evaluation Framework
          </h2>
          <p className="text-xs text-slate-400 mt-1 font-sans">
            Formal mathematical error metrics evaluated against the Karniadakis 2D Cylinder Wake DNS benchmark (Re = 100.0).
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
          {metricsFramework.map((m, idx) => (
            <div key={idx} className="rounded-xl border border-slate-800 bg-slate-950/80 p-4 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-xs text-slate-200">{m.metric}</span>
                <span className="text-[11px] font-mono text-cyan-300 bg-cyan-950/80 px-2 py-0.5 rounded border border-cyan-800">
                  {m.symbol}
                </span>
              </div>
              <p className="text-xs text-slate-400 font-sans leading-relaxed">{m.purpose}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Reproduce Benchmarks Locally */}
      <div className="rounded-2xl border border-slate-800/90 bg-slate-900/70 p-6 backdrop-blur-xl shadow-2xl glass-panel space-y-4">
        <div className="border-b border-slate-800/80 pb-3">
          <h2 className="text-base font-bold text-white flex items-center gap-2 font-mono">
            <Terminal className="h-5 w-5 text-cyan-400" />
            Reproduce Verified Benchmark Results
          </h2>
          <p className="text-xs text-slate-400 mt-1 font-sans">
            Run the reproducible multi-seed evaluation pipelines from the repository root:
          </p>
        </div>

        <div className="space-y-3 pt-1">
          {benchmarkCommands.map((c, idx) => (
            <div key={idx} className="rounded-xl border border-slate-800 bg-slate-950/90 p-4 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-200 font-mono">{c.title}</span>
                <button
                  type="button"
                  onClick={() => copyToClipboard(c.cmd)}
                  className="flex items-center space-x-1 rounded bg-slate-900 border border-slate-700 px-2.5 py-1 text-[11px] font-mono text-slate-300 hover:bg-slate-800 hover:text-white transition"
                >
                  {copiedCmd === c.cmd ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                  <span>{copiedCmd === c.cmd ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <p className="text-xs text-slate-400 font-sans">{c.desc}</p>
              <div className="rounded-lg bg-slate-900/90 border border-slate-800 px-3 py-2 text-xs font-mono text-cyan-300 select-all">
                $ {c.cmd}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
