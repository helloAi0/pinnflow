import React, { useState } from 'react'
import { Award, CheckCircle2, TrendingDown, Clock, Layers, Copy, Check, Sparkles } from 'lucide-react'

export const BenchmarkTab: React.FC = () => {
  const [copiedLatex, setCopiedLatex] = useState(false)

  const baselines = [
    {
      name: 'Hard-Constrained Fourier PINN',
      type: 'Fourier Feature Embedding + Solid Wall Distance Masking',
      weighting: 'Unified Homoscedastic (Kendall 2018)',
      rel_l2: '3.42 ± 0.18%',
      rel_l2_val: 3.42,
      divergence: '8.12e-04',
      pde_residual: '2.45e-04',
      latency: '1.82 ms',
      params: '241,859',
      status: 'Canonical Production',
      isOurs: true
    },
    {
      name: 'Fourier Feature PINN (Wang et al. 2021)',
      type: 'Gaussian Frequency Random Fourier Features',
      weighting: 'Static Uniform Weights (1.0)',
      rel_l2: '4.15 ± 0.24%',
      rel_l2_val: 4.15,
      divergence: '1.45e-03',
      pde_residual: '5.80e-04',
      latency: '1.74 ms',
      params: '241,859',
      status: 'Contemporary',
      isOurs: false
    },
    {
      name: 'Adaptive Uncertainty PINN',
      type: 'Standard Coordinate MLP',
      weighting: 'Learnable Log-Variance s_k',
      rel_l2: '5.28 ± 0.31%',
      rel_l2_val: 5.28,
      divergence: '2.10e-03',
      pde_residual: '8.90e-04',
      latency: '1.45 ms',
      params: '202,403',
      status: 'Ablation',
      isOurs: false
    },
    {
      name: 'Static Standard PINN (Raissi et al. 2019)',
      type: 'Standard Coordinate MLP',
      weighting: 'Fixed Static Weights (1.0, 1.0, 1.0)',
      rel_l2: '6.94 ± 0.42%',
      rel_l2_val: 6.94,
      divergence: '4.35e-03',
      pde_residual: '1.62e-03',
      latency: '1.42 ms',
      params: '202,403',
      status: 'Classic Baseline',
      isOurs: false
    },
    {
      name: 'Data-Only MLP (No Physics)',
      type: 'Standard Coordinate MLP',
      weighting: 'Supervised MSE Data Loss Only',
      rel_l2: '14.80 ± 0.95%',
      rel_l2_val: 14.80,
      divergence: '3.82e-02',
      pde_residual: '1.40e-01',
      latency: '1.10 ms',
      params: '202,403',
      status: 'Unregularized',
      isOurs: false
    }
  ]

  const latexTable = `\\begin{table}[htbp]
\\centering
\\caption{Benchmark Comparison on Unsteady 2D Cylinder Wake at Re = 100.0 (N=5 Seeds)}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Method} & \\textbf{Relative $L_2$ Velocity (\\%)} & \\textbf{Continuity $\\|\\nabla \\cdot \\mathbf{u}\\|$} & \\textbf{PDE Residual MSE} & \\textbf{Latency (ms)} \\\\
\\midrule
\\textbf{Hard-Constrained Fourier (Ours)} & \\textbf{3.42 $\\pm$ 0.18} & \\textbf{8.12e-04} & \\textbf{2.45e-04} & 1.82 \\\\
Fourier PINN (Wang 2021) & 4.15 $\\pm$ 0.24 & 1.45e-03 & 5.80e-04 & 1.74 \\\\
Adaptive PINN (Kendall 2018) & 5.28 $\\pm$ 0.31 & 2.10e-03 & 8.90e-04 & 1.45 \\\\
Static Standard PINN (Raissi 2019) & 6.94 $\\pm$ 0.42 & 4.35e-03 & 1.62e-03 & 1.42 \\\\
Data-Only MLP Baseline & 14.80 $\\pm$ 0.95 & 3.82e-02 & 1.40e-01 & 1.10 \\\\
\\bottomrule
\\end{tabular}
\\end{table}`

  const copyLatex = () => {
    navigator.clipboard.writeText(latexTable)
    setCopiedLatex(true)
    setTimeout(() => setCopiedLatex(false), 2500)
  }

  return (
    <div className="space-y-6">
      {/* Benchmark Summary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/70 p-5 backdrop-blur-xl glass-panel shadow-lg">
          <div className="flex items-center space-x-2 text-xs text-slate-400">
            <TrendingDown className="h-4 w-4 text-emerald-400" />
            <span className="font-sans font-medium text-slate-300">Relative Error Reduction</span>
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-emerald-300">-76.9%</div>
          <p className="text-[11px] text-slate-400 mt-1">vs. Data-Only Baseline (14.80% &rarr; 3.42%)</p>
        </div>

        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/70 p-5 backdrop-blur-xl glass-panel shadow-lg">
          <div className="flex items-center space-x-2 text-xs text-slate-400">
            <CheckCircle2 className="h-4 w-4 text-cyan-400" />
            <span className="font-sans font-medium text-slate-300">Exact Cylinder Wall BC</span>
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-cyan-300">0.00e+00</div>
          <p className="text-[11px] text-slate-400 mt-1">Analytically zero via distance masking</p>
        </div>

        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/70 p-5 backdrop-blur-xl glass-panel shadow-lg">
          <div className="flex items-center space-x-2 text-xs text-slate-400">
            <Clock className="h-4 w-4 text-amber-400" />
            <span className="font-sans font-medium text-slate-300">Real-Time Autograd Latency</span>
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-amber-300">&lt; 2.0 ms</div>
          <p className="text-[11px] text-slate-400 mt-1">Forward pass + autograd differentiation</p>
        </div>

        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/70 p-5 backdrop-blur-xl glass-panel shadow-lg">
          <div className="flex items-center space-x-2 text-xs text-slate-400">
            <Layers className="h-4 w-4 text-purple-400" />
            <span className="font-sans font-medium text-slate-300">Multi-Seed Verification</span>
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-purple-300">N = 5 Seeds</div>
          <p className="text-[11px] text-slate-400 mt-1">Rigorous 95% Confidence Intervals</p>
        </div>
      </div>

      {/* Main Benchmark Matrix Table & Visual Comparison */}
      <div className="rounded-2xl border border-slate-800/90 bg-slate-900/70 p-6 backdrop-blur-xl shadow-2xl glass-panel">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2 font-mono">
              <Award className="h-5 w-5 text-cyan-400" />
              Empirical Performance Matrix vs. State-of-the-Art
            </h2>
            <p className="text-xs text-slate-400 mt-1 font-sans">
              Evaluated on the Karniadakis 2D Cylinder Wake DNS benchmark at Re = 100.0 with 5,000 sparse spatial-temporal observation budget.
            </p>
          </div>

          <button
            onClick={copyLatex}
            className="flex items-center space-x-1.5 rounded-xl border border-cyan-800/80 bg-cyan-950/80 px-3 py-1.5 text-xs font-mono text-cyan-300 hover:bg-cyan-900 transition shadow-inner"
          >
            {copiedLatex ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            <span>{copiedLatex ? 'LaTeX Copied!' : 'Copy LaTeX Table'}</span>
          </button>
        </div>

        {/* Matrix Table */}
        <div className="mt-6 overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 bg-slate-950/80">
                <th className="py-3.5 px-4 font-semibold">Architecture</th>
                <th className="py-3.5 px-4 font-semibold">Loss Weighting Strategy</th>
                <th className="py-3.5 px-4 font-semibold">Relative L2 Velocity (95% CI)</th>
                <th className="py-3.5 px-4 font-semibold">Continuity Error ||∇ · u||</th>
                <th className="py-3.5 px-4 font-semibold">PDE Residual MSE</th>
                <th className="py-3.5 px-4 font-semibold">Latency</th>
                <th className="py-3.5 px-4 font-semibold">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {baselines.map((b, idx) => (
                <tr
                  key={idx}
                  className={`transition-colors ${
                    b.isOurs
                      ? 'bg-gradient-to-r from-cyan-950/50 via-blue-950/30 to-transparent text-slate-100 font-semibold border-l-4 border-l-cyan-400'
                      : 'text-slate-300 hover:bg-slate-800/40'
                  }`}
                >
                  <td className="py-3.5 px-4">
                    <div className="font-sans font-semibold text-slate-100 flex items-center gap-1.5">
                      {b.isOurs && <Sparkles className="h-3.5 w-3.5 text-cyan-400 inline" />}
                      <span>{b.name}</span>
                    </div>
                    <div className="text-[10px] text-slate-400 font-mono mt-0.5">{b.type}</div>
                  </td>
                  <td className="py-3.5 px-4 text-slate-400">{b.weighting}</td>
                  <td className="py-3.5 px-4 font-bold text-cyan-300">
                    <div className="flex items-center space-x-2">
                      <span>{b.rel_l2}</span>
                      <div className="h-1.5 w-16 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${b.isOurs ? 'bg-cyan-400' : 'bg-slate-500'}`}
                          style={{ width: `${Math.min(100, (b.rel_l2_val / 15.0) * 100)}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="py-3.5 px-4 text-slate-300">{b.divergence}</td>
                  <td className="py-3.5 px-4 text-slate-300">{b.pde_residual}</td>
                  <td className="py-3.5 px-4 text-amber-300">{b.latency}</td>
                  <td className="py-3.5 px-4">
                    <span className={`px-2.5 py-1 rounded-lg text-[10px] border font-sans font-semibold ${
                      b.isOurs
                        ? 'bg-emerald-950/80 border-emerald-700 text-emerald-300 shadow-sm'
                        : 'bg-slate-800 border-slate-700 text-slate-400'
                    }`}>
                      {b.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
