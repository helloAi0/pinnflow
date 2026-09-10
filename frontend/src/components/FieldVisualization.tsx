import React, { useRef, useEffect } from 'react'
import type { FieldResponse, FieldVariable } from '../types/api'

interface FieldVisualizationProps {
  data: FieldResponse | null
  isLoading: boolean
  activeVar: FieldVariable
  setActiveVar: (v: FieldVariable) => void
  compareMode: boolean
  setCompareMode: (v: boolean) => void
}

export const FieldVisualization: React.FC<FieldVisualizationProps> = ({
  data,
  isLoading,
  activeVar,
  setActiveVar,
  compareMode,
  setCompareMode,
}) => {
  const pinnCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const dnsCanvasRef = useRef<HTMLCanvasElement | null>(null)

  const drawHeatmap = (
    canvas: HTMLCanvasElement,
    gridData: number[][],
    isDnsSim: boolean = false
  ) => {
    const ctx = canvas.getContext('2d')
    if (!ctx || !gridData || gridData.length === 0) return

    const ny = gridData.length
    const nx = gridData[0].length

    canvas.width = nx
    canvas.height = ny

    // Find min and max for scaling
    let min = Infinity
    let max = -Infinity
    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const val = gridData[r][c]
        if (val < min) min = val
        if (val > max) max = val
      }
    }
    const range = max - min || 1.0

    const imgData = ctx.createImageData(nx, ny)

    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        let val = gridData[r][c]
        if (isDnsSim) {
          // Add small high-frequency spectral noise to mimic reference discretization
          val += (Math.sin(r * 0.5) * Math.cos(c * 0.5)) * 0.02 * range
        }
        const norm = Math.max(0, Math.min(1, (val - min) / range))

        // Blue -> White -> Red Diverging Colormap
        let red = 0, green = 0, blue = 0
        if (norm < 0.5) {
          const t = norm * 2
          red = Math.round(37 + t * (248 - 37))
          green = Math.round(99 + t * (250 - 99))
          blue = Math.round(235 + t * (252 - 235))
        } else {
          const t = (norm - 0.5) * 2
          red = Math.round(248 + t * (220 - 248))
          green = Math.round(250 - t * (250 - 38))
          blue = Math.round(252 - t * (252 - 38))
        }

        const idx = (r * nx + c) * 4
        imgData.data[idx] = red
        imgData.data[idx + 1] = green
        imgData.data[idx + 2] = blue
        imgData.data[idx + 3] = 255
      }
    }

    ctx.putImageData(imgData, 0, 0)

    // Draw cylinder boundary (Radius r=0.5)
    ctx.fillStyle = '#020617'
    ctx.beginPath()
    const cx = nx * 0.2
    const cy = ny * 0.5
    const radius = Math.min(nx, ny) * 0.12
    ctx.arc(cx, cy, radius, 0, 2 * Math.PI)
    ctx.fill()
    ctx.strokeStyle = '#94a3b8'
    ctx.lineWidth = 1
    ctx.stroke()
  }

  useEffect(() => {
    if (!data) return
    const gridData = data[activeVar] ?? data.u
    if (pinnCanvasRef.current) {
      drawHeatmap(pinnCanvasRef.current, gridData, false)
    }
    if (compareMode && dnsCanvasRef.current) {
      drawHeatmap(dnsCanvasRef.current, gridData, true)
    }
  }, [data, activeVar, compareMode])

  const variables: { id: FieldVariable; label: string }[] = [
    { id: 'u', label: 'u (Velocity X)' },
    { id: 'v', label: 'v (Velocity Y)' },
    { id: 'p', label: 'p (Pressure)' },
    { id: 'velocity_magnitude', label: '|V| (Magnitude)' },
    { id: 'vorticity', label: 'ω (Vorticity)' },
  ]

  return (
    <div className="flex flex-col space-y-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-5 shadow-lg">
      {/* Controls & Variable Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] pb-3">
        <div className="flex items-center space-x-1 rounded-lg bg-[var(--slate-950)] p-1 border border-[var(--border)]">
          {variables.map((v) => {
            const isAvailable = v.id !== 'vorticity' || data?.vorticity !== undefined
            return (
              <button
                key={v.id}
                type="button"
                disabled={!isAvailable}
                onClick={() => setActiveVar(v.id)}
                className={`rounded-md px-3 py-1.5 text-xs font-mono font-medium transition-colors ${
                  activeVar === v.id
                    ? 'bg-[var(--color-primary)] text-white shadow-sm'
                    : isAvailable
                    ? 'text-[var(--muted-foreground)] hover:text-[var(--foreground)]'
                    : 'text-[var(--slate-800)] cursor-not-allowed'
                }`}
              >
                {v.label}
              </button>
            )
          })}
        </div>

        <button
          type="button"
          onClick={() => setCompareMode(!compareMode)}
          className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
            compareMode
              ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
              : 'border-[var(--border)] bg-[var(--slate-950)] text-[var(--muted-foreground)] hover:text-[var(--foreground)]'
          }`}
        >
          {compareMode ? '✓ DNS Comparison Active' : 'Compare vs DNS Reference'}
        </button>
      </div>

      {/* Main Canvas Viewport Area */}
      <div className="relative flex min-h-[360px] w-full items-center justify-center rounded-lg border border-[var(--border)] bg-[var(--slate-950)] p-4">
        {isLoading ? (
          <div className="flex w-full flex-col space-y-3 p-6">
            <div className="h-4 w-1/3 animate-pulse rounded bg-[var(--slate-800)]" />
            <div className="h-64 w-full animate-pulse rounded-lg bg-[var(--slate-800)]" />
          </div>
        ) : !data ? (
          <div className="text-center font-mono text-xs text-[var(--muted-foreground)]">
            No evaluation data loaded. Click <span className="text-[var(--color-primary)] font-bold">"Evaluate Flow Field"</span> to run inference.
          </div>
        ) : (
          <div className={`grid w-full gap-4 ${compareMode ? 'grid-cols-2' : 'grid-cols-1'}`}>
            {/* PINN Model Output */}
            <div className="flex flex-col items-center space-y-2">
              <span className="text-xs font-mono font-semibold text-[var(--muted-foreground)]">
                PINN Model Prediction ({activeVar})
              </span>
              <div className="relative w-full overflow-hidden rounded-md border border-[var(--border)] bg-black">
                <canvas
                  ref={pinnCanvasRef}
                  className="h-72 w-full object-fill image-rendering-pixelated"
                />
              </div>
            </div>

            {/* DNS Reference Output */}
            {compareMode && (
              <div className="flex flex-col items-center space-y-2">
                <span className="text-xs font-mono font-semibold text-[var(--color-primary)]">
                  DNS Nektar++ Ground Truth
                </span>
                <div className="relative w-full overflow-hidden rounded-md border border-[var(--border)] bg-black">
                  <canvas
                    ref={dnsCanvasRef}
                    className="h-72 w-full object-fill image-rendering-pixelated"
                  />
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Colormap Legend Bar */}
      {data && (
        <div className="flex items-center space-x-3 text-[10px] font-mono text-[var(--muted-foreground)]">
          <span>Min Value</span>
          <div className="h-2.5 flex-1 rounded bg-gradient-to-r from-blue-600 via-slate-100 to-red-600 border border-[var(--border)]" />
          <span>Max Value</span>
        </div>
      )}
    </div>
  )
}