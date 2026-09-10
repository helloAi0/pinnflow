import React, { useRef, useEffect } from 'react'
import type { FieldResponse, FieldVariable, FieldRequest } from '../types/api'

interface FieldVisualizationProps {
  data: FieldResponse | null
  refData: FieldResponse | null
  params: FieldRequest
  isLoading: boolean
  activeVar: FieldVariable
  setActiveVar: (v: FieldVariable) => void
  compareMode: boolean
  setCompareMode: (v: boolean) => void
  refAvailable: boolean
}

export const FieldVisualization: React.FC<FieldVisualizationProps> = ({
  data,
  refData,
  params,
  isLoading,
  activeVar,
  setActiveVar,
  compareMode,
  setCompareMode,
  refAvailable,
}) => {
  const pinnCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const dnsCanvasRef = useRef<HTMLCanvasElement | null>(null)

  const drawHeatmap = (
    canvas: HTMLCanvasElement,
    gridData: number[][],
    globalMin: number,
    globalMax: number
  ) => {
    const ctx = canvas.getContext('2d')
    if (!ctx || !gridData || gridData.length === 0) return

    const ny = gridData.length
    const nx = gridData[0].length

    canvas.width = nx
    canvas.height = ny

    const range = globalMax - globalMin || 1.0
    const imgData = ctx.createImageData(nx, ny)

    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const val = gridData[r][c]
        const norm = Math.max(0, Math.min(1, (val - globalMin) / range))

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

    // Accurate Cylinder Overlay based on Domain Bounds
    const domainWidth = params.x_max - params.x_min
    const domainHeight = params.y_max - params.y_min
    const cylinderRadius = 0.5 // Real physics config
    const cylinderX = 0.0
    const cylinderY = 0.0

    const cx = ((cylinderX - params.x_min) / domainWidth) * nx
    const cy = ny - ((cylinderY - params.y_min) / domainHeight) * ny // Invert Y for canvas
    const rPx = (cylinderRadius / domainWidth) * nx

    ctx.fillStyle = '#020617'
    ctx.beginPath()
    ctx.arc(cx, cy, rPx, 0, 2 * Math.PI)
    ctx.fill()
    ctx.strokeStyle = '#94a3b8'
    ctx.lineWidth = 1
    ctx.stroke()
  }

  useEffect(() => {
    if (!data) return
    const primaryGrid = data[activeVar] ?? data.u
    const refGrid = compareMode && refData ? (refData[activeVar] ?? refData.u) : null

    // Compute shared min/max for honest comparison
    let min = Infinity
    let max = -Infinity

    const updateBounds = (grid: number[][]) => {
      for (let r = 0; r < grid.length; r++) {
        for (let c = 0; c < grid[r].length; c++) {
          if (grid[r][c] < min) min = grid[r][c]
          if (grid[r][c] > max) max = grid[r][c]
        }
      }
    }

    updateBounds(primaryGrid)
    if (refGrid) updateBounds(refGrid)

    if (pinnCanvasRef.current) {
      drawHeatmap(pinnCanvasRef.current, primaryGrid, min, max)
    }
    if (compareMode && dnsCanvasRef.current && refGrid) {
      drawHeatmap(dnsCanvasRef.current, refGrid, min, max)
    }
  }, [data, refData, activeVar, compareMode, params])

  const variables: { id: FieldVariable; label: string }[] = [
    { id: 'u', label: 'u (Velocity X)' },
    { id: 'v', label: 'v (Velocity Y)' },
    { id: 'p', label: 'p (Pressure)' },
    { id: 'velocity_magnitude', label: '|V| (Magnitude)' },
  ]

  return (
    <div className="flex flex-col space-y-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-5 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] pb-3">
        <div className="flex items-center space-x-1 rounded-lg bg-[var(--slate-950)] p-1 border border-[var(--border)]">
          {variables.map((v) => (
            <button
              key={v.id}
              type="button"
              onClick={() => setActiveVar(v.id)}
              className={`rounded-md px-3 py-1.5 text-xs font-mono font-medium transition-colors ${
                activeVar === v.id
                  ? 'bg-[var(--color-primary)] text-white shadow-sm'
                  : 'text-[var(--muted-foreground)] hover:text-[var(--foreground)]'
              }`}
            >
              {v.label}
            </button>
          ))}
        </div>

        <button
          type="button"
          onClick={() => setCompareMode(!compareMode)}
          disabled={!refAvailable}
          className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
            compareMode
              ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
              : refAvailable 
              ? 'border-[var(--border)] bg-[var(--slate-950)] text-[var(--muted-foreground)] hover:text-[var(--foreground)]'
              : 'border-[var(--slate-800)] bg-[var(--slate-950)] text-[var(--slate-700)] cursor-not-allowed'
          }`}
        >
          {compareMode ? '✓ DNS Comparison Active' : refAvailable ? 'Compare vs DNS Reference' : 'DNS Reference Unavailable'}
        </button>
      </div>

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
            <div className="flex flex-col items-center space-y-2">
              <span className="text-xs font-mono font-semibold text-[var(--muted-foreground)]">
                PINN Model Prediction ({activeVar})
              </span>
              <div className="relative w-full overflow-hidden rounded-md border border-[var(--border)] bg-black">
                <canvas ref={pinnCanvasRef} className="h-72 w-full object-fill image-rendering-pixelated" />
              </div>
            </div>

            {compareMode && (
              <div className="flex flex-col items-center space-y-2">
                <span className="text-xs font-mono font-semibold text-[var(--color-primary)]">
                  DNS Reference (measured)
                </span>
                <div className="relative w-full overflow-hidden rounded-md border border-[var(--border)] bg-black">
                  {refData ? (
                    <canvas ref={dnsCanvasRef} className="h-72 w-full object-fill image-rendering-pixelated" />
                  ) : (
                    <div className="flex h-full w-full items-center justify-center text-xs font-mono text-[var(--muted-foreground)]">
                      Run evaluation to load reference
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}