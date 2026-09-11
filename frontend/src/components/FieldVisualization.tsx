import React, { useRef, useEffect, useState } from 'react'
import { Compass } from 'lucide-react'
import type { FieldResponse, FieldVariable, VisualizationMode } from '../types/api'

interface FieldVisualizationProps {
  data: FieldResponse | null
  refData: FieldResponse | null
  activeVar: FieldVariable
  setActiveVar: (v: FieldVariable) => void
  visMode: VisualizationMode
  setVisMode: (m: VisualizationMode) => void
  isLoading: boolean
  compareMode: boolean
}

function sampleColor(val: number, min: number, max: number, palette: string = 'turbo'): [number, number, number] {
  const norm = max > min ? Math.max(0, Math.min(1, (val - min) / (max - min))) : 0.5
  
  if (palette === 'inferno') {
    const r = Math.floor(255 * Math.pow(norm, 0.7))
    const g = Math.floor(255 * Math.pow(norm, 1.8) * 0.9)
    const b = Math.floor(255 * Math.sin(norm * Math.PI) * 0.5 + 40 * (1 - norm))
    return [r, g, b]
  } else if (palette === 'coolwarm') {
    const r = Math.floor(255 * Math.min(1, Math.max(0, 2 * norm)))
    const b = Math.floor(255 * Math.min(1, Math.max(0, 2 * (1 - norm))))
    const g = Math.floor(255 * (1 - Math.abs(2 * norm - 1)))
    return [r, g, b]
  } else {
    const x = norm
    const r = Math.floor(255 * (0.1357 + x * (4.5155 + x * (-8.5638 + x * 4.9086))))
    const g = Math.floor(255 * (0.0914 + x * (2.1942 + x * (4.8429 + x * (-14.185 + x * 7.645)))))
    const b = Math.floor(255 * (0.1067 + x * (12.573 + x * (-86.801 + x * (268.86 + x * (-338.63 + x * 153.28))))))
    return [Math.max(0, Math.min(255, r)), Math.max(0, Math.min(255, g)), Math.max(0, Math.min(255, b))]
  }
}

export const FieldVisualization: React.FC<FieldVisualizationProps> = ({
  data,
  refData,
  activeVar,
  setActiveVar,
  visMode,
  setVisMode,
  isLoading
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [colormap, setColormap] = useState<'turbo' | 'inferno' | 'coolwarm'>('turbo')
  const [stats, setStats] = useState<{ min: number; max: number; mean: number } | null>(null)

  useEffect(() => {
    if (!canvasRef.current || !data) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let gridToRender: number[][] = []
    
    if (visMode === 'prediction') {
      if (activeVar === 'u') gridToRender = data.u
      else if (activeVar === 'v') gridToRender = data.v
      else if (activeVar === 'p') gridToRender = data.p
      else if (activeVar === 'vorticity') gridToRender = data.vorticity ?? data.u
      else gridToRender = data.velocity_magnitude
    } else if (visMode === 'reference' && refData) {
      if (activeVar === 'u') gridToRender = refData.u
      else if (activeVar === 'v') gridToRender = refData.v
      else if (activeVar === 'p') gridToRender = refData.p
      else if (activeVar === 'vorticity') gridToRender = refData.vorticity ?? refData.u
      else gridToRender = refData.velocity_magnitude
    } else if (visMode === 'absolute_error' && refData) {
      const pGrid = activeVar === 'u' ? data.u : (activeVar === 'v' ? data.v : (activeVar === 'p' ? data.p : data.velocity_magnitude))
      const rGrid = activeVar === 'u' ? refData.u : (activeVar === 'v' ? refData.v : (activeVar === 'p' ? refData.p : refData.velocity_magnitude))
      gridToRender = pGrid.map((row, r) => row.map((val, c) => Math.abs(val - (rGrid[r]?.[c] ?? val))))
    } else {
      gridToRender = data.u.map((row) => row.map(() => data.metrics.pde_loss))
    }

    const ny = gridToRender.length
    const nx = gridToRender[0]?.length || 0
    if (ny === 0 || nx === 0) return

    let min = Infinity, max = -Infinity, sum = 0, count = 0
    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const v = gridToRender[r][c]
        if (!isNaN(v)) {
          if (v < min) min = v
          if (v > max) max = v
          sum += v
          count++
        }
      }
    }
    setStats({ min, max, mean: count > 0 ? sum / count : 0 })

    const width = canvas.width
    const height = canvas.height
    const imgData = ctx.createImageData(width, height)

    for (let py = 0; py < height; py++) {
      const gy = Math.floor(((height - 1 - py) / height) * ny)
      for (let px = 0; px < width; px++) {
        const gx = Math.floor((px / width) * nx)
        const val = gridToRender[gy]?.[gx] ?? 0
        const [r, g, b] = sampleColor(val, min, max, colormap)

        const idx = (py * width + px) * 4
        imgData.data[idx] = r
        imgData.data[idx + 1] = g
        imgData.data[idx + 2] = b
        imgData.data[idx + 3] = 255
      }
    }
    ctx.putImageData(imgData, 0, 0)

    const xMin = data.x[0][0]
    const xMax = data.x[0][nx - 1]
    const yMin = data.y[0][0]
    const yMax = data.y[ny - 1][0]

    const cylX = 0.0
    const cylY = 0.0
    const cylR = 0.5

    if (cylX >= xMin && cylX <= xMax) {
      const canvasCylX = ((cylX - xMin) / (xMax - xMin)) * width
      const canvasCylY = height - ((cylY - yMin) / (yMax - yMin)) * height
      const canvasCylR = (cylR / (xMax - xMin)) * width

      ctx.beginPath()
      ctx.arc(canvasCylX, canvasCylY, canvasCylR, 0, 2 * Math.PI)
      ctx.fillStyle = '#1e293b'
      ctx.fill()
      ctx.lineWidth = 2
      ctx.strokeStyle = '#06b6d4'
      ctx.stroke()
    }
  }, [data, refData, activeVar, visMode, colormap])

  return (
    <div className="flex flex-col space-y-4 rounded-xl border border-slate-800 bg-slate-900/70 p-5 backdrop-blur-md shadow-xl">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex flex-wrap items-center gap-1.5 bg-slate-950 p-1 rounded-lg border border-slate-800">
          <button
            onClick={() => setVisMode('prediction')}
            className={`px-3 py-1 text-xs font-semibold rounded-md transition ${
              visMode === 'prediction' ? 'bg-cyan-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Prediction (PINN)
          </button>
          <button
            onClick={() => setVisMode('reference')}
            disabled={!refData}
            className={`px-3 py-1 text-xs font-semibold rounded-md transition ${
              visMode === 'reference' ? 'bg-cyan-600 text-white shadow' : 'text-slate-400 hover:text-slate-200 disabled:opacity-40'
            }`}
          >
            Reference DNS
          </button>
          <button
            onClick={() => setVisMode('absolute_error')}
            disabled={!refData}
            className={`px-3 py-1 text-xs font-semibold rounded-md transition ${
              visMode === 'absolute_error' ? 'bg-cyan-600 text-white shadow' : 'text-slate-400 hover:text-slate-200 disabled:opacity-40'
            }`}
          >
            Absolute Error
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 font-mono text-xs">
          {(['u', 'v', 'p', 'velocity_magnitude', 'vorticity'] as FieldVariable[]).map((v) => (
            <button
              key={v}
              onClick={() => setActiveVar(v)}
              className={`px-2.5 py-1 rounded transition font-medium ${
                activeVar === v ? 'bg-slate-800 text-cyan-400 border border-slate-700' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {v === 'velocity_magnitude' ? '|u|' : (v === 'vorticity' ? 'ω' : v)}
            </button>
          ))}
        </div>

        <div className="flex items-center space-x-2 text-xs">
          <span className="text-slate-500">Colormap:</span>
          <select
            value={colormap}
            onChange={(e) => setColormap(e.target.value as 'turbo' | 'inferno' | 'coolwarm')}
            className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-300"
          >
            <option value="turbo">Turbo</option>
            <option value="inferno">Inferno</option>
            <option value="coolwarm">Coolwarm</option>
          </select>
        </div>
      </div>

      <div className="relative aspect-[2/1] w-full overflow-hidden rounded-lg border border-slate-800 bg-slate-950 flex items-center justify-center">
        {isLoading && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-slate-950/70 backdrop-blur-sm text-cyan-400">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-cyan-500 border-t-transparent mb-2" />
            <span className="text-xs font-mono">Computing Physical Flow Field...</span>
          </div>
        )}

        {!data && !isLoading && (
          <div className="flex flex-col items-center justify-center text-slate-500 space-y-2">
            <Compass className="h-8 w-8 text-slate-600" />
            <p className="text-xs">Click "Evaluate Physical Field" to execute inference</p>
          </div>
        )}

        <canvas
          ref={canvasRef}
          width={800}
          height={400}
          className="h-full w-full object-contain"
        />
      </div>

      {stats && (
        <div className="flex flex-wrap items-center justify-between text-xs font-mono text-slate-400 px-2">
          <div>
            <span className="text-slate-500">Min:</span> <span className="text-slate-200">{stats.min.toFixed(4)}</span>
          </div>
          <div>
            <span className="text-slate-500">Mean:</span> <span className="text-cyan-300">{stats.mean.toFixed(4)}</span>
          </div>
          <div>
            <span className="text-slate-500">Max:</span> <span className="text-slate-200">{stats.max.toFixed(4)}</span>
          </div>
        </div>
      )}
    </div>
  )
}