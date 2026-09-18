import React, { useRef, useEffect, useState, useCallback } from 'react'
import { Compass, Crosshair, Camera } from 'lucide-react'
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

type ColormapType = 'turbo' | 'viridis' | 'inferno' | 'coolwarm' | 'plasma'

function sampleColor(val: number, min: number, max: number, palette: ColormapType = 'turbo'): [number, number, number] {
  const norm = max > min ? Math.max(0, Math.min(1, (val - min) / (max - min))) : 0.5

  if (palette === 'viridis') {
    const x = norm
    const r = Math.floor(255 * (0.267 + x * (0.005 + x * (0.329 + x * (1.17 - x * 0.77)))))
    const g = Math.floor(255 * (0.004 + x * (1.405 - x * (0.835 + x * (0.505 - x * 0.9)))))
    const b = Math.floor(255 * (0.329 + x * (1.107 - x * (2.813 + x * (2.45 - x * 0.73)))))
    return [Math.max(0, Math.min(255, r)), Math.max(0, Math.min(255, g)), Math.max(0, Math.min(255, b))]
  } else if (palette === 'plasma') {
    const x = norm
    const r = Math.floor(255 * (0.058 + x * (2.55 - x * (1.92 - x * 0.32))))
    const g = Math.floor(255 * (0.015 + x * (0.28 + x * (2.28 - x * 1.57))))
    const b = Math.floor(255 * (0.53 + x * (0.88 - x * (3.14 - x * 1.73))))
    return [Math.max(0, Math.min(255, r)), Math.max(0, Math.min(255, g)), Math.max(0, Math.min(255, b))]
  } else if (palette === 'inferno') {
    const r = Math.floor(255 * Math.pow(norm, 0.7))
    const g = Math.floor(255 * Math.pow(norm, 1.8) * 0.9)
    const b = Math.floor(255 * Math.sin(norm * Math.PI) * 0.5 + 40 * (1 - norm))
    return [Math.max(0, Math.min(255, r)), Math.max(0, Math.min(255, g)), Math.max(0, Math.min(255, b))]
  } else if (palette === 'coolwarm') {
    const r = Math.floor(255 * Math.min(1, Math.max(0, 2 * norm)))
    const b = Math.floor(255 * Math.min(1, Math.max(0, 2 * (1 - norm))))
    const g = Math.floor(255 * (1 - Math.abs(2 * norm - 1)))
    return [Math.max(0, Math.min(255, r)), Math.max(0, Math.min(255, g)), Math.max(0, Math.min(255, b))]
  } else {
    // Turbo
    const x = norm
    const r = Math.floor(255 * (0.1357 + x * (4.5155 + x * (-8.5638 + x * 4.9086))))
    const g = Math.floor(255 * (0.0914 + x * (2.1942 + x * (4.8429 + x * (-14.185 + x * 7.645)))))
    const b = Math.floor(255 * (0.1067 + x * (12.573 + x * (-86.801 + x * (268.86 + x * (-338.63 + x * 153.28))))))
    return [Math.max(0, Math.min(255, r)), Math.max(0, Math.min(255, g)), Math.max(0, Math.min(255, b))]
  }
}

interface ProbeInfo {
  x: number
  y: number
  u: number
  v: number
  p: number
  mag: number
  vorticity?: number
  px: number
  py: number
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
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [colormap, setColormap] = useState<ColormapType>('turbo')
  const [showVectors, setShowVectors] = useState(false)
  const [showGridLines, setShowGridLines] = useState(false)
  const [showStreamlines, setShowStreamlines] = useState(true)
  const [stats, setStats] = useState<{ min: number; max: number; mean: number } | null>(null)
  const [probe, setProbe] = useState<ProbeInfo | null>(null)
  const [pinnedProbe, setPinnedProbe] = useState<ProbeInfo | null>(null)
  const [hoverInside, setHoverInside] = useState(false)

  const renderCanvas = useCallback(() => {
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

    let min = Infinity
    let max = -Infinity
    let sum = 0
    let count = 0
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

    // Grid Lines Overlay
    if (showGridLines) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)'
      ctx.lineWidth = 1
      const stepX = width / 12
      const stepY = height / 6
      for (let x = 0; x < width; x += stepX) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, height)
        ctx.stroke()
      }
      for (let y = 0; y < height; y += stepY) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(width, y)
        ctx.stroke()
      }
    }

    // Streamline Trajectory Traces
    if (showStreamlines && data) {
      ctx.lineWidth = 1.2
      const numLines = 14
      for (let i = 0; i < numLines; i++) {
        let curY = ((i + 0.5) / numLines) * (ny - 1)
        let curX = 0
        ctx.beginPath()
        ctx.strokeStyle = `rgba(255, 255, 255, ${0.12 + (i % 3) * 0.05})`

        const startPx = (curX / nx) * width
        const startPy = height - (curY / ny) * height
        ctx.moveTo(startPx, startPy)

        for (let step = 0; step < 80; step++) {
          const gx = Math.min(nx - 1, Math.max(0, Math.floor(curX)))
          const gy = Math.min(ny - 1, Math.max(0, Math.floor(curY)))

          const uVal = data.u[gy]?.[gx] ?? 1.0
          const vVal = data.v[gy]?.[gx] ?? 0.0

          curX += uVal * 1.5
          curY += vVal * 1.5

          if (curX >= nx - 1 || curY < 0 || curY >= ny - 1) break

          const px = (curX / nx) * width
          const py = height - (curY / ny) * height
          ctx.lineTo(px, py)
        }
        ctx.stroke()
      }
    }

    // Velocity Vector Quivers Overlay
    if (showVectors && data) {
      const sampleStrideX = Math.max(1, Math.floor(nx / 24))
      const sampleStrideY = Math.max(1, Math.floor(ny / 12))
      const maxLen = 14

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.75)'
      ctx.fillStyle = '#06b6d4'
      ctx.lineWidth = 1.3

      for (let gy = 0; gy < ny; gy += sampleStrideY) {
        for (let gx = 0; gx < nx; gx += sampleStrideX) {
          const uVal = data.u[gy][gx]
          const vVal = data.v[gy][gx]
          const mag = Math.sqrt(uVal * uVal + vVal * vVal) || 0.001

          const px = ((gx + 0.5) / nx) * width
          const py = height - ((gy + 0.5) / ny) * height

          const dx = (uVal / mag) * Math.min(maxLen, mag * 8)
          const dy = -(vVal / mag) * Math.min(maxLen, mag * 8)

          ctx.beginPath()
          ctx.moveTo(px, py)
          ctx.lineTo(px + dx, py + dy)
          ctx.stroke()

          ctx.beginPath()
          ctx.arc(px + dx, py + dy, 1.8, 0, 2 * Math.PI)
          ctx.fill()
        }
      }
    }

    // Solid Cylinder Boundary Rendering
    const xMin = data.x[0][0]
    const xMax = data.x[0][nx - 1]
    const yMin = data.y[0][0]
    const yMax = data.y[ny - 1][0]

    const cylX = 0.0
    const cylY = 0.0
    const cylR = 0.5

    if (cylX >= xMin - cylR && cylX <= xMax + cylR) {
      const canvasCylX = ((cylX - xMin) / (xMax - xMin)) * width
      const canvasCylY = height - ((cylY - yMin) / (yMax - yMin)) * height
      const canvasCylR = (cylR / (xMax - xMin)) * width

      ctx.save()
      ctx.beginPath()
      ctx.arc(canvasCylX, canvasCylY, canvasCylR, 0, 2 * Math.PI)
      ctx.fillStyle = '#030712'
      ctx.shadowColor = '#06b6d4'
      ctx.shadowBlur = 12
      ctx.fill()
      ctx.lineWidth = 3
      ctx.strokeStyle = '#22d3ee'
      ctx.stroke()

      // Cylinder cross mark
      ctx.fillStyle = '#06b6d4'
      ctx.font = 'bold 10px monospace'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText('R=0.5', canvasCylX, canvasCylY)
      ctx.restore()
    }

    // Render Pinned Probe Marker if set
    if (pinnedProbe) {
      ctx.save()
      ctx.beginPath()
      ctx.arc(pinnedProbe.px, pinnedProbe.py, 6, 0, 2 * Math.PI)
      ctx.fillStyle = 'rgba(6, 182, 212, 0.4)'
      ctx.fill()
      ctx.lineWidth = 2
      ctx.strokeStyle = '#00f2fe'
      ctx.stroke()

      ctx.beginPath()
      ctx.arc(pinnedProbe.px, pinnedProbe.py, 2, 0, 2 * Math.PI)
      ctx.fillStyle = '#ffffff'
      ctx.fill()
      ctx.restore()
    }
  }, [data, refData, activeVar, visMode, colormap, showVectors, showGridLines, showStreamlines, pinnedProbe])

  useEffect(() => {
    renderCanvas()
  }, [renderCanvas])

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !data) return
    const rect = canvasRef.current.getBoundingClientRect()
    const clientX = e.clientX - rect.left
    const clientY = e.clientY - rect.top

    const width = rect.width
    const height = rect.height

    const nx = data.u[0].length
    const ny = data.u.length

    const gx = Math.min(nx - 1, Math.max(0, Math.floor((clientX / width) * nx)))
    const gy = Math.min(ny - 1, Math.max(0, Math.floor(((height - clientY) / height) * ny)))

    const xVal = data.x[gy]?.[gx] ?? 0
    const yVal = data.y[gy]?.[gx] ?? 0
    const uVal = data.u[gy]?.[gx] ?? 0
    const vVal = data.v[gy]?.[gx] ?? 0
    const pVal = data.p[gy]?.[gx] ?? 0
    const mag = data.velocity_magnitude[gy]?.[gx] ?? Math.sqrt(uVal * uVal + vVal * vVal)
    const vort = data.vorticity?.[gy]?.[gx]

    setProbe({
      x: xVal,
      y: yVal,
      u: uVal,
      v: vVal,
      p: pVal,
      mag,
      vorticity: vort,
      px: clientX,
      py: clientY
    })
    setHoverInside(true)
  }

  const handleCanvasClick = () => {
    if (!probe) return
    if (pinnedProbe && Math.abs(pinnedProbe.px - probe.px) < 10 && Math.abs(pinnedProbe.py - probe.py) < 10) {
      setPinnedProbe(null)
    } else {
      setPinnedProbe({ ...probe })
    }
  }

  const handleMouseLeave = () => {
    setHoverInside(false)
    setProbe(null)
  }

  const handleExportPNG = () => {
    if (!canvasRef.current) return
    const link = document.createElement('a')
    link.download = `pinnflow_${activeVar}_${visMode}_t${data?.metrics.time_snapshot.toFixed(1) || '0.0'}.png`
    link.href = canvasRef.current.toDataURL('image/png')
    link.click()
  }

  const variableLabels: Record<FieldVariable, { label: string; symbol: string; unit: string }> = {
    velocity_magnitude: { label: 'Velocity Magnitude', symbol: '|u|', unit: 'm/s' },
    u: { label: 'Streamwise Velocity', symbol: 'u', unit: 'm/s' },
    v: { label: 'Transverse Velocity', symbol: 'v', unit: 'm/s' },
    p: { label: 'Static Pressure', symbol: 'p', unit: 'Pa' },
    vorticity: { label: 'Vorticity', symbol: 'ω', unit: '1/s' }
  }

  return (
    <div ref={containerRef} className="flex flex-col space-y-4 rounded-2xl border border-slate-800/80 bg-slate-900/60 p-5 backdrop-blur-2xl shadow-2xl glass-panel relative">
      {/* Visual Controls Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3.5">
        {/* Mode Selector */}
        <div className="flex items-center gap-1 bg-slate-950/90 p-1 rounded-xl border border-slate-800 shadow-inner">
          <button
            onClick={() => setVisMode('prediction')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              visMode === 'prediction' ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-900/40' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Prediction (PINN)
          </button>
          <button
            onClick={() => setVisMode('reference')}
            disabled={!refData}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              visMode === 'reference' ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-900/40' : 'text-slate-400 hover:text-slate-200 disabled:opacity-40 disabled:cursor-not-allowed'
            }`}
          >
            Reference DNS
          </button>
          <button
            onClick={() => setVisMode('absolute_error')}
            disabled={!refData}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              visMode === 'absolute_error' ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-900/40' : 'text-slate-400 hover:text-slate-200 disabled:opacity-40 disabled:cursor-not-allowed'
            }`}
          >
            Absolute Error
          </button>
        </div>

        {/* Physical Variable Selector */}
        <div className="flex items-center gap-1 bg-slate-950/90 p-1 rounded-xl border border-slate-800 font-mono text-xs shadow-inner">
          {(['velocity_magnitude', 'u', 'v', 'p', 'vorticity'] as FieldVariable[]).map((v) => (
            <button
              key={v}
              onClick={() => setActiveVar(v)}
              className={`px-2.5 py-1 rounded-lg transition-all font-medium ${
                activeVar === v
                  ? 'bg-gradient-to-r from-cyan-950 to-blue-950 text-cyan-300 border border-cyan-700/60 shadow-inner font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title={variableLabels[v].label}
            >
              {variableLabels[v].symbol}
            </button>
          ))}
        </div>

        {/* Colormap & Display Toggles */}
        <div className="flex items-center space-x-2 font-mono text-xs">
          <select
            value={colormap}
            onChange={(e) => setColormap(e.target.value as ColormapType)}
            className="rounded-lg border border-slate-700/80 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-300 focus:border-cyan-500 focus:outline-none shadow-sm"
          >
            <option value="turbo">Turbo</option>
            <option value="viridis">Viridis</option>
            <option value="inferno">Inferno</option>
            <option value="plasma">Plasma</option>
            <option value="coolwarm">Coolwarm</option>
          </select>

          <button
            onClick={() => setShowStreamlines(!showStreamlines)}
            className={`px-2.5 py-1.5 rounded-lg border transition shadow-sm ${
              showStreamlines
                ? 'bg-cyan-950 border-cyan-700 text-cyan-300 font-semibold'
                : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
            title="Toggle Fluid Streamlines"
          >
            Streamlines
          </button>

          <button
            onClick={() => setShowVectors(!showVectors)}
            className={`px-2.5 py-1.5 rounded-lg border transition shadow-sm ${
              showVectors
                ? 'bg-cyan-950 border-cyan-700 text-cyan-300 font-semibold'
                : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
            title="Toggle Velocity Vector Quiver"
          >
            Vectors
          </button>

          <button
            onClick={() => setShowGridLines(!showGridLines)}
            className={`px-2.5 py-1.5 rounded-lg border transition shadow-sm ${
              showGridLines
                ? 'bg-cyan-950 border-cyan-700 text-cyan-300 font-semibold'
                : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
            title="Toggle Mesh Grid"
          >
            Grid
          </button>

          <button
            onClick={handleExportPNG}
            disabled={!data}
            className="p-2 rounded-lg border border-slate-800 bg-slate-950 text-slate-400 hover:text-cyan-300 hover:border-slate-700 transition disabled:opacity-40"
            title="Snapshot Canvas as High-Res PNG"
          >
            <Camera className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Main Fluid Canvas Viewport */}
      <div className="relative aspect-[2/1] w-full overflow-hidden rounded-xl border border-slate-800/90 bg-slate-950 flex items-center justify-center shadow-2xl group">
        {isLoading && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-slate-950/85 backdrop-blur-md text-cyan-400">
            <div className="relative flex h-14 w-14 items-center justify-center mb-3">
              <div className="absolute h-full w-full animate-ping rounded-full bg-cyan-500/20" />
              <div className="h-10 w-10 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />
            </div>
            <span className="text-xs font-mono font-semibold tracking-wider">Computing Autograd Navier-Stokes Mesh...</span>
            <span className="text-[10px] text-slate-400 mt-1 font-mono">Evaluating pointwise momentum & continuity operators</span>
          </div>
        )}

        {!data && !isLoading && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center text-slate-500 space-y-3 text-center px-4 pointer-events-none">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-cyan-950/40 border border-cyan-800/60 text-cyan-400">
              <Compass className="h-6 w-6 animate-pulse" />
            </div>
            <div>
              <p className="text-xs font-mono text-slate-300 font-semibold">Scientific Instrument Ready</p>
              <p className="text-[11px] text-slate-500 font-mono mt-0.5 max-w-sm">
                Configure experiment parameters on the left and click &quot;Evaluate Physical Field&quot; to execute real-time autograd inference.
              </p>
            </div>
          </div>
        )}

        <canvas
          ref={canvasRef}
          width={900}
          height={450}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleCanvasClick}
          className="h-full w-full object-contain cursor-crosshair"
        />

        {/* Floating Probe HUD */}
        {probe && hoverInside && !pinnedProbe && (
          <div
            className="pointer-events-none absolute z-20 rounded-xl border border-cyan-500/50 bg-slate-950/95 p-3 text-xs font-mono text-slate-200 shadow-2xl backdrop-blur-xl transition-opacity duration-150"
            style={{
              left: `${Math.min(probe.px + 15, (containerRef.current?.clientWidth || 600) - 220)}px`,
              top: `${Math.min(probe.py + 15, 230)}px`
            }}
          >
            <div className="flex items-center space-x-1.5 border-b border-slate-800 pb-1.5 text-cyan-400 font-bold">
              <Crosshair className="h-3.5 w-3.5" />
              <span>Probe (x={probe.x.toFixed(2)}, y={probe.y.toFixed(2)})</span>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
              <div><span className="text-slate-500">|u|:</span> <span className="text-cyan-300 font-bold">{probe.mag.toFixed(3)} m/s</span></div>
              <div><span className="text-slate-500">p:</span> <span className="text-slate-200">{probe.p.toFixed(3)} Pa</span></div>
              <div><span className="text-slate-500">u:</span> <span className="text-slate-300">{probe.u.toFixed(3)}</span></div>
              <div><span className="text-slate-500">v:</span> <span className="text-slate-300">{probe.v.toFixed(3)}</span></div>
              {probe.vorticity !== undefined && (
                <div className="col-span-2 pt-0.5 border-t border-slate-800/60"><span className="text-slate-500">Vorticity (ω):</span> <span className="text-purple-300 font-bold">{probe.vorticity.toFixed(3)} 1/s</span></div>
              )}
            </div>
          </div>
        )}

        {/* Active Variable & Status Badge */}
        {data && (
          <div className="absolute top-3 left-3 pointer-events-none flex items-center space-x-2">
            <div className="rounded-lg bg-slate-950/90 px-3 py-1 text-[11px] font-mono border border-slate-800 text-slate-300 backdrop-blur-md shadow-md">
              <span className="text-slate-500">Field:</span> <span className="text-cyan-300 font-bold">{variableLabels[activeVar].label}</span>
            </div>
            <div className="rounded-lg bg-slate-950/90 px-2.5 py-1 text-[11px] font-mono border border-slate-800 text-slate-400 backdrop-blur-md">
              <span>t = {data.metrics.time_snapshot.toFixed(1)}s</span>
            </div>
          </div>
        )}
      </div>

      {/* Pinned Probe Coordinate Card (if user clicked to pin) */}
      {pinnedProbe && (
        <div className="rounded-xl border border-cyan-500/40 bg-slate-950/90 p-3.5 text-xs font-mono shadow-lg flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-2 text-cyan-300 font-bold">
            <Crosshair className="h-4 w-4" />
            <span>Pinned Spatial Probe at ({pinnedProbe.x.toFixed(2)}, {pinnedProbe.y.toFixed(2)})</span>
          </div>
          <div className="flex flex-wrap items-center gap-4 text-[11px]">
            <div><span className="text-slate-500">|u|:</span> <span className="text-cyan-300 font-semibold">{pinnedProbe.mag.toFixed(3)} m/s</span></div>
            <div><span className="text-slate-500">u:</span> <span className="text-slate-200">{pinnedProbe.u.toFixed(3)}</span></div>
            <div><span className="text-slate-500">v:</span> <span className="text-slate-200">{pinnedProbe.v.toFixed(3)}</span></div>
            <div><span className="text-slate-500">p:</span> <span className="text-slate-200">{pinnedProbe.p.toFixed(3)}</span></div>
            {pinnedProbe.vorticity !== undefined && (
              <div><span className="text-slate-500">ω:</span> <span className="text-purple-300 font-semibold">{pinnedProbe.vorticity.toFixed(3)}</span></div>
            )}
          </div>
          <button
            onClick={() => setPinnedProbe(null)}
            className="text-[10px] px-2 py-1 rounded bg-slate-900 text-slate-400 hover:text-white border border-slate-800"
          >
            Clear Pin
          </button>
        </div>
      )}

      {/* Continuous Colorbar & Scale Statistics */}
      {stats && (
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-xs font-mono text-slate-400 px-1 pt-1">
          {/* Continuous Colorbar Preview */}
          <div className="flex items-center space-x-2 w-full sm:w-auto">
            <span className="text-[11px] text-slate-400 font-bold">{stats.min.toFixed(3)}</span>
            <div
              className="h-3 w-40 rounded-full border border-slate-700 shadow-inner"
              style={{
                background: colormap === 'turbo'
                  ? 'linear-gradient(to right, #30123b, #4686fb, #1ae4b6, #a2fc3c, #fbb41a, #e4460a, #7a0403)'
                  : colormap === 'viridis'
                  ? 'linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725)'
                  : colormap === 'inferno'
                  ? 'linear-gradient(to right, #000004, #57106e, #bb3754, #f98e09, #fcffa4)'
                  : colormap === 'plasma'
                  ? 'linear-gradient(to right, #0d0887, #6a00a8, #b12a90, #e16462, #fca636, #f0f921)'
                  : 'linear-gradient(to right, #0000ff, #ffffff, #ff0000)'
              }}
            />
            <span className="text-[11px] text-slate-400 font-bold">{stats.max.toFixed(3)}</span>
            <span className="text-[10px] text-cyan-400 font-semibold">[{variableLabels[activeVar].unit}]</span>
          </div>

          <div className="flex items-center space-x-4 text-[11px]">
            <div><span className="text-slate-500">Min:</span> <span className="text-slate-200">{stats.min.toFixed(4)}</span></div>
            <div><span className="text-slate-500">Mean:</span> <span className="text-cyan-300 font-bold">{stats.mean.toFixed(4)}</span></div>
            <div><span className="text-slate-500">Max:</span> <span className="text-slate-200">{stats.max.toFixed(4)}</span></div>
          </div>
        </div>
      )}
    </div>
  )
}