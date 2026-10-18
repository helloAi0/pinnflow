import React, { useRef, useEffect, useState, useCallback } from 'react'
import { Crosshair, Camera, AlertCircle, RefreshCw, Cpu, Layers } from 'lucide-react'
import type { FieldResponse, FieldVariable, VisualizationMode } from '../types/api'
import { ColormapType, sampleColormapLUT, getColorbarGradient } from '../lib/colormaps'

interface FieldVisualizationProps {
  data: FieldResponse | null
  refData: FieldResponse | null
  activeVar: FieldVariable
  setActiveVar: (v: FieldVariable) => void
  visMode: VisualizationMode
  setVisMode: (m: VisualizationMode) => void
  isLoading: boolean
  compareMode: boolean
  onComputeVorticity?: () => void
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
  isLoading,
  compareMode: _compareMode,
  onComputeVorticity
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

  // Determine which grid of data to render based on active mode
  const getActiveGrid = useCallback((): { grid: number[][]; hasNaN: boolean; isUniform: boolean } | null => {
    if (!data) return null

    let target: number[][] = []
    if (visMode === 'prediction') {
      if (activeVar === 'u') target = data.u
      else if (activeVar === 'v') target = data.v
      else if (activeVar === 'p') target = data.p
      else if (activeVar === 'vorticity') target = data.vorticity ?? []
      else target = data.velocity_magnitude
    } else if (visMode === 'reference') {
      if (!refData) return null
      if (activeVar === 'u') target = refData.u
      else if (activeVar === 'v') target = refData.v
      else if (activeVar === 'p') target = refData.p
      else if (activeVar === 'vorticity') target = refData.vorticity ?? []
      else target = refData.velocity_magnitude
    } else if (visMode === 'absolute_error') {
      if (!refData) return null
      const pGrid = activeVar === 'u' ? data.u : (activeVar === 'v' ? data.v : (activeVar === 'p' ? data.p : (activeVar === 'vorticity' ? data.vorticity : data.velocity_magnitude)))
      const rGrid = activeVar === 'u' ? refData.u : (activeVar === 'v' ? refData.v : (activeVar === 'p' ? refData.p : (activeVar === 'vorticity' ? refData.vorticity : refData.velocity_magnitude)))
      if (!pGrid || !rGrid) return null

      target = pGrid.map((row, r) =>
        row.map((val, c) => Math.abs(val - (rGrid[r]?.[c] ?? val)))
      )
    } else {
      // PDE Residual
      target = data.u.map((row) => row.map(() => data.metrics.pde_loss))
    }

    if (!target || target.length === 0 || !target[0] || target[0].length === 0) {
      return null
    }

    let min = Infinity
    let max = -Infinity
    let hasNaN = false
    for (let r = 0; r < target.length; r++) {
      for (let c = 0; c < target[0].length; c++) {
        const val = target[r][c]
        if (isNaN(val) || !isFinite(val)) {
          hasNaN = true
        } else {
          if (val < min) min = val
          if (val > max) max = val
        }
      }
    }

    const isUniform = !hasNaN && Math.abs(max - min) < 1e-12
    return { grid: target, hasNaN, isUniform }
  }, [data, refData, activeVar, visMode])

  // Canvas rendering with HiDPI canvas sizing + bilinear interpolation + LUT sampling
  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const activeInfo = getActiveGrid()
    if (!activeInfo || activeInfo.grid.length === 0) return

    const { grid } = activeInfo
    const ny = grid.length
    const nx = grid[0].length
    if (ny < 2 || nx < 2) return

    // Sized to container x devicePixelRatio
    const rect = container.getBoundingClientRect()
    const dpr = Math.max(1, window.devicePixelRatio || 1)
    const targetWidth = Math.round(rect.width * dpr)
    const targetHeight = Math.round(rect.height * dpr)

    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth
      canvas.height = targetHeight
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // Compute min / max / mean with 1-99 percentile clipping for vorticity and absolute error
    const validVals: number[] = []
    let sum = 0
    let absMin = Infinity
    let absMax = -Infinity

    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const v = grid[r][c]
        if (!isNaN(v) && isFinite(v)) {
          validVals.push(v)
          sum += v
          if (v < absMin) absMin = v
          if (v > absMax) absMax = v
        }
      }
    }

    let min = isFinite(absMin) ? absMin : 0
    let max = isFinite(absMax) ? absMax : 1
    const count = validVals.length
    const mean = count > 0 ? sum / count : 0

    // Percentile 1-99 range clipping to prevent outliers from washing out dynamic range
    if (count > 0 && (activeVar === 'vorticity' || visMode === 'absolute_error')) {
      validVals.sort((a, b) => a - b)
      const p1 = validVals[Math.floor(count * 0.01)]
      const p99 = validVals[Math.min(count - 1, Math.floor(count * 0.99))]
      if (p99 > p1) {
        if (activeVar === 'vorticity') {
          const maxAbs = Math.max(Math.abs(p1), Math.abs(p99))
          min = -maxAbs
          max = maxAbs
        } else {
          min = p1
          max = p99
        }
      }
    }

    setStats({
      min,
      max,
      mean
    })

    const imgData = ctx.createImageData(width, height)
    const dataBuf = imgData.data

    // High-performance bilinear interpolation across grid cells
    for (let py = 0; py < height; py++) {
      // Invert Y so bottom is y_min and top is y_max
      const gyNorm = ((height - 1 - py) / (height - 1)) * (ny - 1)
      const gy0 = Math.floor(gyNorm)
      const gy1 = Math.min(gy0 + 1, ny - 1)
      const fy = gyNorm - gy0

      for (let px = 0; px < width; px++) {
        const gxNorm = (px / (width - 1)) * (nx - 1)
        const gx0 = Math.floor(gxNorm)
        const gx1 = Math.min(gx0 + 1, nx - 1)
        const fx = gxNorm - gx0

        // Bilinear interpolation
        const v00 = grid[gy0]?.[gx0] ?? 0
        const v10 = grid[gy0]?.[gx1] ?? 0
        const v01 = grid[gy1]?.[gx0] ?? 0
        const v11 = grid[gy1]?.[gx1] ?? 0

        const interpVal = (1 - fx) * (1 - fy) * v00 +
                          fx * (1 - fy) * v10 +
                          (1 - fx) * fy * v01 +
                          fx * fy * v11

        const [r, g, b] = sampleColormapLUT(interpVal, min, max, colormap)
        const idx = (py * width + px) * 4
        dataBuf[idx] = r
        dataBuf[idx + 1] = g
        dataBuf[idx + 2] = b
        dataBuf[idx + 3] = 255
      }
    }

    ctx.putImageData(imgData, 0, 0)

    // Optional Grid Lines Overlay
    if (showGridLines) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)'
      ctx.lineWidth = 1 * dpr
      const stepX = width / 12
      const stepY = height / 6
      for (let x = 0; x <= width; x += stepX) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, height)
        ctx.stroke()
      }
      for (let y = 0; y <= height; y += stepY) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(width, y)
        ctx.stroke()
      }
    }

    // Optional Streamlines Overlay: Physical units numerical integration
    if (showStreamlines && data && data.x && data.y && data.u && data.v && data.x[0] && data.y[0]) {
      ctx.lineWidth = 1.3 * dpr
      const numLines = 14
      const xMin = data.x[0][0]
      const xMax = data.x[0][nx - 1]
      const yMin = data.y[0][0]
      const yMax = data.y[ny - 1][0]
      const spanX = xMax - xMin || 1.0
      const spanY = yMax - yMin || 1.0

      for (let i = 0; i < numLines; i++) {
        let curPhysX = xMin + 0.02 * spanX
        let curPhysY = yMin + ((i + 0.5) / numLines) * spanY

        ctx.beginPath()
        ctx.strokeStyle = `rgba(255, 255, 255, ${0.22 + (i % 3) * 0.07})`

        const startPx = ((curPhysX - xMin) / spanX) * width
        const startPy = height - ((curPhysY - yMin) / spanY) * height
        ctx.moveTo(startPx, startPy)

        const dt = 0.05 * (spanX / nx)
        let isValidStream = true

        for (let step = 0; step < 90; step++) {
          const gxNorm = ((curPhysX - xMin) / spanX) * (nx - 1)
          const gyNorm = ((curPhysY - yMin) / spanY) * (ny - 1)

          if (gxNorm < 0 || gxNorm >= nx - 1 || gyNorm < 0 || gyNorm >= ny - 1) break

          const gx0 = Math.floor(gxNorm)
          const gx1 = Math.min(gx0 + 1, nx - 1)
          const fx = gxNorm - gx0

          const gy0 = Math.floor(gyNorm)
          const gy1 = Math.min(gy0 + 1, ny - 1)
          const fy = gyNorm - gy0

          const u00 = data.u[gy0]?.[gx0]
          const u10 = data.u[gy0]?.[gx1]
          const u01 = data.u[gy1]?.[gx0]
          const u11 = data.u[gy1]?.[gx1]

          const v00 = data.v[gy0]?.[gx0]
          const v10 = data.v[gy0]?.[gx1]
          const v01 = data.v[gy1]?.[gx0]
          const v11 = data.v[gy1]?.[gx1]

          if (
            u00 === undefined || u10 === undefined || u01 === undefined || u11 === undefined ||
            v00 === undefined || v10 === undefined || v01 === undefined || v11 === undefined ||
            isNaN(u00) || isNaN(u10) || isNaN(u01) || isNaN(u11) ||
            isNaN(v00) || isNaN(v10) || isNaN(v01) || isNaN(v11) ||
            !isFinite(u00) || !isFinite(u10) || !isFinite(u01) || !isFinite(u11) ||
            !isFinite(v00) || !isFinite(v10) || !isFinite(v01) || !isFinite(v11)
          ) {
            isValidStream = false
            break
          }

          const uVal = (1 - fx) * (1 - fy) * u00 + fx * (1 - fy) * u10 + (1 - fx) * fy * u01 + fx * fy * u11
          const vVal = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 + (1 - fx) * fy * v01 + fx * fy * v11

          curPhysX += uVal * dt
          curPhysY += vVal * dt

          const px = ((curPhysX - xMin) / spanX) * width
          const py = height - ((curPhysY - yMin) / spanY) * height
          ctx.lineTo(px, py)
        }

        if (isValidStream) {
          ctx.stroke()
        }
      }
    }

    // Optional Velocity Quiver Vectors
    if (showVectors && data) {
      const sampleStrideX = Math.max(1, Math.floor(nx / 24))
      const sampleStrideY = Math.max(1, Math.floor(ny / 12))
      const maxLen = 14 * dpr

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)'
      ctx.fillStyle = '#06b6d4'
      ctx.lineWidth = 1.3 * dpr

      for (let gy = 0; gy < ny; gy += sampleStrideY) {
        for (let gx = 0; gx < nx; gx += sampleStrideX) {
          const uVal = data.u[gy][gx]
          const vVal = data.v[gy][gx]
          const mag = Math.sqrt(uVal * uVal + vVal * vVal) || 0.001

          const px = (gx / (nx - 1)) * width
          const py = height - (gy / (ny - 1)) * height

          const dx = (uVal / mag) * Math.min(maxLen, mag * 8 * dpr)
          const dy = -(vVal / mag) * Math.min(maxLen, mag * 8 * dpr)

          ctx.beginPath()
          ctx.moveTo(px, py)
          ctx.lineTo(px + dx, py + dy)
          ctx.stroke()

          ctx.beginPath()
          ctx.arc(px + dx, py + dy, 2.0 * dpr, 0, 2 * Math.PI)
          ctx.fill()
        }
      }
    }

    // Solid Cylinder Obstacle Boundary Overlay (R=0.5 at x=0, y=0)
    if (data && data.x && data.y && data.x[0] && data.y[0]) {
      const xMin = data.x[0][0]
      const xMax = data.x[0][nx - 1]
      const yMin = data.y[0][0]
      const yMax = data.y[ny - 1][0]

      const cylX = 0.0
      const cylY = 0.0
      const cylR = 0.5

      if (xMin <= cylX + cylR && xMax >= cylX - cylR && yMin <= cylY + cylR && yMax >= cylY - cylR) {
        const cylPx = ((cylX - xMin) / (xMax - xMin)) * width
        const cylPy = height - ((cylY - yMin) / (yMax - yMin)) * height
        const cylRpx = (cylR / (xMax - xMin)) * width

        ctx.beginPath()
        ctx.arc(cylPx, cylPy, cylRpx, 0, 2 * Math.PI)
        ctx.fillStyle = '#020617'
        ctx.fill()
        ctx.lineWidth = 2 * dpr
        ctx.strokeStyle = '#38bdf8'
        ctx.stroke()
      }
    }
  }, [getActiveGrid, colormap, showGridLines, showStreamlines, showVectors, data, activeVar, visMode])

  // Trigger render on changes or resize
  useEffect(() => {
    renderCanvas()
  }, [renderCanvas])

  useEffect(() => {
    const handleResize = () => {
      renderCanvas()
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [renderCanvas])

  // Coordinate probe on canvas mouse move
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data || !canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const relX = (e.clientX - rect.left) / rect.width
    const relY = 1.0 - (e.clientY - rect.top) / rect.height

    const nx = data.u[0].length
    const ny = data.u.length

    const gx = Math.min(nx - 1, Math.max(0, Math.floor(relX * nx)))
    const gy = Math.min(ny - 1, Math.max(0, Math.floor(relY * ny)))

    const xCoord = data.x[gy]?.[gx] ?? 0
    const yCoord = data.y[gy]?.[gx] ?? 0
    const uVal = data.u[gy]?.[gx] ?? 0
    const vVal = data.v[gy]?.[gx] ?? 0
    const pVal = data.p[gy]?.[gx] ?? 0
    const magVal = data.velocity_magnitude[gy]?.[gx] ?? Math.sqrt(uVal * uVal + vVal * vVal)
    const vortVal = data.vorticity ? data.vorticity[gy]?.[gx] : undefined

    setProbe({
      x: xCoord,
      y: yCoord,
      u: uVal,
      v: vVal,
      p: pVal,
      mag: magVal,
      vorticity: vortVal,
      px: e.clientX - rect.left,
      py: e.clientY - rect.top
    })
  }

  const handleDownloadSnapshot = () => {
    if (!canvasRef.current) return
    const link = document.createElement('a')
    link.download = `pinnflow_${activeVar}_t${data?.metrics.time_snapshot ?? 0}.png`
    link.href = canvasRef.current.toDataURL('image/png')
    link.click()
  }

  const activeInfo = getActiveGrid()
  const isVorticityMissing = activeVar === 'vorticity' && (!data?.vorticity || data.vorticity.length === 0)
  const isReferenceMissing = (visMode === 'reference' || visMode === 'absolute_error') && !refData
  const isDegenerateField = activeInfo ? (activeInfo.hasNaN || activeInfo.isUniform) : false

  return (
    <div className="flex flex-col space-y-4 rounded-xl border border-slate-800/90 bg-slate-900/70 p-5 backdrop-blur-xl shadow-2xl glass-panel relative">
      {/* Top Header: Variable Toggles & Colormap controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs font-bold text-slate-300 font-mono mr-1">Variable:</span>
          {(['velocity_magnitude', 'u', 'v', 'p', 'vorticity'] as FieldVariable[]).map((v) => (
            <button
              key={v}
              type="button"
              onClick={() => setActiveVar(v)}
              className={`rounded-lg px-2.5 py-1 text-xs font-mono transition ${
                activeVar === v
                  ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/30'
                  : 'bg-slate-950/80 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-slate-800'
              }`}
            >
              {v === 'velocity_magnitude' ? '|u| Magnitude' : v === 'vorticity' ? 'ω Vorticity' : v}
            </button>
          ))}
        </div>

        <div className="flex items-center space-x-2">
          {/* Colormap Selector */}
          <select
            value={colormap}
            onChange={(e) => setColormap(e.target.value as ColormapType)}
            className="rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1 text-xs text-slate-300 font-mono focus:border-cyan-500 focus:outline-none"
          >
            <option value="turbo">Turbo (Scientific)</option>
            <option value="viridis">Viridis (Perceptual)</option>
            <option value="inferno">Inferno</option>
            <option value="coolwarm">Coolwarm (Diverging)</option>
            <option value="plasma">Plasma</option>
          </select>

          {/* Snapshot capture */}
          <button
            type="button"
            onClick={handleDownloadSnapshot}
            className="p-1.5 rounded-lg border border-slate-800 bg-slate-950 text-slate-400 hover:text-white hover:bg-slate-800 transition"
            title="Download PNG Snapshot"
          >
            <Camera className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Mode Tabs (Prediction, Reference, Absolute Error, PDE Residual) */}
      <div className="flex items-center justify-between text-xs font-mono">
        <div className="flex space-x-1.5 bg-slate-950/80 p-1 rounded-xl border border-slate-800">
          <button
            type="button"
            onClick={() => setVisMode('prediction')}
            className={`px-3 py-1 rounded-lg transition ${
              visMode === 'prediction' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-white'
            }`}
          >
            PINN Prediction
          </button>
          <button
            type="button"
            onClick={() => setVisMode('reference')}
            className={`px-3 py-1 rounded-lg transition ${
              visMode === 'reference' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-white'
            }`}
          >
            DNS Reference
          </button>
          <button
            type="button"
            onClick={() => setVisMode('absolute_error')}
            className={`px-3 py-1 rounded-lg transition ${
              visMode === 'absolute_error' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-white'
            }`}
          >
            |Error| Residual
          </button>
          <button
            type="button"
            onClick={() => setVisMode('pde_residual')}
            className={`px-3 py-1 rounded-lg transition ${
              visMode === 'pde_residual' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-white'
            }`}
          >
            PDE Residual
          </button>
        </div>

        {/* Display Overlays Toggles */}
        <div className="flex items-center space-x-3 text-slate-400">
          <label className="flex items-center space-x-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showStreamlines}
              onChange={(e) => setShowStreamlines(e.target.checked)}
              className="rounded border-slate-800 bg-slate-950 text-cyan-500"
            />
            <span className="text-[11px]">Streamlines</span>
          </label>
          <label className="flex items-center space-x-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showVectors}
              onChange={(e) => setShowVectors(e.target.checked)}
              className="rounded border-slate-800 bg-slate-950 text-cyan-500"
            />
            <span className="text-[11px]">Vectors</span>
          </label>
          <label className="flex items-center space-x-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showGridLines}
              onChange={(e) => setShowGridLines(e.target.checked)}
              className="rounded border-slate-800 bg-slate-950 text-cyan-500"
            />
            <span className="text-[11px]">Mesh</span>
          </label>
        </div>
      </div>

      {/* Main Canvas Viewport Container */}
      <div
        ref={containerRef}
        className="relative w-full h-[420px] rounded-xl overflow-hidden border border-slate-800/80 bg-slate-950 shadow-inner group flex items-center justify-center"
        onMouseEnter={() => setHoverInside(true)}
        onMouseLeave={() => setHoverInside(false)}
      >
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onClick={() => setPinnedProbe((prev) => (prev ? null : probe))}
          className="w-full h-full cursor-crosshair block"
        />

        {/* OVERLAY 1: Loading Scientific Simulation (Only when no prior data exists) */}
        {!data && isLoading && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-slate-950/80 backdrop-blur-md transition-all space-y-3">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-cyan-950/80 border border-cyan-500 text-cyan-400 shadow-xl">
              <RefreshCw className="h-7 w-7 animate-spin" />
            </div>
            <div className="text-center space-y-1">
              <h4 className="text-sm font-bold text-white font-mono">Loading Scientific Simulation...</h4>
              <p className="text-xs text-slate-400 font-sans">Evaluating Navier-Stokes physical coordinates in 500-point batches</p>
            </div>
          </div>
        )}

        {/* Frame compute indicator (Keeps canvas visible without flickering) */}
        {data && isLoading && (
          <div className="absolute top-3 left-3 z-30 flex items-center space-x-2 bg-slate-950/85 border border-cyan-500/60 rounded-lg px-2.5 py-1 text-[11px] font-mono text-cyan-300 backdrop-blur-md shadow-lg pointer-events-none">
            <RefreshCw className="h-3 w-3 animate-spin text-cyan-400" />
            <span>Computing frame...</span>
          </div>
        )}

        {/* OVERLAY 2: Awaiting Compute Parameters */}
        {!data && !isLoading && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-slate-950/90 backdrop-blur-md p-6 text-center space-y-3">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-900 border border-slate-700 text-slate-400">
              <Cpu className="h-7 w-7 animate-pulse" />
            </div>
            <div className="space-y-1 max-w-sm">
              <h4 className="text-sm font-bold text-white font-mono">Awaiting Compute Parameters</h4>
              <p className="text-xs text-slate-400">
                Click &quot;Evaluate Physical Field&quot; to begin autograd neural Navier-Stokes flow reconstruction.
              </p>
            </div>
          </div>
        )}

        {/* OVERLAY 3: Missing Vorticity */}
        {data && !isLoading && isVorticityMissing && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-slate-950/85 backdrop-blur-md p-6 text-center space-y-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-amber-950/80 border border-amber-600 text-amber-400">
              <AlertCircle className="h-6 w-6" />
            </div>
            <div className="space-y-1 max-w-sm">
              <h4 className="text-sm font-bold text-white font-mono">Autograd Vorticity Field Not Computed</h4>
              <p className="text-xs text-slate-400">
                Vorticity (ω = ∂v/∂x - ∂u/∂y) requires second-order autograd graphs. Check &quot;Compute Exact Autograd Vorticity&quot; and evaluate.
              </p>
            </div>
            {onComputeVorticity && (
              <button
                type="button"
                onClick={onComputeVorticity}
                className="mt-2 px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 text-xs font-bold text-white shadow-lg hover:from-cyan-500 hover:to-blue-500 font-mono"
              >
                ⚡ Compute Vorticity Now
              </button>
            )}
          </div>
        )}

        {/* OVERLAY 4: Missing Reference DNS Ground Truth */}
        {data && !isLoading && isReferenceMissing && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-slate-950/85 backdrop-blur-md p-6 text-center space-y-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-indigo-950/80 border border-indigo-600 text-indigo-400">
              <Layers className="h-6 w-6" />
            </div>
            <div className="space-y-1 max-w-sm">
              <h4 className="text-sm font-bold text-white font-mono">Reference DNS Dataset Unavailable</h4>
              <p className="text-xs text-slate-400">
                The authentic Karniadakis cylinder wake DNS dataset (cylinder_nektar_wake.mat) is required for ground-truth comparison.
                Execute <code className="text-cyan-300 font-mono bg-slate-900 px-1 py-0.5 rounded">python scripts/download_data.py</code> to download authentic data.
              </p>
            </div>
          </div>
        )}

        {/* OVERLAY 5: NaN / Uniform Fields Notice */}
        {data && !isLoading && isDegenerateField && !isVorticityMissing && !isReferenceMissing && (
          <div className="absolute top-4 right-4 z-20 max-w-xs rounded-xl border border-amber-800/80 bg-amber-950/80 p-3 text-xs text-amber-200 backdrop-blur-md shadow-xl space-y-1">
            <div className="font-bold flex items-center space-x-1">
              <AlertCircle className="h-4 w-4 text-amber-400" />
              <span>Uniform or Degenerate Field Detected</span>
            </div>
            <p className="text-[11px] text-amber-300/90 font-mono">
              The evaluated field has uniform values across the spatial mesh (min ≈ max). Check coordinates and model calibration.
            </p>
          </div>
        )}

        {/* Floating Probe Coordinate HUD */}
        {(hoverInside || pinnedProbe) && (probe || pinnedProbe) && (
          <div className="absolute bottom-3 left-3 z-20 rounded-xl border border-slate-700/80 bg-slate-950/90 p-3 font-mono text-[11px] shadow-2xl backdrop-blur-md space-y-1 text-slate-200 pointer-events-none">
            <div className="flex items-center space-x-1.5 text-cyan-400 font-bold border-b border-slate-800 pb-1">
              <Crosshair className="h-3 w-3" />
              <span>Spatial Probe (x, y)</span>
              {pinnedProbe && <span className="text-[10px] text-amber-400 ml-1">[Pinned]</span>}
            </div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px]">
              <div>x: <span className="text-white">{(pinnedProbe || probe)!.x.toFixed(3)}</span></div>
              <div>y: <span className="text-white">{(pinnedProbe || probe)!.y.toFixed(3)}</span></div>
              <div>u: <span className="text-cyan-300 font-semibold">{(pinnedProbe || probe)!.u.toFixed(4)}</span></div>
              <div>v: <span className="text-cyan-300 font-semibold">{(pinnedProbe || probe)!.v.toFixed(4)}</span></div>
              <div>p: <span className="text-purple-300 font-semibold">{(pinnedProbe || probe)!.p.toFixed(4)}</span></div>
              <div>|u|: <span className="text-emerald-300 font-semibold">{(pinnedProbe || probe)!.mag.toFixed(4)}</span></div>
              {(pinnedProbe || probe)!.vorticity !== undefined && (
                <div className="col-span-2">ω: <span className="text-amber-300 font-semibold">{(pinnedProbe || probe)!.vorticity!.toFixed(4)}</span></div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Colormap Legend Bar & Field Extremas */}
      {stats && (
        <div className="flex flex-col space-y-1.5 pt-1">
          <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
            <span>Min: <strong className="text-slate-200">{stats.min.toFixed(4)}</strong></span>
            <span>Mean: <strong className="text-cyan-300">{stats.mean.toFixed(4)}</strong></span>
            <span>Max: <strong className="text-slate-200">{stats.max.toFixed(4)}</strong></span>
          </div>

          <div className="relative h-2.5 w-full rounded-full overflow-hidden border border-slate-800 p-0.5 bg-slate-950">
            <div
              className="h-full w-full rounded-full"
              style={{
                background: getColorbarGradient(colormap)
              }}
            />
          </div>
        </div>
      )}
    </div>
  )
}