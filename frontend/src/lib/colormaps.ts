export type ColormapType = 'turbo' | 'viridis' | 'inferno' | 'coolwarm' | 'plasma'

// Helper to generate 256 RGB points for standard scientific palettes
function generateTurboLUT(): number[][] {
  const lut: number[][] = []
  for (let i = 0; i < 256; i++) {
    const x = i / 255
    const r = Math.floor(255 * Math.max(0, Math.min(1, 0.1357 + x * (4.5155 + x * (-8.5638 + x * 4.9086)))))
    const g = Math.floor(255 * Math.max(0, Math.min(1, 0.0914 + x * (2.1942 + x * (4.8429 + x * (-14.185 + x * 7.645))))))
    const b = Math.floor(255 * Math.max(0, Math.min(1, 0.1067 + x * (12.573 + x * (-86.801 + x * (268.86 + x * (-338.63 + x * 153.28)))))))
    lut.push([r, g, b])
  }
  return lut
}

function generateViridisLUT(): number[][] {
  const lut: number[][] = []
  for (let i = 0; i < 256; i++) {
    const x = i / 255
    const r = Math.floor(255 * Math.max(0, Math.min(1, 0.267 + x * (0.005 + x * (0.329 + x * (1.17 - x * 0.77))))))
    const g = Math.floor(255 * Math.max(0, Math.min(1, 0.004 + x * (1.405 - x * (0.835 + x * (0.505 - x * 0.9))))))
    const b = Math.floor(255 * Math.max(0, Math.min(1, 0.329 + x * (1.107 - x * (2.813 + x * (2.45 - x * 0.73))))))
    lut.push([r, g, b])
  }
  return lut
}

function generateInfernoLUT(): number[][] {
  const lut: number[][] = []
  for (let i = 0; i < 256; i++) {
    const x = i / 255
    const r = Math.floor(255 * Math.max(0, Math.min(1, Math.pow(x, 0.7))))
    const g = Math.floor(255 * Math.max(0, Math.min(1, Math.pow(x, 1.8) * 0.9)))
    const b = Math.floor(255 * Math.max(0, Math.min(1, Math.sin(x * Math.PI) * 0.5 + 0.15 * (1 - x))))
    lut.push([r, g, b])
  }
  return lut
}

function generateCoolwarmLUT(): number[][] {
  const lut: number[][] = []
  for (let i = 0; i < 256; i++) {
    const x = i / 255
    let r = 0, g = 0, b = 0
    if (x < 0.5) {
      const t = x * 2
      r = Math.floor(59 + t * (220 - 59))
      g = Math.floor(76 + t * (220 - 76))
      b = Math.floor(192 + t * (220 - 192))
    } else {
      const t = (x - 0.5) * 2
      r = Math.floor(220 + t * (180 - 220))
      g = Math.floor(220 + t * (4 - 220))
      b = Math.floor(220 + t * (39 - 220))
    }
    lut.push([Math.max(0, Math.min(255, r)), Math.max(0, Math.min(255, g)), Math.max(0, Math.min(255, b))])
  }
  return lut
}

function generatePlasmaLUT(): number[][] {
  const lut: number[][] = []
  for (let i = 0; i < 256; i++) {
    const x = i / 255
    const r = Math.floor(255 * Math.max(0, Math.min(1, 0.058 + x * (2.55 - x * (1.92 - x * 0.32)))))
    const g = Math.floor(255 * Math.max(0, Math.min(1, 0.015 + x * (0.28 + x * (2.28 - x * 1.57)))))
    const b = Math.floor(255 * Math.max(0, Math.min(1, 0.53 + x * (0.88 - x * (3.14 - x * 1.73)))))
    lut.push([r, g, b])
  }
  return lut
}

export const COLORMAP_LUTS: Record<ColormapType, number[][]> = {
  turbo: generateTurboLUT(),
  viridis: generateViridisLUT(),
  inferno: generateInfernoLUT(),
  coolwarm: generateCoolwarmLUT(),
  plasma: generatePlasmaLUT()
}

/**
 * Samples RGB color from a 256-entry precomputed lookup table.
 */
export function sampleColormapLUT(
  val: number,
  min: number,
  max: number,
  palette: ColormapType = 'turbo'
): [number, number, number] {
  if (isNaN(val) || !isFinite(val) || isNaN(min) || isNaN(max)) {
    return [15, 23, 42] // Slate dark background
  }

  const range = max - min
  const norm = range > 1e-12 ? Math.max(0, Math.min(1, (val - min) / range)) : 0.5
  const idx = Math.min(255, Math.max(0, Math.round(norm * 255)))

  const lut = COLORMAP_LUTS[palette] || COLORMAP_LUTS.turbo
  const color = lut[idx]
  return [color[0], color[1], color[2]]
}
