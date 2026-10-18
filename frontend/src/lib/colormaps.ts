export type ColormapType = 'turbo' | 'viridis' | 'inferno' | 'coolwarm' | 'plasma'

export type RGB = [number, number, number]
export type ColorStop = [number, RGB]

export const COLORMAP_STOPS: Record<ColormapType, ColorStop[]> = {
  turbo: [
    [0.0, [48, 18, 59]],
    [0.1, [65, 69, 171]],
    [0.2, [70, 123, 244]],
    [0.3, [56, 185, 250]],
    [0.4, [44, 244, 166]],
    [0.5, [114, 254, 94]],
    [0.65, [194, 229, 55]],
    [0.8, [246, 165, 36]],
    [0.88, [247, 122, 24]],
    [0.95, [198, 46, 8]],
    [1.0, [122, 4, 3]]
  ],
  viridis: [
    [0.0, [68, 1, 84]],
    [0.15, [72, 40, 120]],
    [0.3, [62, 73, 137]],
    [0.45, [49, 104, 142]],
    [0.6, [38, 130, 142]],
    [0.75, [31, 158, 137]],
    [0.85, [53, 183, 121]],
    [0.93, [110, 206, 88]],
    [1.0, [253, 231, 37]]
  ],
  inferno: [
    [0.0, [0, 0, 4]],
    [0.15, [27, 12, 65]],
    [0.3, [74, 12, 107]],
    [0.45, [120, 28, 109]],
    [0.6, [165, 44, 96]],
    [0.72, [207, 68, 70]],
    [0.82, [237, 105, 37]],
    [0.9, [251, 155, 6]],
    [0.96, [247, 209, 60]],
    [1.0, [252, 255, 164]]
  ],
  coolwarm: [
    [0.0, [59, 76, 192]],
    [0.2, [103, 136, 238]],
    [0.35, [155, 196, 245]],
    [0.45, [201, 215, 240]],
    [0.5, [221, 221, 221]],
    [0.55, [237, 209, 194]],
    [0.65, [247, 168, 137]],
    [0.8, [226, 105, 82]],
    [1.0, [180, 4, 38]]
  ],
  plasma: [
    [0.0, [13, 8, 135]],
    [0.18, [70, 3, 159]],
    [0.32, [114, 1, 168]],
    [0.45, [156, 23, 158]],
    [0.58, [189, 55, 134]],
    [0.7, [216, 87, 107]],
    [0.8, [237, 121, 83]],
    [0.9, [251, 159, 58]],
    [0.96, [253, 202, 38]],
    [1.0, [240, 249, 33]]
  ]
}

function buildLUTFromStops(stops: ColorStop[]): RGB[] {
  const lut: RGB[] = []
  for (let i = 0; i < 256; i++) {
    const t = i / 255.0
    let s0 = stops[0]
    let s1 = stops[stops.length - 1]

    for (let j = 0; j < stops.length - 1; j++) {
      if (t >= stops[j][0] && t <= stops[j + 1][0]) {
        s0 = stops[j]
        s1 = stops[j + 1]
        break
      }
    }

    const span = s1[0] - s0[0]
    const frac = span > 0 ? (t - s0[0]) / span : 0
    const r = Math.round(s0[1][0] + frac * (s1[1][0] - s0[1][0]))
    const g = Math.round(s0[1][1] + frac * (s1[1][1] - s0[1][1]))
    const b = Math.round(s0[1][2] + frac * (s1[1][2] - s0[1][2]))

    lut.push([
      Math.max(0, Math.min(255, r)),
      Math.max(0, Math.min(255, g)),
      Math.max(0, Math.min(255, b))
    ])
  }
  return lut
}

export const COLORMAP_LUTS: Record<ColormapType, RGB[]> = {
  turbo: buildLUTFromStops(COLORMAP_STOPS.turbo),
  viridis: buildLUTFromStops(COLORMAP_STOPS.viridis),
  inferno: buildLUTFromStops(COLORMAP_STOPS.inferno),
  coolwarm: buildLUTFromStops(COLORMAP_STOPS.coolwarm),
  plasma: buildLUTFromStops(COLORMAP_STOPS.plasma)
}

/**
 * Returns a CSS linear-gradient string matching the exact stops of the 256-entry LUT.
 */
export function getColorbarGradient(palette: ColormapType): string {
  const stops = COLORMAP_STOPS[palette] || COLORMAP_STOPS.turbo
  const colorList = stops
    .map(([pos, [r, g, b]]) => `rgb(${r}, ${g}, ${b}) ${(pos * 100).toFixed(1)}%`)
    .join(', ')
  return `linear-gradient(to right, ${colorList})`
}

/**
 * Samples RGB color from the 256-entry precomputed lookup table.
 */
export function sampleColormapLUT(
  val: number,
  min: number,
  max: number,
  palette: ColormapType = 'turbo'
): [number, number, number] {
  if (isNaN(val) || !isFinite(val) || isNaN(min) || isNaN(max)) {
    return [15, 23, 42] // Slate dark background for invalid numbers
  }

  const range = max - min
  const norm = range > 1e-12 ? Math.max(0, Math.min(1, (val - min) / range)) : 0.5
  const idx = Math.min(255, Math.max(0, Math.round(norm * 255)))

  const lut = COLORMAP_LUTS[palette] || COLORMAP_LUTS.turbo
  return lut[idx]
}
