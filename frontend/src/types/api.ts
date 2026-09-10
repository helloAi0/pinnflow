
export interface FieldRequest {
  x_min: number
  x_max: number
  y_min: number
  y_max: number
  t: number
  nx: number
  ny: number
  compute_vorticity: boolean
}

export interface FieldResponse {
  x: number[][]
  y: number[][]
  u: number[][]
  v: number[][]
  p: number[][]
  velocity_magnitude: number[][]
  vorticity?: number[][]
}

export interface PointRequest {
  x: number
  y: number
  t: number
  compute_vorticity?: boolean
}

export interface PointResponse {
  u: number
  v: number
  p: number
  velocity_magnitude: number
  vorticity?: number
}

export interface HealthResponse {
  status: string
}

export type FieldVariable = 'u' | 'v' | 'p' | 'velocity_magnitude' | 'vorticity'