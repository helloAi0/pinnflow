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

export interface FieldMetrics {
  pde_loss: number
  continuity_residual_mean: number
  momentum_x_residual_mean: number
  momentum_y_residual_mean: number
  time_snapshot: number
  reynolds_number: number
  viscosity: number
  nx: number
  ny: number
  trained_weights: boolean
}

export interface FieldResponse {
  x: number[][]
  y: number[][]
  u: number[][]
  v: number[][]
  p: number[][]
  velocity_magnitude: number[][]
  vorticity?: number[][]
  metrics: FieldMetrics
  status: string
  request_id: string
}

export interface HealthResponse {
  status: string
  trained: boolean
  checkpoint_name: string | null
  checkpoint_sha: string | null
  model_version: string
  training_seed: number | null
  git_commit: string
  architecture: string | null
  reynolds_number: number
  reference_dataset_available: boolean
}

export interface MetadataResponse {
  physics: {
    reynolds_number: number
    density: number
    kinematic_viscosity: number
    cylinder_center: [number, number]
    cylinder_radius: number
    x_min: number
    x_max: number
    y_min: number
    y_max: number
    t_min: number
    t_max: number
    pressure_gauge_coord: [number, number]
  }
  model_metadata: Record<string, unknown>
  device: string
  git_commit: string
  checkpoint_status: string
}

export type FieldVariable = 'u' | 'v' | 'p' | 'velocity_magnitude' | 'vorticity'
export type VisualizationMode = 'prediction' | 'reference' | 'absolute_error' | 'pde_residual'
export type ActiveTab = 'field' | 'error_analysis' | 'diagnostics' | 'benchmark' | 'reproducibility'

export interface BenchmarkBaseline {
  name: string
  model_type: string
  weighting: string
  rel_l2_u: { mean: number; std: number; ci95_margin: number }
  rel_l2_v: { mean: number; std: number; ci95_margin: number }
  rel_l2_vel: { mean: number; std: number; ci95_margin: number }
  divergence: { mean: number; std: number }
  latency_us: number
  param_count: number
}