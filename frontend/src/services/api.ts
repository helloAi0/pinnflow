import type { FieldRequest, FieldResponse, HealthResponse, MetadataResponse, PointRequest, PointResponse } from '../types/api'

// Dynamically resolve API URL with smart production fallback
export const getApiUrl = (): string => {
  const envUrl = import.meta.env.VITE_API_URL
  if (envUrl && typeof envUrl === 'string' && envUrl.trim() !== '') {
    return envUrl.replace(/\/+$/, '')
  }

  // When running on non-localhost (e.g. Vercel deployment), point directly to Render production backend
  if (typeof window !== 'undefined' && window.location) {
    const hostname = window.location.hostname
    if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
      return 'https://pinnflow.onrender.com'
    }
  }

  // Default fallback for local development
  return 'http://localhost:8000'
}

/**
 * Format FastAPI 422 validation detail arrays or generic errors into human-readable text.
 */
export async function formatResponseError(res: Response, defaultMsg = 'Request failed'): Promise<string> {
  if (res.status === 502) {
    return 'Backend memory limit exceeded (512MB RAM OOM / HTTP 502 Bad Gateway). Reduce grid resolution (nx / ny) or disable vorticity.'
  }
  if (res.status === 504) {
    return 'Backend request timed out (HTTP 504 Gateway Timeout). Render service may still be waking up. Please retry in a few moments.'
  }

  try {
    const data = await res.json()
    if (data && data.detail) {
      if (Array.isArray(data.detail)) {
        // FastAPI 422 validation errors array
        return data.detail
          .map((item: { loc?: (string | number)[]; msg?: string }) => {
            let msg = item.msg || 'Invalid input'
            if (msg.startsWith('Value error, ')) {
              msg = msg.slice('Value error, '.length)
            }
            const field = item.loc ? item.loc.filter((part) => part !== 'body').join('.') : ''
            const prefix = field ? `${field}: ` : ''
            return `${prefix}${msg}`
          })
          .join('; ')
      } else if (typeof data.detail === 'string') {
        return data.detail
      }
    }
  } catch {
    // Non-JSON error body fallback
  }

  return `${defaultMsg} (HTTP ${res.status})`
}

/**
 * Robust fetch wrapper with exponential backoff on network errors, 502, 503, 504,
 * timeout management, and full AbortSignal pass-through support.
 * Never retries 4xx client errors (especially 422).
 */
export async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  maxRetries = 2,
  timeoutMs = 60000
): Promise<Response> {
  let lastError: unknown = null

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    // If external caller signal was already aborted, throw immediately
    if (options.signal?.aborted) {
      throw new DOMException('Request aborted by caller', 'AbortError')
    }

    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)

    // Bridge caller signal to our controller
    const onCallerAbort = () => controller.abort()
    if (options.signal) {
      options.signal.addEventListener('abort', onCallerAbort)
    }

    try {
      const res = await fetch(url, {
        ...options,
        credentials: 'omit',
        signal: controller.signal
      })
      clearTimeout(timer)
      if (options.signal) {
        options.signal.removeEventListener('abort', onCallerAbort)
      }

      // 4xx errors should NEVER be retried (client errors, 422 validation, etc.)
      if (res.ok || (res.status >= 400 && res.status < 500 && res.status !== 408)) {
        return res
      }

      // Retry on transient server errors (502, 503, 504)
      if (attempt < maxRetries && (res.status === 502 || res.status === 503 || res.status === 504)) {
        const delay = 1500 * Math.pow(1.5, attempt)
        await new Promise((r) => setTimeout(r, delay))
        continue
      }

      return res
    } catch (err: unknown) {
      clearTimeout(timer)
      if (options.signal) {
        options.signal.removeEventListener('abort', onCallerAbort)
      }

      // If aborted by caller, rethrow immediately without retrying
      if (options.signal?.aborted) {
        throw err
      }

      lastError = err

      if (attempt < maxRetries) {
        const delay = 1500 * Math.pow(1.5, attempt)
        await new Promise((r) => setTimeout(r, delay))
      }
    }
  }

  if (lastError instanceof Error && lastError.name === 'AbortError') {
    throw new Error(
      'Request timed out (60s limit). Free-tier Render server may still be spinning up. Please try again in 10-20 seconds.'
    )
  }

  throw lastError || new Error('Network connection failed.')
}

// Backward compatibility alias
export const fetchWithTimeoutAndRetry = fetchWithRetry

/**
 * Polls GET /api/v1/wakeup every 2s for up to ~100s until instance responds.
 */
export async function wakeBackend(
  onProgress?: (progress: number) => void,
  totalTimeoutMs = 100000,
  intervalMs = 2000
): Promise<boolean> {
  const startTime = Date.now()
  const apiUrl = getApiUrl()

  while (Date.now() - startTime < totalTimeoutMs) {
    const elapsed = Date.now() - startTime
    const progress = Math.min(98, Math.max(5, Math.round((elapsed / totalTimeoutMs) * 100)))
    if (onProgress) onProgress(progress)

    try {
      const res = await fetch(`${apiUrl}/api/v1/wakeup`, {
        method: 'GET',
        credentials: 'omit',
        signal: AbortSignal.timeout(12000)
      })
      if (res.ok) {
        if (onProgress) onProgress(100)
        return true
      }
    } catch {
      // Continue polling until deadline
    }

    await new Promise((r) => setTimeout(r, intervalMs))
  }

  return false
}

/**
 * Lightweight ping to check if backend is already awake.
 */
export async function pingWakeup(): Promise<boolean> {
  try {
    const apiUrl = getApiUrl()
    const res = await fetch(`${apiUrl}/api/v1/wakeup`, {
      method: 'GET',
      credentials: 'omit',
      signal: AbortSignal.timeout(4000)
    })
    return res.ok
  } catch {
    return false
  }
}

/**
 * Health check with exponential backoff.
 */
export async function checkHealth(maxRetries = 2, initialDelayMs = 1500): Promise<HealthResponse> {
  const apiUrl = getApiUrl()
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetchWithRetry(`${apiUrl}/api/v1/health`, { method: 'GET' }, 0, 15000)
      if (res.ok) {
        return await res.json()
      }
    } catch {
      if (attempt < maxRetries) {
        const delay = initialDelayMs * Math.pow(1.5, attempt)
        await new Promise((resolve) => setTimeout(resolve, delay))
      }
    }
  }

  return {
    status: 'offline',
    trained: false,
    checkpoint_name: null,
    checkpoint_sha: null,
    model_version: '2.0.0',
    training_seed: null,
    git_commit: 'unknown',
    architecture: null,
    reynolds_number: 100.0,
    reference_dataset_available: false
  }
}

export async function fetchMetadata(): Promise<MetadataResponse | null> {
  try {
    const apiUrl = getApiUrl()
    const res = await fetchWithRetry(`${apiUrl}/api/v1/metadata`, { method: 'GET' }, 1, 15000)
    if (!res.ok) return null
    return await res.json()
  } catch {
    return null
  }
}

export async function fetchFieldPrediction(
  payload: FieldRequest,
  signal?: AbortSignal
): Promise<{ data: FieldResponse; latencyMs: number }> {
  const start = performance.now()
  const apiUrl = getApiUrl()
  const res = await fetchWithRetry(
    `${apiUrl}/api/v1/predict/field`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal
    },
    2,
    60000
  )

  if (!res.ok) {
    const errorText = await formatResponseError(res, 'Field Prediction Failed')
    throw new Error(errorText)
  }

  const data: FieldResponse = await res.json()
  const latencyMs = performance.now() - start
  return { data, latencyMs }
}

export async function fetchReferenceField(
  payload: FieldRequest,
  signal?: AbortSignal
): Promise<FieldResponse> {
  const apiUrl = getApiUrl()
  const res = await fetchWithRetry(
    `${apiUrl}/api/v1/reference/field`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal
    },
    2,
    60000
  )

  if (!res.ok) {
    const errorText = await formatResponseError(res, 'Reference Dataset Unavailable')
    throw new Error(errorText)
  }

  return await res.json()
}

export async function fetchPointPrediction(
  payload: PointRequest,
  signal?: AbortSignal
): Promise<PointResponse> {
  const apiUrl = getApiUrl()
  const res = await fetchWithRetry(
    `${apiUrl}/api/v1/predict/point`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal
    },
    1,
    20000
  )

  if (!res.ok) {
    const errorText = await formatResponseError(res, 'Point Prediction Error')
    throw new Error(errorText)
  }

  return await res.json()
}
