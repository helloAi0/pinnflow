import React, { Component, ErrorInfo, ReactNode } from 'react'
import { AlertOctagon, RefreshCw, Terminal } from 'lucide-react'

interface Props {
  children: ReactNode
  fallbackTitle?: string
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[ErrorBoundary caught error]:', error, errorInfo)
    this.setState({ error, errorInfo })
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null })
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center p-8 rounded-2xl border border-red-500/40 bg-slate-900/90 backdrop-blur-xl text-center space-y-4 shadow-2xl glass-panel my-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-red-950/80 border border-red-500/50 text-red-400">
            <AlertOctagon className="h-8 w-8 animate-pulse" />
          </div>

          <div className="space-y-1.5 max-w-md">
            <h3 className="text-base font-bold text-white font-mono">
              {this.props.fallbackTitle || 'Component Execution Error'}
            </h3>
            <p className="text-xs text-slate-400 font-sans leading-relaxed">
              An unexpected render anomaly occurred in this scientific view module. The rest of the platform remains active.
            </p>
          </div>

          {this.state.error && (
            <div className="w-full max-w-lg rounded-lg bg-slate-950/90 border border-slate-800 p-3 text-left font-mono text-[11px] text-red-300 overflow-x-auto">
              <div className="flex items-center space-x-1.5 text-slate-500 pb-1 mb-1 border-b border-slate-800">
                <Terminal className="h-3 w-3" />
                <span>Error Log</span>
              </div>
              <p className="whitespace-pre-wrap">{this.state.error.message}</p>
            </div>
          )}

          <div className="flex items-center space-x-3 pt-2">
            <button
              onClick={this.handleReset}
              className="flex items-center space-x-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 px-4 py-2 text-xs font-mono font-bold text-white transition shadow-lg shadow-cyan-900/40"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span>Retry Component</span>
            </button>
            <button
              onClick={() => window.location.reload()}
              className="rounded-xl border border-slate-700 bg-slate-950 hover:bg-slate-800 px-4 py-2 text-xs font-mono text-slate-300 transition"
            >
              Reload Platform
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
