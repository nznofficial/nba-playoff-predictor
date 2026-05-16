import { useState, useEffect, useCallback } from 'react'
import ConferenceBracket from './components/ConferenceBracket'
import SeriesCard from './components/SeriesCard'
import PredictionModal from './components/PredictionModal'
import './App.css'

const API = import.meta.env.VITE_API_BASE ?? ''

export default function App() {
  const [bracket, setBracket] = useState(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [refreshError, setRefreshError] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [predicting, setPredicting] = useState(false)

  const fetchBracket = useCallback(async (forceRefresh = false) => {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 45_000)
    try {
      const url = forceRefresh ? `${API}/api/refresh` : `${API}/api/bracket`
      const method = forceRefresh ? 'POST' : 'GET'
      const res = await fetch(url, { method, signal: controller.signal })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const data = await res.json()
      setBracket(data)
      if (forceRefresh) setRefreshError(null)
    } catch (err) {
      if (err.name === 'AbortError') {
        setRefreshError('Live Scores timed out — NBA API was too slow. Try again in a moment.')
      } else if (forceRefresh) {
        setRefreshError('Refresh failed. Check that the API server is running.')
      }
      console.error('Failed to load bracket:', err)
    } finally {
      clearTimeout(timer)
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => { fetchBracket() }, [fetchBracket])

  const handleRefresh = () => {
    setRefreshing(true)
    setRefreshError(null)
    fetchBracket(true)
  }

  const handlePredict = async (series) => {
    setPredicting(true)
    setPrediction(null)
    try {
      const res = await fetch(`${API}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          team_a_id: series.id_a,
          team_b_id: series.id_b,
          seed_a: series.seed_a,
          seed_b: series.seed_b,
          wins_a: series.wins_a,
          wins_b: series.wins_b,
        }),
      })
      const data = await res.json()
      setPrediction(data)
    } catch (err) {
      console.error('Prediction failed:', err)
    } finally {
      setPredicting(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <div className="header-logo">🏀</div>
          <div>
            <h1 className="app-title">NBA Playoff Predictor</h1>
            <div className="app-subtitle">2025–26 Season · ML Model</div>
          </div>
        </div>
        <button
          className={`refresh-btn${refreshing ? ' loading' : ''}`}
          onClick={handleRefresh}
          disabled={refreshing}
        >
          <span className="refresh-icon">↻</span>
          {refreshing ? 'Refreshing…' : 'Live Scores'}
        </button>
      </header>

      {refreshError && (
        <div className="refresh-error-banner">
          {refreshError}
          <button className="refresh-error-dismiss" onClick={() => setRefreshError(null)}>✕</button>
        </div>
      )}

      <main className="bracket-root">
        {loading ? (
          <div className="state-message">
            <div className="loading-spinner" />
            Loading bracket…
          </div>
        ) : bracket ? (
          <div className="bracket-layout">
            <ConferenceBracket
              conference="Western Conference"
              data={bracket.West}
              onPredict={handlePredict}
              predicting={predicting}
            />
            <div className="finals-col">
              <div className="finals-label">
                <div className="finals-trophy">🏆</div>
                <div className="finals-title">NBA Finals</div>
              </div>
              <div className="finals-card-wrap">
                {bracket.finals && (
                  <SeriesCard
                    series={bracket.finals}
                    onPredict={handlePredict}
                    predicting={predicting}
                    round={4}
                  />
                )}
              </div>
            </div>
            <ConferenceBracket
              conference="Eastern Conference"
              data={bracket.East}
              onPredict={handlePredict}
              predicting={predicting}
              flipped
            />
          </div>
        ) : (
          <div className="state-message error">
            Failed to load bracket. Make sure the API is running:<br />
            <code>uvicorn api.server:app --reload --port 8000</code>
          </div>
        )}
      </main>

      {(prediction || predicting) && (
        <PredictionModal
          prediction={prediction}
          loading={predicting}
          onClose={() => setPrediction(null)}
        />
      )}
    </div>
  )
}
