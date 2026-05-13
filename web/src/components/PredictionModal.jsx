import { useState, useEffect } from 'react'

export default function PredictionModal({ prediction, loading, onClose }) {
  const [animated, setAnimated] = useState(false)

  useEffect(() => {
    const id = setTimeout(() => setAnimated(true), 40)
    return () => clearTimeout(id)
  }, [])

  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) onClose()
  }

  return (
    <div className="modal-backdrop" onClick={handleBackdropClick}>
      <div className="modal">
        <button className="modal-close" onClick={onClose}>×</button>

        {loading ? (
          <div className="modal-loading">
            <div className="spinner" />
            <span>Running model…</span>
          </div>
        ) : prediction ? (
          <>
            <div className="modal-title">Game Prediction</div>
            <div className="modal-matchup">
              {prediction.team_a_name} vs {prediction.team_b_name}
            </div>

            <ProbRow
              name={prediction.team_a_name}
              prob={prediction.prob_a}
              isWinner={prediction.predicted_winner_id === prediction.team_a_id}
              animated={animated}
            />
            <ProbRow
              name={prediction.team_b_name}
              prob={prediction.prob_b}
              isWinner={prediction.predicted_winner_id === prediction.team_b_id}
              animated={animated}
            />

            <div className="modal-verdict">
              <div className="verdict-label">Predicted Winner</div>
              <div className="verdict-name">{prediction.predicted_winner_name}</div>
              <div className="verdict-conf">
                {Math.round(Math.max(prediction.prob_a, prediction.prob_b) * 100)}% confidence
              </div>
            </div>
            <div className="modal-context">
              Game {prediction.game_num}
            </div>
          </>
        ) : null}
      </div>
    </div>
  )
}

function ProbRow({ name, prob, isWinner, animated }) {
  const pct = Math.round(prob * 100)
  const inlineLabel = pct > 22

  return (
    <div className="prob-row">
      <span className="prob-team-name">{name}</span>
      <div className="prob-bar-wrap">
        <div
          className={`prob-bar${isWinner ? ' winner-bar' : ''}`}
          style={{ width: animated ? `${pct}%` : '0%' }}
        >
          {inlineLabel && <span className="prob-pct">{pct}%</span>}
        </div>
      </div>
      {!inlineLabel && <span className="prob-pct-out">{pct}%</span>}
    </div>
  )
}
