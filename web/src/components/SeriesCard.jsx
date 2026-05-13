import { TEAM_COLORS } from '../teamColors'

export default function SeriesCard({ series, onPredict, predicting, round }) {
  const winsA = series.wins_a ?? 0
  const winsB = series.wins_b ?? 0
  const completed = series.completed ?? false
  const knownA = series.id_a != null
  const knownB = series.id_b != null
  const hasBothTeams = knownA && knownB
  const active = !completed && hasBothTeams
  const winnerA = completed && winsA > winsB
  const winnerB = completed && winsB > winsA

  const colorA = TEAM_COLORS[series.abbr_a] ?? '#334155'
  const colorB = TEAM_COLORS[series.abbr_b] ?? '#334155'

  const nextGame = winsA + winsB + 1

  let statusText = ''
  if (!hasBothTeams) {
    statusText = 'Awaiting opponent'
  } else if (completed) {
    statusText = `${Math.max(winsA, winsB)}–${Math.min(winsA, winsB)}`
  } else if (winsA === 0 && winsB === 0) {
    statusText = 'Not started'
  } else {
    const leader = winsA > winsB ? series.name_a : winsB > winsA ? series.name_b : null
    statusText = leader
      ? `${leader} leads ${Math.max(winsA,winsB)}–${Math.min(winsA,winsB)}`
      : `Tied ${winsA}–${winsB}`
  }

  const cardClass = [
    'series-card',
    active    ? 'active'      : '',
    completed ? 'complete'    : '',
    !knownA && !knownB ? 'tbd' : (!hasBothTeams ? 'partial-tbd' : ''),
  ].filter(Boolean).join(' ')

  return (
    <div className={cardClass}>
      <TeamRow
        seed={series.seed_a} name={series.name_a} abbr={series.abbr_a}
        wins={winsA} color={colorA} isWinner={winnerA} isLoser={winnerB}
        known={knownA}
      />
      <TeamRow
        seed={series.seed_b} name={series.name_b} abbr={series.abbr_b}
        wins={winsB} color={colorB} isWinner={winnerB} isLoser={winnerA}
        known={knownB}
      />
      <div className="series-footer">
        <span className={`series-status${active ? ' live' : ''}`}>
          {statusText}
        </span>
        {active && (
          <button
            className="predict-btn"
            onClick={() => onPredict(series)}
            disabled={predicting}
          >
            Predict G{nextGame}
          </button>
        )}
      </div>
    </div>
  )
}

function TeamRow({ seed, name, abbr, wins, color, isWinner, isLoser, known }) {
  return (
    <div className={`team-row${isWinner ? ' winner' : isLoser ? ' loser' : ''}${!known ? ' unknown' : ''}`}>
      <div className="team-color-bar" style={{ background: known ? color : '#2A3550' }} />
      <span className="team-seed">{seed ?? '—'}</span>
      <span className="team-abbr" style={{ color: known ? color : '#48577A' }}>{abbr}</span>
      <span className="team-name">{name}</span>
      <WinDots wins={wins} isWinner={isWinner} />
    </div>
  )
}

function WinDots({ wins, isWinner }) {
  return (
    <div className="win-dots">
      {Array.from({ length: 4 }, (_, i) => (
        <div
          key={i}
          className={[
            'win-dot',
            i < wins ? 'filled' : '',
            i < wins && isWinner ? 'winner-dot' : '',
          ].filter(Boolean).join(' ')}
        />
      ))}
    </div>
  )
}
