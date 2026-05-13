import SeriesCard from './SeriesCard'

export default function ConferenceBracket({ conference, data, onPredict, predicting, flipped }) {
  if (!data) return null

  const rounds = [
    { key: 'r1', label: 'First Round', series: data.r1, round: 1 },
    { key: 'r2', label: 'Semifinals',  series: data.r2, round: 2 },
    { key: 'r3', label: 'Conf Finals', series: data.r3, round: 3 },
  ]

  const displayRounds = flipped ? [...rounds].reverse() : rounds

  return (
    <div className={`conf-bracket${flipped ? ' flipped' : ''}`}>
      <div className="conf-title">{conference}</div>
      <div className="rounds-row">
        {displayRounds.map(({ key, label, series, round }) => (
          <div key={key} className="round-col">
            <div className="round-label">{label}</div>
            <div className="round-slots">
              {series.map((s, i) => (
                <div key={i} className="series-slot">
                  <SeriesCard
                    series={s}
                    onPredict={onPredict}
                    predicting={predicting}
                    round={round}
                  />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
