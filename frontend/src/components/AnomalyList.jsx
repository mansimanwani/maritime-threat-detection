const BADGES = {
  true_positive: { text: "confirmed", className: "badge confirmed" },
  false_positive: { text: "false alarm", className: "badge dismissed" },
};

export default function AnomalyList({ anomalies, selected, onSelect }) {
  if (anomalies.length === 0) {
    return <p className="muted" style={{ padding: "0.5rem" }}>No anomalies flagged.</p>;
  }

  return (
    <ul className="anomaly-list">
      {anomalies.map((a, i) => {
        const badge = BADGES[a.feedback_label];
        const isSelected = selected && selected.id === a.id;
        return (
          <li
            key={a.id}
            className={isSelected ? "anomaly-item selected" : "anomaly-item"}
            onClick={() => onSelect(a)}
          >
            <div className="anomaly-head">
              <span className="rank">#{i + 1}</span>
              <span className="mmsi">{a.mmsi}</span>
              <span className="score">{a.score.toFixed(2)}</span>
            </div>
            <div className="muted">
              window {a.window_id} &middot; {a.window_start.slice(11, 16)} UTC
              {badge && <span className={badge.className}>{badge.text}</span>}
            </div>
          </li>
        );
      })}
    </ul>
  );
}