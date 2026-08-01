const LABELS = {
  mean_speed: "average speed",
  speed_change: "speed change",
  course_change: "course change",
  gap_minutes: "AIS silence",
  neighbor_count: "nearby vessels",
};

export default function ExplanationPanel({ anomaly, onFeedback, busy }) {
  if (!anomaly) {
    return (
      <div className="explain">
        <p className="muted">Select a flagged vessel to see why it was flagged.</p>
      </div>
    );
  }

  const drivers = Object.entries(anomaly.drivers).sort((a, b) => b[1] - a[1]);

  return (
    <div className="explain">
      <h3>
        Vessel {anomaly.mmsi} <span className="muted">window {anomaly.window_id}</span>
      </h3>

      <p className="reason">{anomaly.reason}</p>

      <div className="drivers">
        {drivers.map(([feature, share]) => (
          <div className="driver" key={feature}>
            <span className="driver-name">{LABELS[feature] ?? feature}</span>
            <span className="driver-track">
              <span className="driver-fill" style={{ width: `${(share * 100).toFixed(1)}%` }} />
            </span>
            <span className="driver-pct">{(share * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>

      {anomaly.feedback_label ? (
        <p className="muted">
          Recorded as{" "}
          <strong>{anomaly.feedback_label === "true_positive" ? "a real threat" : "a false alarm"}</strong>.
          You can change it below.
        </p>
      ) : (
        <p className="muted">Is this a real threat?</p>
      )}

      <div className="actions">
        <button disabled={busy} onClick={() => onFeedback(anomaly, "true_positive")}>
          Confirm threat
        </button>
        <button disabled={busy} onClick={() => onFeedback(anomaly, "false_positive")}>
          False alarm
        </button>
      </div>
    </div>
  );
}