import { useCallback, useEffect, useState } from "react";
import { getStatus, getAnomalies, getTrack, sendFeedback, recalibrate } from "./api";
import MapView from "./components/MapView";
import AnomalyList from "./components/AnomalyList";
import ExplanationPanel from "./components/ExplanationPanel";

export default function App() {
  const [status, setStatus] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [selected, setSelected] = useState(null);
  const [track, setTrack] = useState([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    const [s, a] = await Promise.all([getStatus(), getAnomalies()]);
    setStatus(s);
    setAnomalies(a);
    return a;
  }, []);

  useEffect(() => {
    refresh().catch((e) => setError(e.message));
  }, [refresh]);

  useEffect(() => {
    if (!selected) {
      setTrack([]);
      return;
    }
    getTrack(selected.mmsi)
      .then(setTrack)
      .catch(() => setTrack([]));
  }, [selected]);

  async function handleFeedback(anomaly, label) {
    setBusy(true);
    setMessage("");
    try {
      await sendFeedback(anomaly.window_id, anomaly.mmsi, label);
      const fresh = await refresh();
      setSelected(fresh.find((a) => a.id === anomaly.id) ?? null);
      setMessage(`Recorded. ${label === "true_positive" ? "Confirmed" : "Dismissed"} vessel ${anomaly.mmsi}.`);
    } catch (e) {
      setMessage(`Failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function handleRecalibrate() {
    setBusy(true);
    setMessage("");
    try {
      const r = await recalibrate();
      const fresh = await refresh();
      setSelected((prev) =>
        prev ? fresh.find((a) => a.mmsi === prev.mmsi && a.window_id === prev.window_id) ?? null : null
      );
      setMessage(
        r.ran
          ? `v${r.old_version} to v${r.new_version}: precision ${(r.precision * 100).toFixed(0)}%, ` +
            `threshold ${r.old_percentile.toFixed(1)} to ${r.new_percentile.toFixed(1)}, ` +
            `${r.n_anomalies} anomalies now flagged.`
          : r.message
      );
    } catch (e) {
      setMessage(`Failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  if (error) return <pre style={{ padding: 20, color: "#a3252a" }}>Cannot reach API: {error}</pre>;
  if (!status) return <p style={{ padding: 20 }}>Loading...</p>;

  return (
    <div className="app">
      <header className="header">
        <h1>Maritime Threat Intelligence</h1>
        <span className="muted">
          calibration v{status.calibration.version} &middot;{" "}
          {status.calibration.threshold_percentile.toFixed(1)}th pct &middot; {anomalies.length} flagged &middot;{" "}
          {status.n_feedback} verdicts
        </span>
        <button onClick={handleRecalibrate} disabled={busy}>
          {busy ? "Working..." : "Recalibrate"}
        </button>
        {message && <span className="message">{message}</span>}
      </header>

      <aside className="sidebar">
        <div className="list">
          <AnomalyList anomalies={anomalies} selected={selected} onSelect={setSelected} />
        </div>
        <ExplanationPanel anomaly={selected} onFeedback={handleFeedback} busy={busy} />
      </aside>

      <MapView anomalies={anomalies} selected={selected} track={track} />
    </div>
  );
}