// One place where every call to the backend lives. Components never
// build URLs or touch fetch directly -- they call these functions.

const BASE = "http://localhost:8000/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      // response had no JSON body; keep the status text
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export const getStatus = () => request("/status");

export const getAnomalies = () => request("/anomalies");

export const getTrack = (mmsi) => request(`/track/${mmsi}`);

export const sendFeedback = (windowId, mmsi, label) =>
  request("/feedback", {
    method: "POST",
    body: JSON.stringify({ window_id: windowId, mmsi, label }),
  });

export const recalibrate = () => request("/recalibrate", { method: "POST" });