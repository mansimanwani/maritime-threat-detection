import { useEffect } from "react";
import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const DEFAULT_CENTER = [12.75, 45.0];
const DEFAULT_ZOOM = 8;

const COLORS = {
  true_positive: "#c0392b",
  false_positive: "#95a5a6",
  unjudged: "#e08a1e",
};

function markerColor(anomaly) {
  return COLORS[anomaly.feedback_label] ?? COLORS.unjudged;
}

function Recenter({ position }) {
  const map = useMap();
  useEffect(() => {
    if (position) map.flyTo(position, Math.max(map.getZoom(), 10), { duration: 0.6 });
  }, [position, map]);
  return null;
}

export default function MapView({ anomalies, selected, track }) {
  const focus = selected ? [selected.lat, selected.lon] : null;

  return (
    <MapContainer center={DEFAULT_CENTER} zoom={DEFAULT_ZOOM} className="map">
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />

      {track.length > 0 && (
        <Polyline positions={track.map((p) => [p.lat, p.lon])} color="#2c6fb5" weight={2} opacity={0.8} />
      )}

      {anomalies.map((a) => {
        const isSelected = selected && selected.id === a.id;
        return (
          <CircleMarker
            key={a.id}
            center={[a.lat, a.lon]}
            radius={isSelected ? 11 : 7}
            pathOptions={{
              color: isSelected ? "#16283c" : markerColor(a),
              weight: isSelected ? 3 : 1,
              fillColor: markerColor(a),
              fillOpacity: 0.85,
            }}
          >
            <Tooltip>
              vessel {a.mmsi} &middot; window {a.window_id} &middot; score {a.score.toFixed(2)}
            </Tooltip>
          </CircleMarker>
        );
      })}

      <Recenter position={focus} />
    </MapContainer>
  );
}