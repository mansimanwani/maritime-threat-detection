"""Demo only. Plays the role of an analyst reviewing the flagged list.

Uses data/raw/planted_anomalies.txt, which nothing in backend/ ever reads.
"""

import pandas as pd

from backend.core.db import init_db, insert_feedback, fetch_anomalies, clear_feedback

planted = dict(l.strip().split(",") for l in open("data/raw/planted_anomalies.txt"))

init_db()
clear_feedback()

anomalies = fetch_anomalies()
for a in anomalies:
    label = "true_positive" if str(a["mmsi"]) in planted else "false_positive"
    insert_feedback(a["window_id"], a["mmsi"], label, a["drivers"])

judged = pd.DataFrame(anomalies)
judged["ok"] = judged["mmsi"].astype(str).isin(planted)
print(f"reviewed {len(judged)} flags: {judged.ok.sum()} confirmed, "
      f"{(~judged.ok).sum()} false alarms  (precision {judged.ok.mean():.0%})")