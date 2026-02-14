# has endpoints to /ingest log events, /retrain anomaly model, and /recent events.

from __future__ import annotations
import os, json
from datetime import datetime, timezone
from typing import Dict, Any, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.logs.phi_detector import detect_phi
from app.logs.anomaly_model import (
    load_model, train_isolation_forest, featurize, to_vector, score_event
)

router = APIRouter()

# location for log storage
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
LOG_DIR = os.path.join(DATA_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "events.jsonl")

# in-memory cache (MVP)
EVENTS: List[Dict[str, Any]] = []
#If user sends bad JSON, Pydantic validates.
class LogEvent(BaseModel):
    timestamp: str | None = None
    user: str | None = None
    action: str = Field(default="UNKNOWN")
    resource: str | None = None
    message: str = Field(default="")

# This route ingests a log event, detects PHI, scores it for anomalies, and decides if it should be escalated.
@router.post("/ingest")
def ingest(event: LogEvent):
    os.makedirs(LOG_DIR, exist_ok=True)

    # event is a pytantic object
    ev = event.model_dump()
    if not ev.get("timestamp"):
        ev["timestamp"] = datetime.now(timezone.utc).isoformat()

    phi = detect_phi(ev.get("message", ""))
    ev["phi"] = phi

    model = load_model()
    if model is not None:
        feats = featurize(ev, phi)
        vec = to_vector(feats)
        ev["anomaly"] = score_event(model, vec)
    else:
        ev["anomaly"] = {"is_anomaly": False, "normality": None, "anomaly_score": None, "note": "model not trained yet"}

    EVENTS.append(ev)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")

    return {"ingested": True, "phi": phi, "anomaly": ev["anomaly"]}

@router.post("/retrain")
def retrain(limit: int = 500):
    # load last N events from disk (preferred) else from memory
    rows: List[Dict[str, Any]] = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f.readlines()[-limit:]:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    else:
        rows = EVENTS[-limit:]

    if len(rows) < 30:
        return {"trained": False, "reason": "Need at least ~30 events to train a baseline", "events_seen": len(rows)}

    X = []
    for ev in rows:
        phi = ev.get("phi") or detect_phi(ev.get("message", ""))
        feats = featurize(ev, phi)
        X.append(to_vector(feats)[0])

    import numpy as np
    X = np.array(X, dtype=np.float32)
    train_isolation_forest(X) # train and save model
    return {"trained": True, "events_used": len(rows)}

# returns the last N ingested events in memory
@router.get("/recent")
def recent(limit: int = 50):
    return EVENTS[-limit:]
