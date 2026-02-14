# Anomaly detection model for log events using Isolation Forest
# Isolation Forest is an unsupervised learning algorithm that identifies anomalies by isolating observations in the feature space. It works by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of that feature. Anomalies are more likely to be isolated in fewer splits, resulting in shorter path lengths in the tree structure. The model is trained on historical log event data to learn the normal patterns of behavior, and then it can score new events based on how anomalous they are compared to the learned patterns.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import os 
import joblib # save/load ML models to disk
import numpy as np
from sklearn.ensemble import IsolationForest

# meaning model file path is backend/app/data/models/iforest.joblib
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "models", "iforest.joblib"))

@dataclass # creates a class meant for holding data (like a typed struct).
# instead of passing messy dicts, dataclass holds like 
# EventFeatures(hour=3, msg_len=120, has_phi=1, sensitivity=5)

class EventFeatures:
    hour: int
    msg_len: int
    has_phi: int
    sensitivity: int

#Convert raw event + PHI detection output into numeric features the model can learn from.
def featurize(event: Dict[str, Any], phi: Dict[str, Any]) -> EventFeatures:
    ts = event.get("timestamp", "")
    hour = 12
    # default to noon if timestamp is missing or malformed, but try to extract hour if possible for better signal.
    # Logs with PHI might have different temporal patterns (e.g. more likely during business hours).
    try:
        hour = int(ts[11:13])
    except Exception:
        pass

    msg = (event.get("message") or "") + " " + (event.get("action") or "")
    msg_len = len(msg)
    has_phi = 1 if phi.get("has_phi") else 0
    sensitivity = int(phi.get("sensitivity", 1))
    return EventFeatures(hour=hour, msg_len=msg_len, has_phi=has_phi, sensitivity=sensitivity)

def to_vector(f: EventFeatures) -> np.ndarray:
    return np.array([[f.hour, f.msg_len, f.has_phi, f.sensitivity]], dtype=np.float32)

def train_isolation_forest(feature_vectors: np.ndarray) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200, # number of trees in the forest - more trees can capture more complex patterns but increase training time
        contamination=0.05, # you assume about 5% of events are anomalies,
        #expected proportion of anomalies in the data - helps model set threshold for anomaly scores, adjust based on domain knowledge
        random_state=42 # for reproducibility - ensures same random splits each time you train
    )
    model.fit(feature_vectors) # train the model on the historical event data - it learns patterns of normal behavior
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model

def load_model() -> IsolationForest | None:
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def score_event(model: IsolationForest, vec: np.ndarray) -> Dict[str, Any]:
    # decision_function higher = more normal; lower = more anomalous
    normality = float(model.decision_function(vec)[0])
    pred = int(model.predict(vec)[0])  # -1 anomalous, 1 normal
    is_anomaly = (pred == -1)
    # convert to 0..100 anomaly score (higher = more anomalous)
    anomaly_score = max(0.0, min(100.0, (0.5 - normality) * 100.0))
    return {"is_anomaly": is_anomaly, "normality": normality, "anomaly_score": round(anomaly_score, 2)}
