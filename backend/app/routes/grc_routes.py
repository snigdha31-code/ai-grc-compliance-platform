# GRC automation routes - file handles 
# log ingestion, PHI detection , anomaly detection, 
# escalation logic, RAG based compliance reasoning, dashbaord summary


from __future__ import annotations
from fastapi import APIRouter
from typing import Dict, Any
from app.rag.store_singleton import store
from app.models.db import SessionLocal
from app.models.audit_report import AuditReport

from app.logs.phi_detector import detect_phi
from app.logs.anomaly_model import load_model, featurize, to_vector, score_event
from app.rag.rag_pipeline import answer_with_rag
from app.rag.audit_report import build_audit_report
from app.routes.rag_routes import store  # reuse vector store
from app.rag.report_persist import save_report_to_db

router = APIRouter() # creates FasrAPI router instance
# in-memory tracking - resets when server restarts
STATS = {
    "events_seen": 0,
    "phi_events": 0,
    "anomalies": 0,
    "escalations": 0
}

# main engine
@router.post("/evaluate-event")
def evaluate_event(event: Dict[str, Any]):
    STATS["events_seen"] += 1

    # Step 1: PHI detection
    phi = detect_phi(event.get("message", ""))
    if phi["has_phi"]:
        STATS["phi_events"] += 1

    # Step 2: anomaly scoring
    model = load_model()
    anomaly = None
    if model:
        feats = featurize(event, phi)
        vec = to_vector(feats)
        anomaly = score_event(model, vec)
        if anomaly and anomaly.get("is_anomaly"):
            STATS["anomalies"] += 1

    else:
        anomaly = {"is_anomaly": False, "anomaly_score": None}

    # Step 3: decide if we escalate
    should_escalate = phi["has_phi"] or (anomaly and anomaly.get("is_anomaly"))

    if not should_escalate:
        return {
            "escalated": False,
            "reason": "No PHI and no anomaly detected",
            "phi": phi,
            "anomaly": anomaly
        }
    STATS["escalations"] += 1

    # Step 4: build compliance question dynamically
    # if escalated
    # prompt engineering - build question for RAG based on event details
    question = f"""
Is the following activity compliant with HIPAA?
Action: {event.get("action")}
Message: {event.get("message")}
Time: {event.get("timestamp")}
"""

    rag_result = answer_with_rag(question, store, top_k=5)
    report = build_audit_report(rag_result, policy_filename="privacysummary.pdf")

    save_report_to_db(report)

    return {
        
        "escalated": True,
        "phi": phi,
        "anomaly": anomaly,
        "report_id": report["report_id"],
        "compliance_status": report["compliance_status"],
        "risk": report["risk"]
    }

@router.get("/dashboard-summary")
def dashboard_summary(limit: int = 10):
    # pull recent audit reports from DB
    db = SessionLocal()
    try:
        # equivalent sql: 
        # SELECT * FROM audit_reports
        # ORDER BY generated_at_utc DESC
        # LIMIT 10;
        rows = (
            db.query(AuditReport)
            .order_by(AuditReport.generated_at_utc.desc())
            .limit(limit)
            .all()
        )
        recent_reports = [
            {
                "report_id": r.report_id,
                "generated_at_utc": r.generated_at_utc.isoformat() if r.generated_at_utc else None,
                "question": r.question,
                "compliance_status": r.compliance_status,
                "confidence": r.confidence,
                "risk": r.risk,
            }
            for r in rows
        ]
    finally:
        db.close()

    # risk levels distribution (from stored report risk)
    risk_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0, "Unknown": 0}
    for rr in recent_reports:
        try:
            lvl = rr["risk"]["review_priority"]["level"]
        except Exception:
            lvl = "Unknown"
        risk_counts[lvl] = risk_counts.get(lvl, 0) + 1

    return {
        "stats": STATS,
        "recent_reports": recent_reports,
        "review_priority_distribution_recent": risk_counts
    }
