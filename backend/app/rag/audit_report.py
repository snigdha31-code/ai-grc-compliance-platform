from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4
from typing import Dict, Any




def build_audit_report(query_result: Dict[str, Any], policy_filename: str | None = None) -> Dict[str, Any]:
    report_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

    structured = query_result.get("structured_output") or {}
    compliance_status = structured.get("compliance_status") or query_result.get("compliance_status") or "Unknown"

    return {
        "report_id": report_id,
        "generated_at_utc": now,
        "policy_source": policy_filename or "unknown",
        "question": query_result.get("question"),
        "compliance_status": compliance_status,
        "confidence": query_result.get("confidence"),
        "risk": query_result.get("risk", {}),

        # store structured fields (best for audit + dashboards)
        "explanation": structured.get("explanation", []),
        "evidence": structured.get("evidence", []),
        "recommended_mitigation": structured.get("recommended_mitigation", []),
        "missing_information": structured.get("missing_information", []),

        # keep chunk scores as traceability
        "evidence_chunks": [
            {"id": r.get("id"), "score": r.get("score")}
            for r in query_result.get("retrieved", [])
        ],

        # keep raw for debugging 
        "llm_raw": query_result.get("llm_raw"),
        "llm_json_string": query_result.get("llm_json_string"),
        "parse_error": query_result.get("parse_error"),

        "review_workflow": {
            "owner": "GRC",
            "status": "Needs Review",
            "reviewer": None,
            "review_notes": None
        }
    }


def extract_status(answer_text: str) -> str:
    t = (answer_text or "").lower()
    if "compliance status:" in t:
        # simple parse
        line = [ln.strip() for ln in answer_text.splitlines() if "Compliance status:" in ln]
        if line:
            return line[0].split(":", 1)[1].strip()
    # fallback
    if "non-compliant" in t:
        return "Non-compliant"
    if "compliant" in t:
        return "Compliant"
    return "Unknown"
