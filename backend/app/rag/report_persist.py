from __future__ import annotations
from typing import Dict, Any

from app.models.db import SessionLocal # import the database session factory
from app.models.audit_report import AuditReport # import the ORM model for the audit report
 
# This module defines a function to save the generated audit report into the PostgreSQL database using SQLAlchemy ORM.
def save_report_to_db(report: Dict[str, Any]) -> Dict[str, Any]:
    db = SessionLocal() # create a new database session - start transaction
    try:
        # ORM object creation - we create an instance of the AuditReport class, which corresponds to a row in the audit_reports table.
        row = AuditReport(
    report_id=report["report_id"],
    policy_source=report.get("policy_source"),
    question=report.get("question"),
    compliance_status=report.get("compliance_status"),
    confidence=report.get("confidence"),
    risk=report.get("risk"),
    evidence_chunks=report.get("evidence_chunks"),
    review_workflow=report.get("review_workflow"),
)
 # .get() doesnt give error if key is missing, just returns none
        db.add(row) # add the new report to the session - stage for commit, still in memeory
        db.commit() # commit the transaction - this is where the SQL INSERT happens and data persistance happens is saved to the database
        return {"saved": True, "report_id": row.report_id}
    finally:
        db.close() # close the session - release connection back to pool
