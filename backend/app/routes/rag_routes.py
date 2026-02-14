import os
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from app.rag.audit_report import build_audit_report
from app.models.db import SessionLocal
from app.models.audit_report import AuditReport
import csv
import io
from fastapi.responses import StreamingResponse
from app.models.db import SessionLocal
from app.models.audit_report import AuditReport
from app.rag.vector_store import FaissStore
from app.rag.policy_ingest import ingest_pdf_into_faiss
from app.rag.rag_pipeline import answer_with_rag
from app.rag.store_singleton import store
from app.rag.report_persist import save_report_to_db


router = APIRouter()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DATA_DIR = os.path.abspath(DATA_DIR)
UPLOAD_DIR = os.path.join(DATA_DIR, "policies")
INDEX_DIR = os.path.join(DATA_DIR, "faiss")

INDEX_PATH = os.path.join(INDEX_DIR, "policy.index")
META_PATH = os.path.join(INDEX_DIR, "policy_chunks.pkl")


# Load existing index if present
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    store.load()


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


@router.post("/upload-policy")
async def upload_policy(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, file.filename)

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    n_chunks = ingest_pdf_into_faiss(save_path, store)
    return {"filename": file.filename, "chunks_indexed": n_chunks}


@router.post("/query")
def query_policy(req: QuestionRequest):
    if len(store.chunks) == 0:
        return {"error": "No policy indexed yet. Upload a PDF first via /rag/upload-policy"}
    return answer_with_rag(req.question, store, top_k=req.top_k)

@router.get("/health")
def health():
    return {
        "indexed_chunks": len(store.chunks),
        "faiss_index_path": INDEX_PATH,
        "model": "phi3"
    }

@router.post("/audit-report")
def audit_report(req: QuestionRequest):
    if len(store.chunks) == 0:
        return {"error": "No policy indexed yet. Upload a PDF first via /rag/upload-policy"}
    result = answer_with_rag(req.question, store, top_k=req.top_k)

    # if you want, store last uploaded filename globally later
    return build_audit_report(result, policy_filename="privacysummary.pdf")


def save_report_to_db(report: dict) -> dict:
    db = SessionLocal()
    try:
        row = AuditReport(
    report_id=report["report_id"],
    policy_source=report.get("policy_source"),
    question=report.get("question"),
    compliance_status=report.get("compliance_status"),
    confidence=report.get("confidence"),
    risk=report.get("risk"),
    explanation=report.get("explanation"),
    evidence=report.get("evidence"),
    recommended_mitigation=report.get("recommended_mitigation"),
    missing_information=report.get("missing_information"),

    evidence_chunks=report.get("evidence_chunks"),

    llm_raw=report.get("llm_raw"),
    llm_json_string=report.get("llm_json_string"),
    parse_error=report.get("parse_error"),

    review_workflow=report.get("review_workflow"),
)
        db.add(row)
        db.commit()
        return {"saved": True, "report_id": row.report_id}
    finally:
        db.close()

@router.post("/audit-report/save")
def audit_report_save(req: QuestionRequest):
    result = answer_with_rag(req.question, store, top_k=req.top_k)
    report = build_audit_report(result, policy_filename="privacysummary.pdf")
    return save_report_to_db(report)


@router.get("/audit-report/export.csv")
def export_audit_reports_csv(limit: int = 200):
    db = SessionLocal()
    try:
        rows = (
            db.query(AuditReport)
            .order_by(AuditReport.generated_at_utc.desc())
            .limit(limit)
            .all()
        )

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "report_id",
            "generated_at_utc",
            "policy_source",
            "question",
            "compliance_status",
            "confidence",
            "compliance_risk_score",
            "compliance_risk_level",
            "review_priority_score",
            "review_priority_level"
        ])

        for r in rows:
            risk = r.risk or {}
            cr = (risk.get("compliance_risk") or {})
            rp = (risk.get("review_priority") or {})

            writer.writerow([
                r.report_id,
                r.generated_at_utc.isoformat() if r.generated_at_utc else "",
                r.policy_source or "",
                (r.question or "").replace("\n", " ").strip(),
                r.compliance_status,
                r.confidence if r.confidence is not None else "",
                cr.get("score", ""),
                cr.get("level", ""),
                rp.get("score", ""),
                rp.get("level", "")
            ])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=audit_reports.csv"}
        )
    finally:
        db.close()


@router.get("/audit-report/recent")
def audit_report_recent(limit: int = 10):
    db = SessionLocal()
    try:
        rows = (
            db.query(AuditReport)
            .order_by(AuditReport.generated_at_utc.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "report_id": r.report_id,
                "generated_at_utc": r.generated_at_utc.isoformat(),
                "policy_source": r.policy_source,
                "question": r.question,
                "compliance_status": r.compliance_status,
                "confidence": r.confidence,
            }
            for r in rows
        ]
    finally:
        db.close()


@router.get("/audit-report/{report_id}")
def audit_report_get(report_id: str):
    db = SessionLocal()
    try:
        row = db.query(AuditReport).filter(AuditReport.report_id == report_id).first()
        if not row:
            return {"error": "Report not found"}
        return {
            "report_id": row.report_id,
            "generated_at_utc": row.generated_at_utc.isoformat(),
            "policy_source": row.policy_source,
            "question": row.question,
            "compliance_status": row.compliance_status,
            "confidence": row.confidence,
            "risk": row.risk,
            "evidence_chunks": row.evidence_chunks,
            "llm_output": row.llm_output,
            "review_workflow": row.review_workflow,
        }
    finally:
        db.close()
