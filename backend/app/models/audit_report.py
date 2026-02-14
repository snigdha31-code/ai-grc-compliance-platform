from __future__ import annotations
from sqlalchemy import Column, String, Float, DateTime, JSON
from datetime import datetime, timezone
import uuid

from app.models.db import Base

# means it is a SQLAlchemy model that corresponds to a table in the database. 
# Each instance of this class represents a row in the "audit_reports" table. 
# The columns of the table are defined by the attributes of the class, such as report_id, generated_at_utc, policy_source, etc.
class AuditReport(Base):
    __tablename__ = "audit_reports"
    # tells SQLAlchemy to create table names "audit_reports"

    report_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    generated_at_utc = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    policy_source = Column(String, nullable=True)
    question = Column(String, nullable=False)

    compliance_status = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)

    risk = Column(JSON, nullable=True)
    evidence_chunks = Column(JSON, nullable=True)
    review_workflow = Column(JSON, nullable=True)
