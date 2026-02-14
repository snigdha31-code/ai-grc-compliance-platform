from __future__ import annotations
import re
from typing import Dict, Any, List

# use regex and keyword matching to detect potential PHI in log messages. 

# Simple PHI-like patterns (MVP). Not perfect; good for demo + risk scoring.
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
DOB_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")  # very broad
MRN_RE = re.compile(r"\bMRN[:\s-]*\d{5,12}\b", re.IGNORECASE)

# Keywords that often correlate with PHI contexts
PHI_KEYWORDS = [
    "patient", "diagnosis", "treatment", "lab", "prescription", "rx", "insurance",
    "medical record", "mrn", "ssn", "dob", "address"
]
# mrn is medical record number
def detect_phi(text: str) -> Dict[str, Any]:
    text = text or ""
    matches: Dict[str, List[str]] = {
        "email": EMAIL_RE.findall(text),
        "phone": PHONE_RE.findall(text),
        "ssn": SSN_RE.findall(text),
        "dob": DOB_RE.findall(text),
        "mrn": MRN_RE.findall(text),
    }
    keyword_hits = [k for k in PHI_KEYWORDS if k in text.lower()]

    has_pattern = any(len(v) > 0 for v in matches.values())
    has_keywords = len(keyword_hits) > 0

    # heuristic sensitivity
    sensitivity = 5 if (matches["ssn"] or matches["mrn"]) else (4 if has_pattern else (3 if has_keywords else 1))

    return {
        "has_phi": bool(has_pattern or has_keywords),
        "sensitivity": sensitivity,  # 1..5
        "matches": matches,
        "keyword_hits": keyword_hits,
    }
