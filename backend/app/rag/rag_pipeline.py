from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from app.utils.json_extract import parse_llm_json

from app.rag.ollama_client import ask_ollama
from app.rag.policy_ingest import normalize, EMBED_MODEL_NAME
from app.rag.vector_store import FaissStore


# Global cache - because loading the SentenceTransformer model is expensive, 
# we use a singleton pattern to load it once and reuse it.
# once per server run, not per request.
_EMBEDDER: SentenceTransformer | None = None

# Get the singleton embedder instance. 
# This ensures we only load the model once and reuse it for all embedding operations.
def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBEDDER


@dataclass
class RiskInputs:
    severity: int          # 1..5 - how severe the potential compliance violation is, based on question content (e.g., logging PHI is more severe than other practices)
    sensitivity: int       # 1..5 - how sensitive the data/practice is, inferred from question (e.g., PHI-related is more sensitive)
    frequency: int         # 1..5 - how often this practice occurs (default medium until we have real frequency data from ELK)
    confidence: float      # 0..1 - confidence in the compliance assessment based on retrieved context (higher confidence reduces review priority)

# Main RAG pipeline function. 
# Takes a question and the vector store, retrieves relevant policy chunks, 
# constructs a prompt, gets an answer from the LLM, and parses it into structured output.

# Used to keep:
# confidence between 0 and 1
# risk scores between 0 and 100
# This prevents weird scores.
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_confidence(retrieved: List[Tuple[str, float]]) -> float:
    """
    Convert cosine similarity-ish scores into a 0..1 confidence.
    We use mean of top-3 scores (clamped). 
    """
    if not retrieved:
        return 0.0
    top_scores = [s for _, s in retrieved[:3]]
    # Scores with normalized vectors are cosine similarity in [-1, 1].
    # Clamp to [0, 1] for confidence.
    conf = float(np.mean(top_scores))
    return clamp(conf, 0.0, 1.0)

# Infer risk inputs from question and confidence.
# This is a simple function that assigns severity and sensitivity based on keywords in the question,
def infer_risk_inputs(question: str, confidence: float) -> RiskInputs:
    """
    GRC-style
    - If PHI is mentioned, sensitivity high
    - Logging PHI generally high severity
    - Frequency is unknown for now, default medium (3)
    """
    q = question.lower()

    sensitivity = 3
    if "phi" in q or "protected health information" in q or "hipaa" in q:
        sensitivity = 5

    severity = 3
    if "log" in q or "logs" in q:
        severity = 4
    if "store" in q and ("phi" in q or "protected health information" in q):
        severity = 5

    frequency = 3  # until ELK-driven real frequency
    return RiskInputs(severity=severity, sensitivity=sensitivity, frequency=frequency, confidence=confidence)


def compute_risk_score(inputs: RiskInputs) -> Dict[str, Any]:
    """
    Two scores:
    1) compliance_risk: inherent risk (severity*sensitivity*frequency) scaled to 0..100
    2) review_priority: how urgently a human should review (low confidence => higher)
    """
    base = inputs.severity * inputs.sensitivity * inputs.frequency  # 1..125
    compliance_risk = (base / 125.0) * 100.0
    compliance_risk = clamp(compliance_risk, 0.0, 100.0)

    # review priority increases when confidence is low
    review_priority = (1.0 - clamp(inputs.confidence, 0.0, 1.0)) * 100.0
    review_priority = clamp(review_priority, 0.0, 100.0)

    def level(score: float) -> str:
        if score >= 75:
            return "Critical"
        if score >= 50:
            return "High"
        if score >= 25:
            return "Medium"
        return "Low"

    return {
        "compliance_risk": {"score": round(compliance_risk, 2), "level": level(compliance_risk)},
        "review_priority": {"score": round(review_priority, 2), "level": level(review_priority)},
        "inputs": {
            "severity": inputs.severity,
            "sensitivity": inputs.sensitivity,
            "frequency": inputs.frequency,
            "confidence": round(inputs.confidence, 3),
        },
    }

# Main RAG pipeline function. 
# Takes a question and the vector store, retrieves relevant policy chunks, 
# constructs a prompt, gets an answer from the LLM, and parses it into structured output.
def answer_with_rag(question: str, store: FaissStore, top_k: int = 5) -> Dict[str, Any]:
    embedder = get_embedder()
    q_emb = embedder.encode([question], convert_to_numpy=True)
    q_emb = normalize(q_emb)

    raw = store.search(q_emb, top_k=top_k * 3)  # pull more, then dedupe
    seen = set()
    retrieved = []
    for chunk, score in raw:
        key = chunk.strip()
        if key in seen:
            continue
        seen.add(key)
        retrieved.append((chunk, score))
        if len(retrieved) >= top_k:
            break


    # Stable citations: [C1], [C2], ...
    context_blocks = []
    for i, (chunk, score) in enumerate(retrieved, start=1):
        context_blocks.append(f"[C{i}] (similarity={score:.3f}) {chunk}")

    context = "\n\n".join(context_blocks) if context_blocks else "NO_CONTEXT_FOUND"

    confidence = compute_confidence(retrieved)
    risk_inputs = infer_risk_inputs(question, confidence)
    risk = compute_risk_score(risk_inputs)

    prompt = f"""
You are a compliance assistant for GRC and audit support.
STRICT RULES (must follow):
1) Use ONLY the provided Policy Context. Do not use outside knowledge.
2) Be conservative:
   - If the context does NOT explicitly address the specific practice asked about, set Compliance status = Unknown.
   - Only set Non-compliant if the context explicitly indicates the practice violates a requirement.
   - Only set Compliant if the context explicitly allows the practice with conditions.
3) Evidence:
   - You MUST include 2–4 evidence bullets if Policy Context is present.
   - Each evidence bullet MUST be a DIRECT QUOTE copied from the Policy Context (no paraphrasing).
   - Each evidence bullet MUST end with an exact citation like [C1] or [C4].
   - If the practice is not explicitly mentioned (e.g., "application logs"), choose the closest relevant governance quotes (minimum necessary, access restriction, disclosures).
4) Do NOT invent quotes. Do NOT use placeholders like [C#]. Only use [C1]..[C{len(retrieved)}].
5) Citations are ONLY allowed in the Evidence section (not in Explanation or Recommended mitigation).
6) Formatting:
   - Put each section header on its own line.
   - Use bullet points under Explanation, Evidence, Recommended mitigation.

Policy Context:
{context}

Question: {question}

Return ONLY valid JSON in this exact schema:

{{
  "compliance_status": "Compliant | Non-compliant | Unknown",
  "explanation": [
    "bullet point 1",
    "bullet point 2"
  ],
  "evidence": [
    {{
      "quote": "direct quote from context",
      "citation": "C1"
    }}
  ],
  "recommended_mitigation": [
    "mitigation step 1",
    "mitigation step 2"
  ],
  "missing_information": [
    "what additional policy detail is needed"
  ]
}}
""".strip()


    llm_answer = ask_ollama(prompt)
    try:
        structured, extracted_json = parse_llm_json(llm_answer)
        parse_error = None
    except Exception as e:
        structured = None
        extracted_json = None
        parse_error = str(e)

    return {
    "question": question,
    "top_k": top_k,
    "confidence": round(confidence, 3),
    "risk": risk,
    "retrieved": [{"id": f"C{i}", "score": float(s), "chunk": c} for i, (c, s) in enumerate(retrieved, start=1)],
    "llm_raw": llm_answer,                 # original model output
    "llm_json_string": extracted_json,     # extracted JSON string (no fences)
    "structured_output": structured,       # parsed dict (or None)
    "parse_error": parse_error             # None if OK
}

    
