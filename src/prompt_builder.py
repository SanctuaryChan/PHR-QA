from typing import Any, Dict, List


def _format_evidence(evidence: List[Dict[str, Any]]) -> str:
    lines = []
    for t in evidence:
        h = t.get("h_text", t.get("h", ""))
        r = t.get("r_text", t.get("r", ""))
        tail = t.get("t_text", t.get("t", ""))
        lines.append(f"- {h} | {r} | {tail}")
    return "\n".join(lines)


def build_prompt(question: str, evidence: List[Dict[str, Any]]) -> str:
    question = question.strip()
    header = (
        "You are a factual QA assistant. Answer the question concisely with the final answer only."
    )
    if evidence:
        evidence_block = _format_evidence(evidence)
        return f"{header}\n\nEvidence:\n{evidence_block}\n\nQuestion: {question}\nAnswer:"
    return f"{header}\n\nQuestion: {question}\nAnswer:"
