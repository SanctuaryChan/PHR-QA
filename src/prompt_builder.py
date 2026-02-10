from typing import Any, Dict, List


def build_prompt(question: str, evidence: List[Dict[str, Any]]) -> str:
    _ = evidence
    return question.strip()
