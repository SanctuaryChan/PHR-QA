import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from idmap import IDMap


def extract_candidate_relations(
    sample: Dict[str, Any], idmap: IDMap, topm: Optional[int] = None
) -> List[str]:
    counts: Counter[int] = Counter()
    for t in sample.get("triples", []):
        counts[t["r"]] += 1
    rels = [idmap.relation_text(rid) for rid, _ in counts.most_common()]
    if topm is not None and topm > 0:
        rels = rels[:topm]
    return rels


def build_prompt(question: str, candidate_relations: Sequence[str], topk: int, max_len: int) -> str:
    rel_lines = "\n".join(f"- {r}" for r in candidate_relations)
    return (
        "You are a KGQA planner. Given a question and candidate relations, "
        "output ONLY valid JSON with relation paths.\n\n"
        "Rules:\n"
        f"- Use only relations from the candidate list.\n"
        f"- Each path length must be between 1 and {max_len}.\n"
        f"- Return at most {topk} paths and keep them diverse.\n"
        "- Output exactly one JSON object with this schema: "
        "{\"paths\":[[\"rel1\"],[\"rel2\",\"rel3\"]]}.\n"
        "- If no valid path, output: {\"paths\":[]}.\n"
        "- Do NOT output any extra text, numbering, or markdown.\n\n"
        f"Question: {question.strip()}\n\n"
        "Candidate relations:\n"
        f"{rel_lines}\n\n"
        "JSON:"
    )


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def _extract_first_json(text: str, open_char: str, close_char: str) -> Optional[str]:
    start = text.find(open_char)
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _extract_json_blob(text: str) -> Optional[str]:
    return _extract_first_json(text, "{", "}")


def _extract_json_array(text: str) -> Optional[str]:
    return _extract_first_json(text, "[", "]")


def normalize_paths(paths_obj: Any, max_len: int, topk: int) -> List[List[str]]:
    paths: List[List[str]] = []
    seen = set()

    if isinstance(paths_obj, dict):
        if "paths" in paths_obj:
            paths_obj = paths_obj["paths"]
        elif "path" in paths_obj:
            paths_obj = [paths_obj["path"]]
        elif "rels" in paths_obj:
            paths_obj = [paths_obj["rels"]]

    if not isinstance(paths_obj, list):
        return []

    if paths_obj and all(isinstance(item, str) for item in paths_obj):
        paths_obj = [paths_obj]

    for item in paths_obj:
        rels = None
        if isinstance(item, dict) and "rels" in item:
            rels = item["rels"]
        elif isinstance(item, dict) and "path" in item:
            rels = item["path"]
        elif isinstance(item, list):
            rels = item
        elif isinstance(item, str):
            rels = [item]
        if not isinstance(rels, list):
            continue
        rel_list = [str(r).strip() for r in rels if str(r).strip()]
        if not rel_list:
            continue
        if max_len is not None and max_len > 0 and len(rel_list) > max_len:
            rel_list = rel_list[:max_len]
        if max_len is not None and max_len > 0 and not (1 <= len(rel_list) <= max_len):
            continue
        key = tuple(rel_list)
        if key in seen:
            continue
        seen.add(key)
        paths.append(rel_list)
        if topk is not None and topk > 0 and len(paths) >= topk:
            break
    return paths


def parse_paths(text: str, max_len: int, topk: int) -> List[List[str]]:
    cleaned = _strip_code_fence(text)
    for candidate in (cleaned, _extract_json_blob(cleaned), _extract_json_array(cleaned)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        paths = normalize_paths(obj, max_len, topk)
        if paths:
            return paths

    return []


def fallback_paths(
    question: str, candidate_relations: Sequence[str], topk: int, max_len: int
) -> List[List[str]]:
    if not candidate_relations:
        return []
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    scored: List[Tuple[int, str]] = []
    for rel in candidate_relations:
        rel_tokens = set(re.split(r"[\._/]+", rel.lower()))
        score = len(q_tokens & rel_tokens)
        scored.append((score, rel))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [rel for score, rel in scored if score > 0]
    if not chosen:
        chosen = list(candidate_relations)
    chosen = chosen[: max(topk, 0)] if topk else chosen
    paths = [[rel] for rel in chosen]
    return [p for p in paths if 1 <= len(p) <= max_len]


def plan_paths(
    question: str,
    candidate_relations: Sequence[str],
    reader,
    topk: int,
    max_len: int,
) -> Tuple[List[List[str]], str]:
    prompt = build_prompt(question, candidate_relations, topk, max_len)
    raw = reader.generate(prompt)
    paths = parse_paths(raw, max_len, topk)
    if not paths:
        paths = fallback_paths(question, candidate_relations, topk, max_len)
    return paths, raw


def map_paths_to_ids(
    paths: Iterable[Sequence[str]], relation_to_id: Dict[str, int]
) -> List[List[int]]:
    mapped: List[List[int]] = []
    for rels in paths:
        ids: List[int] = []
        ok = True
        for rel in rels:
            if rel in relation_to_id:
                ids.append(relation_to_id[rel])
            else:
                s = str(rel).strip()
                if s.isdigit():
                    ids.append(int(s))
                else:
                    ok = False
                    break
        if ok and ids:
            mapped.append(ids)
    return mapped
