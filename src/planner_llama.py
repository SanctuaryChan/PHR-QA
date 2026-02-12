import difflib
import json
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from idmap import IDMap


_RELATION_ALIAS = {
    "place of birth": ["birthplace", "born in"],
    "date of birth": ["birth date", "born on", "birthday"],
    "nationality": ["citizenship", "country"],
    "spouse": ["wife", "husband"],
    "children": ["child", "son", "daughter"],
    "parent": ["father", "mother"],
    "gender": ["sex"],
    "profession": ["occupation", "job"],
    "height": ["tall", "height"],
    "located in": ["location", "place in"],
    "capital": ["capital city"],
}

_SLOT_KEYWORDS = {
    "person": ["who", "whom", "whose", "person", "people", "谁", "哪位", "人物"],
    "location": [
        "where",
        "location",
        "place",
        "city",
        "country",
        "state",
        "province",
        "哪",
        "哪里",
        "哪儿",
        "地方",
        "城市",
        "国家",
        "州",
        "省",
    ],
    "time": ["when", "date", "year", "time", "day", "born", "birth", "何时", "哪年", "年", "月", "日", "时间", "出生"],
    "number": ["how many", "how much", "number", "amount", "多少", "几", "数量"],
    "organization": [
        "which organization",
        "organization",
        "company",
        "school",
        "university",
        "institution",
        "team",
        "club",
        "哪个公司",
        "公司",
        "学校",
        "大学",
        "机构",
        "组织",
        "球队",
        "俱乐部",
    ],
}

_SLOT_REL_TOKENS = {
    "person": ["person", "people", "actor", "author", "singer", "player", "coach", "spouse", "parent", "child"],
    "location": ["place", "location", "city", "country", "state", "province", "birthplace"],
    "time": ["date", "time", "year", "born", "birth", "birthday"],
    "number": ["number", "count", "amount", "population", "height", "length", "area", "age"],
    "organization": ["organization", "company", "school", "university", "institution", "team", "club"],
}


def _tokenize_text(text: str) -> List[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]


def _align_dim(vec, target_dim: int):
    import numpy as np

    if vec.shape[0] == target_dim:
        return vec
    if vec.shape[0] > target_dim:
        return vec[:target_dim]
    pad = np.zeros(target_dim - vec.shape[0], dtype=vec.dtype)
    return np.concatenate([vec, pad], axis=0)


def cosine_sim(a, b) -> float:
    import numpy as np

    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalize_relation_text(rel: str) -> str:
    text = rel.replace("/", " ").replace("_", " ").replace(".", " ").lower()
    text = re.sub(r"\s+", " ", text).strip()
    expanded = [text]
    for key, aliases in _RELATION_ALIAS.items():
        if key in text:
            expanded.extend(aliases)
    return " ".join(expanded)


def detect_slots(question: str) -> Dict[str, float]:
    q = question.lower()
    slots: Dict[str, float] = {}
    for slot, keywords in _SLOT_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                slots[slot] = 1.0
                break
    return slots


def type_prior_score(rel_tokens: Sequence[str], slots: Dict[str, float]) -> float:
    if not slots:
        return 0.0
    rel_token_set = set(rel_tokens)
    matched = 0
    for slot in slots:
        if rel_token_set.intersection(_SLOT_REL_TOKENS.get(slot, [])):
            matched += 1
    return matched / max(1, len(slots))


class SbertScorer:
    def __init__(self, model_name_or_path: str, device: str = "cuda", offline: bool = False) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "sentence-transformers not available. Install: pip install sentence-transformers"
            ) from e

        self.model = SentenceTransformer(
            model_name_or_path,
            device=device,
            local_files_only=offline,
        )
        self.dim = self.model.get_sentence_embedding_dimension()
        self._cache: Dict[str, "np.ndarray"] = {}

    def encode(self, text: str):
        if text in self._cache:
            return self._cache[text]
        emb = self.model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )[0]
        self._cache[text] = emb
        return emb

    def similarity(self, question_text: str, question_vec, rel_text: str) -> float:
        q_vec = question_vec
        if q_vec is None or (hasattr(q_vec, "shape") and q_vec.shape[0] != self.dim):
            q_vec = self.encode(question_text)
        rel_vec = self.encode(rel_text)
        return cosine_sim(q_vec, rel_vec)

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


def build_relation_pool(
    sample: Dict[str, Any],
    idmap: IDMap,
    question: str,
    question_vec,
    relation_emb,
    topm: Optional[int],
    lambda_freq: float,
    lambda_sem: float,
    lambda_type: float,
    sem_scorer: Optional[SbertScorer] = None,
    type_decay: float = 0.3,
) -> List[Dict[str, Any]]:
    counts: Counter[int] = Counter()
    for t in sample.get("triples", []):
        counts[t["r"]] += 1
    if not counts:
        return []

    max_count = max(counts.values())
    question_tokens = _tokenize_text(question)
    slots = detect_slots(question)
    slot_strength = 1.0 if slots else type_decay

    pool: List[Dict[str, Any]] = []
    for rid, cnt in counts.most_common():
        rel_text = idmap.relation_text(rid)
        norm_text = normalize_relation_text(rel_text)
        rel_tokens = _tokenize_text(norm_text)

        freq_score = cnt / max_count if max_count else 0.0

        emb_score = 0.0
        if sem_scorer is not None:
            emb_score = sem_scorer.similarity(question, question_vec, norm_text)
        elif relation_emb is not None and question_vec is not None and rid < relation_emb.shape[0]:
            q_vec = question_vec
            if hasattr(q_vec, "shape") and q_vec.shape[0] != relation_emb.shape[1]:
                q_vec = _align_dim(q_vec, relation_emb.shape[1])
            emb_score = cosine_sim(q_vec, relation_emb[rid])

        lex_score = 0.0
        if question_tokens and rel_tokens:
            overlap = len(set(question_tokens) & set(rel_tokens))
            lex_score = overlap / max(1, len(set(rel_tokens)))

        sem_score = 0.7 * emb_score + 0.3 * lex_score
        type_score = type_prior_score(rel_tokens, slots) * slot_strength

        score = lambda_freq * freq_score + lambda_sem * sem_score + lambda_type * type_score
        pool.append(
            {
                "rel": rel_text,
                "rel_id": rid,
                "score": score,
                "freq": freq_score,
                "sem": sem_score,
                "type": type_score,
                "norm_text": norm_text,
                "tokens": rel_tokens,
            }
        )

    pool.sort(key=lambda x: x["score"], reverse=True)
    if topm is not None and topm > 0:
        pool = pool[:topm]
    return pool


def _format_relation_pool(pool: Sequence[Any]) -> str:
    lines: List[str] = []
    for item in pool:
        if isinstance(item, dict):
            rel = item.get("rel", "")
            score = item.get("score")
            if score is not None:
                lines.append(f"- {rel} | score={score:.3f}")
            else:
                lines.append(f"- {rel}")
        else:
            lines.append(f"- {item}")
    return "\n".join(lines)

def build_prompt(question: str, candidate_relations: Sequence[Any], topk: int, max_len: int) -> str:
    rel_lines = _format_relation_pool(candidate_relations)
    return (
        "You are a KGQA planner. Given a question and candidate relations, "
        "output ONLY valid JSON with relation paths.\n\n"
        "Rules:\n"
        f"- Use only relations from the candidate list.\n"
        f"- Each path length must be between 1 and {max_len}.\n"
        f"- Return at most {topk} paths and keep them diverse.\n"
        "- Output exactly one JSON object with this schema:\n"
        "  {\"paths\":[{\"rels\":[\"r1\",\"r2\"],\"confidence\":0.0,\"reason\":\"\"}]}\n"
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


def normalize_path_infos(
    paths_obj: Any,
    max_len: int,
    topk: int,
    rel_allowlist: Optional[set] = None,
) -> List[Dict[str, Any]]:
    paths: List[Dict[str, Any]] = []
    seen = set()

    if isinstance(paths_obj, dict) and "paths" in paths_obj:
        paths_obj = paths_obj["paths"]

    if not isinstance(paths_obj, list):
        return []

    if paths_obj and all(isinstance(item, str) for item in paths_obj):
        paths_obj = [paths_obj]

    for item in paths_obj:
        rels = None
        planner_conf = 0.5
        reason = None
        if isinstance(item, dict):
            if "rels" in item:
                rels = item["rels"]
            elif "path" in item:
                rels = item["path"]
            elif "relations" in item:
                rels = item["relations"]
            if "confidence" in item:
                try:
                    planner_conf = float(item["confidence"])
                except Exception:
                    planner_conf = 0.5
            elif "score" in item:
                try:
                    planner_conf = float(item["score"])
                except Exception:
                    planner_conf = 0.5
            reason = item.get("reason")
        elif isinstance(item, list):
            rels = item
        elif isinstance(item, str):
            rels = [item]

        if not isinstance(rels, list):
            continue
        rel_list = [str(r).strip() for r in rels if str(r).strip()]
        if rel_allowlist is not None:
            rel_list = [r for r in rel_list if r in rel_allowlist]
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
        paths.append(
            {
                "rels": rel_list,
                "planner_conf": max(0.0, min(1.0, planner_conf)),
                "reason": reason,
            }
        )
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


def parse_paths_with_meta(
    text: str,
    max_len: int,
    topk: int,
    rel_allowlist: Optional[set] = None,
) -> List[Dict[str, Any]]:
    cleaned = _strip_code_fence(text)
    for candidate in (cleaned, _extract_json_blob(cleaned), _extract_json_array(cleaned)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        paths = normalize_path_infos(obj, max_len, topk, rel_allowlist=rel_allowlist)
        if paths:
            return paths
    return []


def _build_adjacency(triples: List[Dict[str, int]], direction: str) -> Dict[int, List[tuple]]:
    adj: Dict[int, List[tuple]] = {}
    for idx, t in enumerate(triples):
        h = t["h"]
        r = t["r"]
        tail = t["t"]
        if direction in ("out", "both"):
            adj.setdefault(h, []).append((r, tail, idx))
        if direction in ("in", "both"):
            adj.setdefault(tail, []).append((r, h, idx))
    return adj


def compute_path_reachability(
    sample: Dict[str, Any],
    rel_ids: Sequence[int],
    direction: str,
    branch_cap: int,
) -> Tuple[float, float]:
    if not rel_ids:
        return 0.0, 0.0
    triples = sample.get("triples", [])
    adj = _build_adjacency(triples, direction)
    frontier = set(sample.get("topic_entities", []))
    if not frontier:
        return 0.0, 0.0

    branch_vals: List[float] = []
    complete = True
    for rel in rel_ids:
        if not frontier:
            complete = False
            break
        next_frontier = set()
        branch_counts: List[int] = []
        for node in frontier:
            matches = [nbr for r, nbr, _idx in adj.get(node, []) if r == rel]
            branch_counts.append(len(matches))
            for nbr in matches:
                next_frontier.add(nbr)
        if branch_counts:
            branch_vals.append(sum(branch_counts) / len(branch_counts))
        frontier = next_frontier
    if not frontier:
        complete = False

    reachability = len(frontier) / max(1, len(sample.get("topic_entities", []))) if complete else 0.0
    avg_branch = sum(branch_vals) / len(branch_vals) if branch_vals else 0.0
    if branch_cap <= 0:
        branch_penalty = 0.0
    else:
        branch_penalty = math.log1p(avg_branch) / math.log1p(branch_cap)
        branch_penalty = min(1.0, max(0.0, branch_penalty))
    return reachability, branch_penalty


def _cov_score(question: str, rel_tokens: Sequence[str]) -> float:
    slots = detect_slots(question)
    if not slots:
        return 0.0
    rel_token_set = set(rel_tokens)
    matched = 0
    for slot in slots:
        if rel_token_set.intersection(_SLOT_REL_TOKENS.get(slot, [])):
            matched += 1
    return matched / max(1, len(slots))


def rerank_paths(
    paths: List[Dict[str, Any]],
    question: str,
    relation_pool: Sequence[Dict[str, Any]],
    sample: Dict[str, Any],
    question_vec,
    relation_emb,
    max_len: int,
    w_sem: float,
    w_cov: float,
    w_reach: float,
    w_branch: float,
    w_len: float,
    direction: str,
    sem_scorer: Optional[SbertScorer],
    branch_cap: int,
    topk: int,
) -> List[Dict[str, Any]]:
    if not paths:
        return []
    rel_to_meta = {r["rel"]: r for r in relation_pool}
    rel_to_id = {r["rel"]: r["rel_id"] for r in relation_pool}
    rel_to_sem = {r["rel"]: r.get("sem", 0.0) for r in relation_pool}

    scored: List[Dict[str, Any]] = []
    for p in paths:
        rels = p.get("rels", [])
        if not rels:
            continue
        rel_ids = p.get("rel_ids")
        if not rel_ids:
            rel_ids = [rel_to_id.get(r) for r in rels if r in rel_to_id]
        if len(rel_ids) != len(rels):
            continue

        rel_tokens: List[str] = []
        for r in rels:
            rel_tokens.extend(rel_to_meta.get(r, {}).get("tokens", []))

        f_cov = _cov_score(question, rel_tokens)
        f_len = (len(rels) / max(1, max_len)) if max_len else 0.0

        if sem_scorer is not None:
            path_text = " ".join(normalize_relation_text(r) for r in rels)
            f_sem = sem_scorer.similarity(question, question_vec, path_text)
        else:
            f_sem = sum(rel_to_sem.get(r, 0.0) for r in rels) / max(1, len(rels))

        f_reach, f_branch = compute_path_reachability(
            sample, rel_ids, direction=direction, branch_cap=branch_cap
        )

        rank_score = w_sem * f_sem + w_cov * f_cov + w_reach * f_reach - w_branch * f_branch - w_len * f_len
        map_conf = p.get("map_conf") or []
        if map_conf:
            rank_score *= sum(map_conf) / len(map_conf)
        path_score = w_sem * f_sem + w_cov * f_cov - w_branch * f_branch - w_len * f_len

        scored.append(
            {
                **p,
                "rel_ids": rel_ids,
                "path_score": float(path_score),
                "reachability": float(f_reach),
                "rank_score": float(rank_score),
                "f_sem": float(f_sem),
                "f_cov": float(f_cov),
                "f_reach": float(f_reach),
                "f_branch": float(f_branch),
                "f_len": float(f_len),
            }
        )

    scored.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)
    if topk is not None and topk > 0:
        scored = scored[:topk]
    return scored


def fallback_paths(
    question: str,
    relation_pool: Sequence[Dict[str, Any]],
    topk: int,
    max_len: int,
    sample: Dict[str, Any],
    direction: str,
    max_templates: int = 5,
    branch_cap: int = 50,
) -> List[Dict[str, Any]]:
    if not relation_pool:
        return []

    paths: List[Dict[str, Any]] = []
    for rel in relation_pool:
        if max_len < 1:
            continue
        paths.append(
            {
                "rels": [rel["rel"]],
                "planner_conf": float(rel.get("score", 0.0)),
                "source": "fallback",
            }
        )

    if max_len >= 2:
        top_rels = relation_pool[: min(5, len(relation_pool))]
        added = 0
        for i, r1 in enumerate(top_rels):
            for j, r2 in enumerate(top_rels):
                if i == j:
                    continue
                rel_ids = [r1["rel_id"], r2["rel_id"]]
                reach, _branch = compute_path_reachability(
                    sample, rel_ids, direction=direction, branch_cap=branch_cap
                )
                if reach <= 0.0:
                    continue
                paths.append(
                    {
                        "rels": [r1["rel"], r2["rel"]],
                        "planner_conf": float((r1.get("score", 0.0) + r2.get("score", 0.0)) / 2),
                        "source": "fallback",
                    }
                )
                added += 1
                if added >= max_templates:
                    break
            if added >= max_templates:
                break

    if topk is not None and topk > 0:
        paths = paths[:topk]
    return paths


def plan_paths(
    sample: Dict[str, Any],
    question: str,
    relation_pool: Sequence[Dict[str, Any]],
    reader,
    relation_to_id: Dict[str, int],
    question_vec,
    relation_emb,
    topk: int,
    proposal_k: int,
    max_len: int,
    w_sem: float,
    w_cov: float,
    w_reach: float,
    w_branch: float,
    w_len: float,
    direction: str,
    sem_scorer: Optional[SbertScorer],
    branch_cap: int,
    fallback_max_templates: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    rel_allowlist = {r["rel"] for r in relation_pool}
    raw: Optional[str] = None
    proposals: List[Dict[str, Any]] = []

    if reader is not None:
        prompt = build_prompt(question, relation_pool, proposal_k, max_len)
        raw = reader.generate(prompt)
        proposals = parse_paths_with_meta(raw, max_len, proposal_k, rel_allowlist=rel_allowlist)

    if not proposals:
        proposals = fallback_paths(
            question,
            relation_pool,
            proposal_k,
            max_len,
            sample,
            direction=direction,
            max_templates=fallback_max_templates,
            branch_cap=branch_cap,
        )

    for p in proposals:
        p.setdefault("source", "llm")

    mapped = map_paths_to_ids(proposals, relation_to_id, relation_pool=relation_pool)
    reranked = rerank_paths(
        mapped,
        question,
        relation_pool,
        sample,
        question_vec,
        relation_emb,
        max_len=max_len,
        w_sem=w_sem,
        w_cov=w_cov,
        w_reach=w_reach,
        w_branch=w_branch,
        w_len=w_len,
        direction=direction,
        sem_scorer=sem_scorer,
        branch_cap=branch_cap,
        topk=topk,
    )

    for idx, p in enumerate(reranked):
        p["path_id"] = f"p{idx}"
    return reranked, raw


_CANON_REPLACEMENTS = {
    "birthplace": "place of birth",
    "born in": "place of birth",
    "born on": "date of birth",
    "birth date": "date of birth",
    "citizenship": "nationality",
}


def canonicalize_relation_name(rel: str) -> str:
    text = rel.lower().replace("/", " ").replace("_", " ").replace(".", " ")
    for key, repl in _CANON_REPLACEMENTS.items():
        if key in text:
            text = text.replace(key, repl)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_relation_lookup(relation_names: Iterable[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for rel in relation_names:
        lookup[canonicalize_relation_name(rel)] = rel
    return lookup


def _map_relation_name(
    rel: str,
    relation_to_id: Dict[str, int],
    relation_lookup: Dict[str, str],
) -> Tuple[Optional[int], float]:
    if rel in relation_to_id:
        return relation_to_id[rel], 1.0
    canon = canonicalize_relation_name(rel)
    if canon in relation_lookup:
        return relation_to_id[relation_lookup[canon]], 0.9

    best_rel = None
    best_ratio = 0.0
    for canon_name, orig_rel in relation_lookup.items():
        ratio = difflib.SequenceMatcher(a=canon, b=canon_name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_rel = orig_rel
    if best_rel is not None and best_ratio >= 0.75:
        return relation_to_id[best_rel], 0.7
    return None, 0.0


def map_paths_to_ids(
    paths: Iterable[Dict[str, Any]],
    relation_to_id: Dict[str, int],
    relation_pool: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if relation_pool is not None:
        pool_rels = [r["rel"] for r in relation_pool]
    else:
        pool_rels = list(relation_to_id.keys())
    relation_lookup = _build_relation_lookup(pool_rels)

    mapped: List[Dict[str, Any]] = []
    for p in paths:
        rels = p.get("rels", [])
        if not rels:
            continue
        rel_ids: List[int] = []
        confs: List[float] = []
        ok = True
        for rel in rels:
            if isinstance(rel, int):
                rel_ids.append(rel)
                confs.append(1.0)
                continue
            rel_id, conf = _map_relation_name(str(rel), relation_to_id, relation_lookup)
            if rel_id is None:
                ok = False
                break
            rel_ids.append(rel_id)
            confs.append(conf)
        if not ok or not rel_ids:
            continue
        mapped.append(
            {
                **p,
                "rel_ids": rel_ids,
                "map_conf": confs,
            }
        )
    return mapped
