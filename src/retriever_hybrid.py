from typing import Any, Dict, List, Sequence, Tuple

from idmap import IDMap
from retriever_path import traverse_paths
from triple_scorer import check_embeddings_bounds, score_triple


def retrieve(
    sample: Dict[str, Any],
    idmap: IDMap,
    question_vec,
    entity_emb,
    relation_emb,
    paths: Sequence[Any],
    topn: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    direction: str = "out",
    oob_policy: str = "skip",
) -> Tuple[List[Dict[str, Any]], int]:
    scored: List[Dict[str, Any]] = []
    oob_skipped = 0
    path_hits: Dict[int, Dict[str, float]] = {}
    for p in paths or []:
        if isinstance(p, dict):
            rel_ids = p.get("rel_ids") or []
            path_score = float(p.get("path_score", 0.0))
            reachability = float(p.get("reachability", 0.0))
            map_conf = p.get("map_conf") or []
            if map_conf:
                conf = sum(map_conf) / len(map_conf)
                path_score *= conf
                reachability *= conf
        else:
            rel_ids = list(p) if isinstance(p, (list, tuple)) else []
            path_score = 0.0
            reachability = 0.0

        if not rel_ids:
            continue
        hit_indices = traverse_paths(sample, [rel_ids], direction=direction)
        for idx in hit_indices:
            slot = path_hits.setdefault(idx, {"path_score": 0.0, "reachability": 0.0})
            if path_score > slot["path_score"]:
                slot["path_score"] = path_score
            if reachability > slot["reachability"]:
                slot["reachability"] = reachability

    for idx, triple in enumerate(sample.get("triples", [])):
        ok, msg = check_embeddings_bounds(triple, entity_emb, relation_emb)
        if not ok:
            if oob_policy == "error":
                raise ValueError(f"{msg} for sample id={sample['id']}")
            if oob_policy == "skip":
                oob_skipped += 1
                continue
            if oob_policy == "zero":
                sem_score = -1.0
            else:
                raise ValueError(f"Unknown oob_policy: {oob_policy}")
        else:
            sem_score = score_triple(question_vec, triple, entity_emb, relation_emb)

        hit_info = path_hits.get(idx)
        path_hit = 1.0 if hit_info is not None else 0.0
        path_score = hit_info["path_score"] if hit_info else 0.0
        reachability = hit_info["reachability"] if hit_info else 0.0
        score = alpha * sem_score + beta * path_hit + gamma * path_score + delta * reachability

        h = triple["h"]
        r = triple["r"]
        t = triple["t"]
        scored.append(
            {
                "h": h,
                "r": r,
                "t": t,
                "score": score,
                "sem_score": sem_score,
                "path_hit": path_hit,
                "path_score": path_score,
                "path_reachability": reachability,
                "h_text": idmap.entity_text(h),
                "r_text": idmap.relation_text(r),
                "t_text": idmap.entity_text(t),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    if topn is not None and topn > 0:
        return scored[:topn], oob_skipped
    return scored, oob_skipped
