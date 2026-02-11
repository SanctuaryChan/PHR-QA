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
    path_ids: Sequence[Sequence[int]],
    topn: int,
    alpha: float,
    path_bonus: float,
    direction: str = "out",
    oob_policy: str = "skip",
) -> Tuple[List[Dict[str, Any]], int]:
    scored: List[Dict[str, Any]] = []
    oob_skipped = 0
    path_hits = traverse_paths(sample, path_ids, direction=direction)

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

        path_hit = 1.0 if idx in path_hits else 0.0
        score = alpha * sem_score + path_bonus * path_hit

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
                "h_text": idmap.entity_text(h),
                "r_text": idmap.relation_text(r),
                "t_text": idmap.entity_text(t),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    if topn is not None and topn > 0:
        return scored[:topn], oob_skipped
    return scored, oob_skipped
