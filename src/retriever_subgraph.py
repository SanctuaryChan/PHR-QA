from typing import Any, Dict, List, Tuple

from idmap import IDMap
from triple_scorer import check_embeddings_bounds, score_triple


def retrieve(
    sample: Dict[str, Any],
    idmap: IDMap,
    question_vec,
    entity_emb,
    relation_emb,
    topn: int,
    oob_policy: str = "skip",
) -> Tuple[List[Dict[str, Any]], int]:
    scored: List[Dict[str, Any]] = []
    oob_skipped = 0
    for triple in sample["triples"]:
        ok, msg = check_embeddings_bounds(triple, entity_emb, relation_emb)
        if not ok:
            if oob_policy == "error":
                raise ValueError(f"{msg} for sample id={sample['id']}")
            if oob_policy == "skip":
                oob_skipped += 1
                continue
            if oob_policy == "zero":
                score = -1.0
            else:
                raise ValueError(f"Unknown oob_policy: {oob_policy}")
        else:
            score = score_triple(question_vec, triple, entity_emb, relation_emb)
        h = triple["h"]
        r = triple["r"]
        t = triple["t"]
        scored.append(
            {
                "h": h,
                "r": r,
                "t": t,
                "score": score,
                "h_text": idmap.entity_text(h),
                "r_text": idmap.relation_text(r),
                "t_text": idmap.entity_text(t),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    if topn is not None and topn > 0:
        return scored[:topn], oob_skipped
    return scored, oob_skipped
