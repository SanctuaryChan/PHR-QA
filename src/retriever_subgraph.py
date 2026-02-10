from typing import Any, Dict, List

from idmap import IDMap
from triple_scorer import check_embeddings_bounds, score_triple


def retrieve(
    sample: Dict[str, Any],
    idmap: IDMap,
    question_vec,
    entity_emb,
    relation_emb,
    topn: int,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for triple in sample["triples"]:
        ok, msg = check_embeddings_bounds(triple, entity_emb, relation_emb)
        if not ok:
            raise ValueError(f"{msg} for sample id={sample['id']}")
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
        return scored[:topn]
    return scored
