from typing import Any, Dict, List, Sequence, Set

from idmap import IDMap


def _build_adjacency(
    triples: List[Dict[str, int]], direction: str
) -> Dict[int, List[tuple]]:
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


def traverse_paths(
    sample: Dict[str, Any],
    path_ids: Sequence[Sequence[int]],
    direction: str = "out",
) -> Set[int]:
    if not path_ids:
        return set()
    if direction not in ("out", "in", "both"):
        raise ValueError("direction must be out|in|both")

    triples = sample.get("triples", [])
    adj = _build_adjacency(triples, direction)
    hit_indices: Set[int] = set()

    for rel_seq in path_ids:
        if not rel_seq:
            continue
        frontier = set(sample.get("topic_entities", []))
        for rel in rel_seq:
            if not frontier:
                break
            next_frontier = set()
            for node in frontier:
                for r, nbr, idx in adj.get(node, []):
                    if r != rel:
                        continue
                    hit_indices.add(idx)
                    next_frontier.add(nbr)
            frontier = next_frontier
    return hit_indices


def retrieve(
    sample: Dict[str, Any],
    idmap: IDMap,
    path_ids: Sequence[Sequence[int]],
    topn: int | None = None,
    direction: str = "out",
) -> List[Dict[str, Any]]:
    hit_indices = traverse_paths(sample, path_ids, direction=direction)
    evidence: List[Dict[str, Any]] = []
    triples = sample.get("triples", [])
    for idx in hit_indices:
        t = triples[idx]
        evidence.append(
            {
                "h": t["h"],
                "r": t["r"],
                "t": t["t"],
                "score": 1.0,
                "h_text": idmap.entity_text(t["h"]),
                "r_text": idmap.relation_text(t["r"]),
                "t_text": idmap.entity_text(t["t"]),
            }
        )
    if topn is not None and topn > 0:
        return evidence[:topn]
    return evidence
