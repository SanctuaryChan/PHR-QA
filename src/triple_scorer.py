from typing import Dict, Tuple


def _tokenize(text: str) -> list:
    return [tok for tok in text.lower().split() if tok]


def compute_question_vec(question: str, vocab_index: Dict[str, int], word_emb):
    import numpy as np

    tokens = _tokenize(question)
    if not tokens:
        return np.zeros(word_emb.shape[1], dtype=word_emb.dtype)

    vecs = []
    for tok in tokens:
        idx = vocab_index.get(tok)
        if idx is None:
            continue
        vecs.append(word_emb[idx])
    if not vecs:
        return np.zeros(word_emb.shape[1], dtype=word_emb.dtype)
    return np.mean(np.stack(vecs, axis=0), axis=0)


def compute_triple_vec(
    h: int,
    r: int,
    t: int,
    entity_emb,
    relation_emb,
) -> "np.ndarray":
    return entity_emb[h] + relation_emb[r] + entity_emb[t]


def cosine_sim(a, b) -> float:
    import numpy as np

    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def score_triple(
    question_vec,
    triple: Dict[str, int],
    entity_emb,
    relation_emb,
) -> float:
    h = triple["h"]
    r = triple["r"]
    t = triple["t"]
    triple_vec = compute_triple_vec(h, r, t, entity_emb, relation_emb)
    return cosine_sim(question_vec, triple_vec)


def check_embeddings_bounds(
    triple: Dict[str, int],
    entity_emb,
    relation_emb,
) -> Tuple[bool, str]:
    h = triple["h"]
    r = triple["r"]
    t = triple["t"]
    if not (0 <= h < entity_emb.shape[0]):
        return False, f"entity_id out of range: h={h}"
    if not (0 <= t < entity_emb.shape[0]):
        return False, f"entity_id out of range: t={t}"
    if not (0 <= r < relation_emb.shape[0]):
        return False, f"relation_id out of range: r={r}"
    return True, ""
