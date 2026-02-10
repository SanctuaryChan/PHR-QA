from pathlib import Path
from typing import Dict, Tuple, Union


def _read_vocab(path: Path) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            token = line.strip()
            if token and token not in vocab:
                vocab[token] = idx
    return vocab


def load_embeddings(data_dir: Union[str, Path]) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", Dict[str, int]]:
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("numpy is required for Phase 2. Install: pip install numpy") from e

    data_dir = Path(data_dir)
    ent_path = data_dir / "entity_emb_100d.npy"
    rel_path = data_dir / "relation_emb_100d.npy"
    word_path = data_dir / "word_emb.npy"
    vocab_path = data_dir / "vocab.txt"

    if not ent_path.exists():
        raise FileNotFoundError(f"Missing entity embeddings: {ent_path}")
    if not rel_path.exists():
        raise FileNotFoundError(f"Missing relation embeddings: {rel_path}")
    if not word_path.exists():
        raise FileNotFoundError(f"Missing word embeddings: {word_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab: {vocab_path}")

    entity_emb = np.load(ent_path)
    relation_emb = np.load(rel_path)
    word_emb = np.load(word_path)
    vocab = _read_vocab(vocab_path)

    return entity_emb, relation_emb, word_emb, vocab
