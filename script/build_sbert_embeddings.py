import argparse
import os
from pathlib import Path
from typing import List, Tuple


def read_lines(path: Path, max_rows: int | None = None) -> List[str]:
    items: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(line)
            if max_rows is not None and len(items) >= max_rows:
                break
    return items


def encode_to_memmap(
    model,
    texts: List[str],
    batch_size: int,
    device: str,
    tmp_path: Path,
) -> Tuple[Path, int]:
    import numpy as np
    from numpy.lib.format import open_memmap

    dim = model.get_sentence_embedding_dimension()
    mmap = open_memmap(tmp_path, mode="w+", dtype="float32", shape=(len(texts), dim))
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=True,
            device=device,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        mmap[i : i + len(batch)] = emb.astype("float32")
    del mmap
    return tmp_path, dim


def pca_reduce(
    src_path: Path,
    out_path: Path,
    dim: int,
    batch_size: int,
) -> None:
    import numpy as np
    from numpy.lib.format import open_memmap
    from sklearn.decomposition import IncrementalPCA

    src = np.load(src_path, mmap_mode="r")
    ipca = IncrementalPCA(n_components=dim, batch_size=batch_size)
    for i in range(0, src.shape[0], batch_size):
        ipca.partial_fit(src[i : i + batch_size])

    out = open_memmap(out_path, mode="w+", dtype="float32", shape=(src.shape[0], dim))
    for i in range(0, src.shape[0], batch_size):
        out[i : i + batch_size] = ipca.transform(src[i : i + batch_size]).astype("float32")
    del out


def build_embeddings(
    data_dir: Path,
    model_name: str,
    entity_out: Path,
    relation_out: Path,
    dim: int,
    batch_size: int,
    pca_batch_size: int,
    device: str,
    max_rows: int | None,
    keep_tmp: bool,
) -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers not available. Install: pip install sentence-transformers"
        ) from e

    entities = read_lines(data_dir / "entities.txt", max_rows=max_rows)
    relations = read_lines(data_dir / "relations.txt", max_rows=max_rows)
    if not entities:
        raise ValueError("entities.txt is empty")
    if not relations:
        raise ValueError("relations.txt is empty")

    model = SentenceTransformer(model_name, device=device)

    cache_dir = data_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ent_tmp = cache_dir / "entity_emb_tmp.npy"
    rel_tmp = cache_dir / "relation_emb_tmp.npy"

    print(f"Encoding entities: {len(entities)}")
    encode_to_memmap(model, entities, batch_size, device, ent_tmp)
    print("Running PCA for entities")
    pca_reduce(ent_tmp, entity_out, dim, pca_batch_size)

    print(f"Encoding relations: {len(relations)}")
    encode_to_memmap(model, relations, batch_size, device, rel_tmp)
    print("Running PCA for relations")
    pca_reduce(rel_tmp, relation_out, dim, pca_batch_size)

    if not keep_tmp:
        for p in (ent_tmp, rel_tmp):
            try:
                os.remove(p)
            except OSError:
                pass

    print(f"Entity embedding saved: {entity_out}")
    print(f"Relation embedding saved: {relation_out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SBERT embeddings for entities/relations")
    parser.add_argument("--data_dir", required=True, help="Dataset directory with entities.txt/relations.txt")
    parser.add_argument("--model", default="all-mpnet-base-v2")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pca_batch_size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--keep_tmp", action="store_true")
    parser.add_argument(
        "--entity_out",
        default="entity_emb_sbert_100d.npy",
        help="Output filename for entity embeddings",
    )
    parser.add_argument(
        "--relation_out",
        default="relation_emb_sbert_100d.npy",
        help="Output filename for relation embeddings",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    entity_out = Path(args.entity_out)
    relation_out = Path(args.relation_out)
    if not entity_out.is_absolute():
        entity_out = data_dir / entity_out
    if not relation_out.is_absolute():
        relation_out = data_dir / relation_out

    build_embeddings(
        data_dir=data_dir,
        model_name=args.model,
        entity_out=entity_out,
        relation_out=relation_out,
        dim=args.dim,
        batch_size=args.batch_size,
        pca_batch_size=args.pca_batch_size,
        device=args.device,
        max_rows=args.max_rows,
        keep_tmp=args.keep_tmp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
