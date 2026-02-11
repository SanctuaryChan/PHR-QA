import argparse
import os
from pathlib import Path
from typing import List, Tuple


def read_questions(path: Path, max_rows: int | None = None) -> List[str]:
    import json

    items: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question", "")
            items.append(str(q))
            if max_rows is not None and len(items) >= max_rows:
                break
    return items


def encode_to_memmap(
    model,
    texts: List[str],
    batch_size: int,
    device: str,
    out_path: Path,
) -> Tuple[Path, int]:
    import numpy as np
    from numpy.lib.format import open_memmap

    dim = model.get_sentence_embedding_dimension()
    mmap = open_memmap(out_path, mode="w+", dtype="float32", shape=(len(texts), dim))
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
    return out_path, dim


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SBERT question embeddings")
    parser.add_argument("--data_dir", required=True, help="Dataset directory with split JSONL")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--model", default="all-mpnet-base-v2", help="HF model id or local path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--offline", action="store_true", help="Force offline local loading")
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename (default: question_emb_sbert_768d_{split}.npy)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split_path = data_dir / f"{args.split}.json"
    if not split_path.exists():
        split_path = data_dir / f"{args.split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers not available. Install: pip install sentence-transformers"
        ) from e

    model = SentenceTransformer(args.model, device=args.device, local_files_only=args.offline)

    questions = read_questions(split_path, max_rows=args.max_rows)
    if not questions:
        raise ValueError("No questions found in split file")

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = data_dir / out_path
    else:
        out_path = data_dir / f"question_emb_sbert_768d_{args.split}.npy"

    print(f"Encoding questions: {len(questions)}")
    encode_to_memmap(model, questions, args.batch_size, args.device, out_path)
    print(f"Question embedding saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
