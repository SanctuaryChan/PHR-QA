import argparse
import json
from pathlib import Path
from typing import List, Optional

from dataloader import load_gnnrag_split
from embeddings import load_embeddings
from eval import compute_em_f1, write_summary_csv
from idmap import IDMap
from prompt_builder import build_prompt
from reader_llama import build_reader
from retriever_llm_only import retrieve as retrieve_llm_only
from retriever_subgraph import retrieve as retrieve_subgraph
from triple_scorer import compute_question_vec


def _format_topic_entities(topic_entities: List[int], idmap: IDMap) -> str:
    parts = [f"{eid}:{idmap.entity_text(eid)}" for eid in topic_entities]
    return "[" + ", ".join(parts) + "]"


def cmd_phase0(args: argparse.Namespace) -> int:
    samples = load_gnnrag_split(args.data_dir, args.split, limit=args.limit)
    idmap = IDMap.from_dir(args.data_dir)

    for i, s in enumerate(samples):
        print(f"[{i}] id={s['id']}")
        print(f"question={s['question']}")
        print(f"topic_entities={_format_topic_entities(s['topic_entities'], idmap)}")
        print(f"num_triples={len(s['triples'])}")
        print("")
    return 0


def _infer_dataset_name(data_dir: str) -> str:
    return Path(data_dir).name or "dataset"


class _SimpleProgress:
    def __init__(self, total: int, every: int) -> None:
        self.total = total
        self.every = max(1, every)
        self.count = 0

    def update(self, n: int = 1) -> None:
        self.count += n
        if self.count % self.every == 0 or self.count >= self.total:
            print(f"Progress: {self.count}/{self.total}", flush=True)

    def close(self) -> None:
        return None


def _build_progress(total: int, enabled: bool, every: int):
    if not enabled:
        return None
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(total=total)
    except Exception:
        return _SimpleProgress(total, every)


def cmd_phase1(args: argparse.Namespace) -> int:
    samples = load_gnnrag_split(args.data_dir, args.split, limit=args.limit)
    if args.progress:
        print(f"Loaded {len(samples)} samples, initializing reader={args.reader}...", flush=True)
    reader = build_reader(
        args.reader,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        chat_template=args.chat_template,
    )

    dataset = args.dataset or _infer_dataset_name(args.data_dir)
    out_dir = Path(args.output_dir) / dataset / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "llm_only.jsonl"
    summary_path = out_dir / "summary.csv"

    total_em = 0.0
    total_f1 = 0.0
    count = 0

    progress = _build_progress(len(samples), args.progress, args.progress_every)
    printed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            evidence = retrieve_llm_only(s)
            prompt = build_prompt(s["question"], evidence)
            pred = reader.generate(prompt)
            em, f1 = compute_em_f1(pred, s["gold_answer_texts"])

            record = {
                "id": s["id"],
                "question": s["question"],
                "prediction": pred,
                "gold_answers": s["gold_answer_texts"],
                "em": em,
                "f1": f1,
            }
            if args.save_prompt:
                record["prompt"] = prompt
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_em += em
            total_f1 += f1
            count += 1
            if progress is not None:
                progress.update(1)
            if args.print_every > 0 and count % args.print_every == 0:
                printed += 1
                print(f"\n[{count}] id={s['id']}")
                print(f"question={s['question']}")
                if args.print_prompt:
                    print("prompt=")
                    print(prompt)
                print(f"prediction={pred}")
                print(f"gold_answers={s['gold_answer_texts']}")
                print(f"em={em:.4f} f1={f1:.4f}")
                if args.print_limit is not None and printed >= args.print_limit:
                    print("Print limit reached; suppressing further sample logs.")
                    args.print_every = 0

    if progress is not None:
        progress.close()

    avg_em = total_em / count if count else 0.0
    avg_f1 = total_f1 / count if count else 0.0
    write_summary_csv(
        summary_path,
        dataset=dataset,
        split=args.split,
        method="llm_only",
        num_samples=count,
        em=avg_em,
        f1=avg_f1,
    )

    print(f"Wrote {count} samples to {out_path}")
    print(f"Summary: EM={avg_em:.4f} F1={avg_f1:.4f} -> {summary_path}")
    return 0


def cmd_phase2(args: argparse.Namespace) -> int:
    samples = load_gnnrag_split(args.data_dir, args.split, limit=args.limit)
    if args.progress:
        print(f"Loaded {len(samples)} samples, initializing reader={args.reader}...", flush=True)
    reader = build_reader(
        args.reader,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        chat_template=args.chat_template,
    )
    idmap = IDMap.from_dir(args.data_dir)
    entity_emb, relation_emb, word_emb, vocab = load_embeddings(args.data_dir)

    dataset = args.dataset or _infer_dataset_name(args.data_dir)
    out_dir = Path(args.output_dir) / dataset / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "subgraph_only.jsonl"
    summary_path = out_dir / "summary.csv"

    total_em = 0.0
    total_f1 = 0.0
    count = 0

    progress = _build_progress(len(samples), args.progress, args.progress_every)
    printed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            q_vec = compute_question_vec(s["question"], vocab, word_emb)
            evidence = retrieve_subgraph(
                s,
                idmap,
                q_vec,
                entity_emb,
                relation_emb,
                topn=args.topn,
            )
            prompt = build_prompt(s["question"], evidence)
            pred = reader.generate(prompt)
            em, f1 = compute_em_f1(pred, s["gold_answer_texts"])

            record = {
                "id": s["id"],
                "question": s["question"],
                "prediction": pred,
                "gold_answers": s["gold_answer_texts"],
                "em": em,
                "f1": f1,
            }
            if args.save_prompt:
                record["prompt"] = prompt
            if args.save_evidence:
                record["evidence"] = evidence
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_em += em
            total_f1 += f1
            count += 1
            if progress is not None:
                progress.update(1)
            if args.print_every > 0 and count % args.print_every == 0:
                printed += 1
                print(f"\n[{count}] id={s['id']}")
                print(f"question={s['question']}")
                if args.print_prompt:
                    print("prompt=")
                    print(prompt)
                print(f"prediction={pred}")
                print(f"gold_answers={s['gold_answer_texts']}")
                print(f"em={em:.4f} f1={f1:.4f}")
                if args.print_limit is not None and printed >= args.print_limit:
                    print("Print limit reached; suppressing further sample logs.")
                    args.print_every = 0

    if progress is not None:
        progress.close()

    avg_em = total_em / count if count else 0.0
    avg_f1 = total_f1 / count if count else 0.0
    write_summary_csv(
        summary_path,
        dataset=dataset,
        split=args.split,
        method="subgraph_only",
        num_samples=count,
        em=avg_em,
        f1=avg_f1,
    )

    print(f"Wrote {count} samples to {out_path}")
    print(f"Summary: EM={avg_em:.4f} F1={avg_f1:.4f} -> {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PHR-QA experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p0 = sub.add_parser("phase0", help="Load data and print sample info")
    p0.add_argument("--data_dir", required=True, help="Dataset directory containing JSONL and id maps")
    p0.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p0.add_argument("--limit", type=int, default=10)
    p0.set_defaults(func=cmd_phase0)

    p1 = sub.add_parser("phase1", help="LLM-only baseline with EM/F1 eval")
    p1.add_argument("--data_dir", required=True, help="Dataset directory containing JSONL and id maps")
    p1.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p1.add_argument("--limit", type=int, default=None)
    p1.add_argument("--dataset", default=None, help="Dataset name override (default: data_dir basename)")
    p1.add_argument("--output_dir", default="outputs", help="Output root directory")
    p1.add_argument(
        "--reader",
        default="dummy",
        choices=["dummy", "echo", "hf"],
        help="Reader backend (dummy returns empty answers)",
    )
    p1.add_argument("--model_path", default=None, help="Local HF model path (required for reader=hf)")
    p1.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p1.add_argument("--dtype", default="auto", help="auto|float16|bfloat16|float32")
    p1.add_argument("--max_new_tokens", type=int, default=128)
    p1.add_argument("--temperature", type=float, default=0.0)
    p1.add_argument("--top_p", type=float, default=1.0)
    p1.add_argument("--chat_template", default="auto", help="auto|on|off")
    p1.add_argument("--progress", action="store_true", help="Show progress bar for inference loop")
    p1.add_argument("--progress_every", type=int, default=10, help="Fallback progress print interval")
    p1.add_argument("--save_prompt", action="store_true", help="Write prompt into output jsonl")
    p1.add_argument("--print_every", type=int, default=0, help="Print every N samples to stdout")
    p1.add_argument("--print_limit", type=int, default=None, help="Max number of printed samples")
    p1.add_argument("--print_prompt", action="store_true", help="Include prompt in printed logs")
    p1.set_defaults(func=cmd_phase1)

    p2 = sub.add_parser("phase2", help="Subgraph-only baseline with semantic scoring")
    p2.add_argument("--data_dir", required=True, help="Dataset directory containing JSONL and id maps")
    p2.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p2.add_argument("--limit", type=int, default=None)
    p2.add_argument("--dataset", default=None, help="Dataset name override (default: data_dir basename)")
    p2.add_argument("--output_dir", default="outputs", help="Output root directory")
    p2.add_argument(
        "--reader",
        default="dummy",
        choices=["dummy", "echo", "hf"],
        help="Reader backend (dummy returns empty answers)",
    )
    p2.add_argument("--model_path", default=None, help="Local HF model path (required for reader=hf)")
    p2.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p2.add_argument("--dtype", default="auto", help="auto|float16|bfloat16|float32")
    p2.add_argument("--max_new_tokens", type=int, default=128)
    p2.add_argument("--temperature", type=float, default=0.0)
    p2.add_argument("--top_p", type=float, default=1.0)
    p2.add_argument("--chat_template", default="auto", help="auto|on|off")
    p2.add_argument("--progress", action="store_true", help="Show progress bar for inference loop")
    p2.add_argument("--progress_every", type=int, default=10, help="Fallback progress print interval")
    p2.add_argument("--save_prompt", action="store_true", help="Write prompt into output jsonl")
    p2.add_argument("--save_evidence", action="store_true", help="Write evidence into output jsonl")
    p2.add_argument("--print_every", type=int, default=0, help="Print every N samples to stdout")
    p2.add_argument("--print_limit", type=int, default=None, help="Max number of printed samples")
    p2.add_argument("--print_prompt", action="store_true", help="Include prompt in printed logs")
    p2.add_argument("--topn", type=int, default=50, help="Top-N evidence triples")
    p2.set_defaults(func=cmd_phase2)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
