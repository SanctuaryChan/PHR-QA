import argparse
import json
from pathlib import Path
from typing import List, Optional

from cache import load_json, write_json
from dataloader import load_gnnrag_split
from embeddings import load_embeddings
from eval import compute_em_f1, write_summary_csv
from idmap import IDMap
from planner_llama import (
    extract_candidate_relations,
    fallback_paths,
    map_paths_to_ids,
    normalize_paths,
    plan_paths,
)
from prompt_builder import build_prompt
from reader_llama import build_reader
from retriever_hybrid import retrieve as retrieve_hybrid
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


def _resolve_path(value: str | None, data_dir: str) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return str(cwd_candidate)
    return str(Path(data_dir) / path)

def _iter_batches(items: List[dict], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield i, items[i : i + batch_size]


def _generate_batch(reader, prompts: List[str]) -> List[str]:
    if hasattr(reader, "generate_batch"):
        return reader.generate_batch(prompts)
    return [reader.generate(p) for p in prompts]


def _load_question_emb(path: str, count: int):
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("numpy is required to load question embeddings") from e

    emb = np.load(path, mmap_mode="r")
    if emb.shape[0] < count:
        raise ValueError(
            f"question_emb rows {emb.shape[0]} < samples {count}"
        )
    return emb


def cmd_phase1(args: argparse.Namespace) -> int:
    samples = load_gnnrag_split(args.data_dir, args.split, limit=args.limit)
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
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
        attn_implementation=args.attn_impl,
        device_map=args.device_map,
        max_memory=args.max_memory,
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
        for batch_start, batch in _iter_batches(samples, args.batch_size):
            prompts: List[str] = []
            for s in batch:
                evidence = retrieve_llm_only(s)
                prompt = build_prompt(s["question"], evidence)
                prompts.append(prompt)

            preds = _generate_batch(reader, prompts)
            if len(preds) != len(batch):
                raise RuntimeError("Batch size mismatch between prompts and predictions")

            for s, prompt, pred in zip(batch, prompts, preds):
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
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
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
        attn_implementation=args.attn_impl,
        device_map=args.device_map,
        max_memory=args.max_memory,
    )
    idmap = IDMap.from_dir(args.data_dir, name_map_path=args.entity_name_map)
    entity_emb, relation_emb, word_emb, vocab = load_embeddings(
        args.data_dir,
        entity_emb_file=args.entity_emb,
        relation_emb_file=args.relation_emb,
        word_emb_file=args.word_emb,
        vocab_file=args.vocab,
    )
    q_emb = None
    if args.question_emb is not None:
        q_path = _resolve_path(args.question_emb, args.data_dir)
        q_emb = _load_question_emb(q_path, len(samples))
        if q_emb.shape[1] != entity_emb.shape[1]:
            raise ValueError(
                f"question_emb dim {q_emb.shape[1]} != entity_emb dim {entity_emb.shape[1]}"
            )

    dataset = args.dataset or _infer_dataset_name(args.data_dir)
    out_dir = Path(args.output_dir) / dataset / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "subgraph_only.jsonl"
    summary_path = out_dir / "summary.csv"

    total_em = 0.0
    total_f1 = 0.0
    count = 0
    total_oob_skipped = 0

    progress = _build_progress(len(samples), args.progress, args.progress_every)
    printed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for batch_start, batch in _iter_batches(samples, args.batch_size):
            prompts: List[str] = []
            evidences: List[List[dict]] = []
            for j, s in enumerate(batch):
                if q_emb is not None:
                    q_vec = q_emb[batch_start + j]
                else:
                    q_vec = compute_question_vec(
                        s["question"], vocab, word_emb, target_dim=entity_emb.shape[1]
                    )
                evidence, oob_skipped = retrieve_subgraph(
                    s,
                    idmap,
                    q_vec,
                    entity_emb,
                    relation_emb,
                    topn=args.topn,
                    oob_policy=args.oob_policy,
                )
                total_oob_skipped += oob_skipped
                evidences.append(evidence)
                prompts.append(build_prompt(s["question"], evidence))

            preds = _generate_batch(reader, prompts)
            if len(preds) != len(batch):
                raise RuntimeError("Batch size mismatch between prompts and predictions")

            for s, prompt, evidence, pred in zip(batch, prompts, evidences, preds):
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
    if total_oob_skipped > 0:
        print(f"Skipped {total_oob_skipped} triples due to out-of-range embeddings")
    print(f"Summary: EM={avg_em:.4f} F1={avg_f1:.4f} -> {summary_path}")
    return 0


def cmd_phase3(args: argparse.Namespace) -> int:
    samples = load_gnnrag_split(args.data_dir, args.split, limit=args.limit)
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
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
        attn_implementation=args.attn_impl,
        device_map=args.device_map,
        max_memory=args.max_memory,
    )

    planner_reader = None
    if args.planner == "hf":
        planner_model_path = args.planner_model_path or args.model_path
        if planner_model_path is None:
            raise ValueError("planner_model_path is required when planner=hf")
        if args.planner_separate or planner_model_path != args.model_path:
            planner_reader = build_reader(
                "hf",
                model_path=planner_model_path,
                device=args.device,
                dtype=args.dtype,
                max_new_tokens=args.planner_max_new_tokens,
                temperature=args.planner_temperature,
                top_p=args.planner_top_p,
                chat_template=args.chat_template,
                attn_implementation=args.attn_impl,
                device_map=args.device_map,
                max_memory=args.max_memory,
            )
        else:
            planner_reader = reader
    elif args.planner == "dummy":
        planner_reader = None
    else:
        raise ValueError(f"Unsupported planner: {args.planner}")

    idmap = IDMap.from_dir(args.data_dir, name_map_path=args.entity_name_map)
    relation_to_id = {name: idx for idx, name in enumerate(idmap.relations)}

    entity_emb, relation_emb, word_emb, vocab = load_embeddings(
        args.data_dir,
        entity_emb_file=args.entity_emb,
        relation_emb_file=args.relation_emb,
        word_emb_file=args.word_emb,
        vocab_file=args.vocab,
    )
    q_emb = None
    if args.question_emb is not None:
        q_path = _resolve_path(args.question_emb, args.data_dir)
        q_emb = _load_question_emb(q_path, len(samples))
        if q_emb.shape[1] != entity_emb.shape[1]:
            raise ValueError(
                f"question_emb dim {q_emb.shape[1]} != entity_emb dim {entity_emb.shape[1]}"
            )

    dataset = args.dataset or _infer_dataset_name(args.data_dir)
    out_dir = Path(args.output_dir) / dataset / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hybrid.jsonl"
    summary_path = out_dir / "summary.csv"

    total_em = 0.0
    total_f1 = 0.0
    count = 0
    total_oob_skipped = 0

    cache_root = Path(args.planner_cache_dir) if args.planner_cache_dir else None

    progress = _build_progress(len(samples), args.progress, args.progress_every)
    printed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for batch_start, batch in _iter_batches(samples, args.batch_size):
            prompts: List[str] = []
            evidences: List[List[dict]] = []
            paths_list: List[List[List[str]]] = []
            planner_raw_list: List[Optional[str]] = []

            for j, s in enumerate(batch):
                if q_emb is not None:
                    q_vec = q_emb[batch_start + j]
                else:
                    q_vec = compute_question_vec(
                        s["question"], vocab, word_emb, target_dim=entity_emb.shape[1]
                    )

                candidate_relations = extract_candidate_relations(
                    s, idmap, topm=args.planner_rel_topm
                )

                paths: List[List[str]] = []
                raw_plan: Optional[str] = None
                cache_path = None
                if cache_root is not None:
                    cache_path = cache_root / dataset / args.split / f"{s['id']}.json"
                    cached = load_json(cache_path)
                    if isinstance(cached, dict) and cached.get("paths"):
                        paths = normalize_paths(cached["paths"], args.path_max_len, args.path_topk)
                        raw_plan = cached.get("raw")

                if not paths:
                    if args.planner == "hf":
                        if planner_reader is None:
                            raise RuntimeError("planner_reader not initialized")
                        paths, raw_plan = plan_paths(
                            s["question"],
                            candidate_relations,
                            planner_reader,
                            args.path_topk,
                            args.path_max_len,
                        )
                    else:
                        paths = fallback_paths(
                            s["question"], candidate_relations, args.path_topk, args.path_max_len
                        )

                    if cache_path is not None:
                        write_json(
                            cache_path,
                            {
                                "id": s["id"],
                                "question": s["question"],
                                "paths": paths,
                                "raw": raw_plan,
                            },
                        )

                path_ids = map_paths_to_ids(paths, relation_to_id)
                evidence, oob_skipped = retrieve_hybrid(
                    s,
                    idmap,
                    q_vec,
                    entity_emb,
                    relation_emb,
                    path_ids,
                    topn=args.topn,
                    alpha=args.hybrid_alpha,
                    path_bonus=args.path_bonus,
                    direction=args.path_direction,
                    oob_policy=args.oob_policy,
                )
                total_oob_skipped += oob_skipped

                prompts.append(build_prompt(s["question"], evidence))
                evidences.append(evidence)
                paths_list.append(paths)
                planner_raw_list.append(raw_plan)

            preds = _generate_batch(reader, prompts)
            if len(preds) != len(batch):
                raise RuntimeError("Batch size mismatch between prompts and predictions")

            for s, prompt, evidence, pred, paths, raw_plan in zip(
                batch, prompts, evidences, preds, paths_list, planner_raw_list
            ):
                em, f1 = compute_em_f1(pred, s["gold_answer_texts"])
                record = {
                    "id": s["id"],
                    "question": s["question"],
                    "prediction": pred,
                    "gold_answers": s["gold_answer_texts"],
                    "em": em,
                    "f1": f1,
                    "paths": paths,
                }
                if args.save_prompt:
                    record["prompt"] = prompt
                if args.save_evidence:
                    record["evidence"] = evidence
                if args.save_planner_raw and raw_plan is not None:
                    record["planner_raw"] = raw_plan
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
        method="hybrid",
        num_samples=count,
        em=avg_em,
        f1=avg_f1,
    )

    print(f"Wrote {count} samples to {out_path}")
    if total_oob_skipped > 0:
        print(f"Skipped {total_oob_skipped} triples due to out-of-range embeddings")
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
    p1.add_argument("--chat_template", default="on", help="auto|on|off")
    p1.add_argument(
        "--attn_impl",
        default="auto",
        help="auto|flash_attention_2|sdpa|eager",
    )
    p1.add_argument(
        "--device_map",
        default=None,
        help="Device map for HF loading (auto|balanced|balanced_low_0|sequential|none)",
    )
    p1.add_argument(
        "--max_memory",
        default=None,
        help="Per-GPU max memory, e.g. '0:20GiB,1:20GiB'",
    )
    p1.add_argument("--progress", action="store_true", help="Show progress bar for inference loop")
    p1.add_argument("--progress_every", type=int, default=10, help="Fallback progress print interval")
    p1.add_argument("--save_prompt", action="store_true", help="Write prompt into output jsonl")
    p1.add_argument("--print_every", type=int, default=0, help="Print every N samples to stdout")
    p1.add_argument("--print_limit", type=int, default=None, help="Max number of printed samples")
    p1.add_argument("--print_prompt", action="store_true", help="Include prompt in printed logs")
    p1.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
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
    p2.add_argument("--chat_template", default="on", help="auto|on|off")
    p2.add_argument(
        "--attn_impl",
        default="auto",
        help="auto|flash_attention_2|sdpa|eager",
    )
    p2.add_argument(
        "--device_map",
        default=None,
        help="Device map for HF loading (auto|balanced|balanced_low_0|sequential|none)",
    )
    p2.add_argument(
        "--max_memory",
        default=None,
        help="Per-GPU max memory, e.g. '0:20GiB,1:20GiB'",
    )
    p2.add_argument("--progress", action="store_true", help="Show progress bar for inference loop")
    p2.add_argument("--progress_every", type=int, default=10, help="Fallback progress print interval")
    p2.add_argument("--save_prompt", action="store_true", help="Write prompt into output jsonl")
    p2.add_argument("--save_evidence", action="store_true", help="Write evidence into output jsonl")
    p2.add_argument("--print_every", type=int, default=0, help="Print every N samples to stdout")
    p2.add_argument("--print_limit", type=int, default=None, help="Max number of printed samples")
    p2.add_argument("--print_prompt", action="store_true", help="Include prompt in printed logs")
    p2.add_argument("--topn", type=int, default=50, help="Top-N evidence triples")
    p2.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    p2.add_argument(
        "--entity_name_map",
        default=None,
        help="JSON map of entity MID to readable name",
    )
    p2.add_argument("--entity_emb", default=None, help="Custom entity embedding filename/path")
    p2.add_argument("--relation_emb", default=None, help="Custom relation embedding filename/path")
    p2.add_argument("--word_emb", default=None, help="Custom word embedding filename/path")
    p2.add_argument("--vocab", default=None, help="Custom vocab filename/path")
    p2.add_argument("--question_emb", default=None, help="Precomputed question embedding .npy")
    p2.add_argument(
        "--oob_policy",
        default="skip",
        choices=["skip", "zero", "error"],
        help="Out-of-range embedding policy: skip|zero|error",
    )
    p2.set_defaults(func=cmd_phase2)

    p3 = sub.add_parser("phase3", help="Hybrid retriever with planner (paths + semantic)")
    p3.add_argument("--data_dir", required=True, help="Dataset directory containing JSONL and id maps")
    p3.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p3.add_argument("--limit", type=int, default=None)
    p3.add_argument("--dataset", default=None, help="Dataset name override (default: data_dir basename)")
    p3.add_argument("--output_dir", default="outputs", help="Output root directory")
    p3.add_argument(
        "--reader",
        default="dummy",
        choices=["dummy", "echo", "hf"],
        help="Reader backend (dummy returns empty answers)",
    )
    p3.add_argument("--model_path", default=None, help="Local HF model path (required for reader=hf)")
    p3.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p3.add_argument("--dtype", default="auto", help="auto|float16|bfloat16|float32")
    p3.add_argument("--max_new_tokens", type=int, default=128)
    p3.add_argument("--temperature", type=float, default=0.0)
    p3.add_argument("--top_p", type=float, default=1.0)
    p3.add_argument("--chat_template", default="on", help="auto|on|off")
    p3.add_argument(
        "--attn_impl",
        default="auto",
        help="auto|flash_attention_2|sdpa|eager",
    )
    p3.add_argument(
        "--device_map",
        default=None,
        help="Device map for HF loading (auto|balanced|balanced_low_0|sequential|none)",
    )
    p3.add_argument(
        "--max_memory",
        default=None,
        help="Per-GPU max memory, e.g. '0:20GiB,1:20GiB'",
    )
    p3.add_argument("--progress", action="store_true", help="Show progress bar for inference loop")
    p3.add_argument("--progress_every", type=int, default=10, help="Fallback progress print interval")
    p3.add_argument("--save_prompt", action="store_true", help="Write prompt into output jsonl")
    p3.add_argument("--save_evidence", action="store_true", help="Write evidence into output jsonl")
    p3.add_argument("--save_planner_raw", action="store_true", help="Write planner raw output into jsonl")
    p3.add_argument("--print_every", type=int, default=0, help="Print every N samples to stdout")
    p3.add_argument("--print_limit", type=int, default=None, help="Max number of printed samples")
    p3.add_argument("--print_prompt", action="store_true", help="Include prompt in printed logs")
    p3.add_argument("--topn", type=int, default=50, help="Top-N evidence triples")
    p3.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    p3.add_argument(
        "--entity_name_map",
        default=None,
        help="JSON map of entity MID to readable name",
    )
    p3.add_argument("--entity_emb", default=None, help="Custom entity embedding filename/path")
    p3.add_argument("--relation_emb", default=None, help="Custom relation embedding filename/path")
    p3.add_argument("--word_emb", default=None, help="Custom word embedding filename/path")
    p3.add_argument("--vocab", default=None, help="Custom vocab filename/path")
    p3.add_argument("--question_emb", default=None, help="Precomputed question embedding .npy")
    p3.add_argument(
        "--oob_policy",
        default="skip",
        choices=["skip", "zero", "error"],
        help="Out-of-range embedding policy: skip|zero|error",
    )
    p3.add_argument("--planner", default="hf", choices=["hf", "dummy"], help="Planner backend")
    p3.add_argument("--planner_model_path", default=None, help="Planner model path (defaults to reader model)")
    p3.add_argument("--planner_max_new_tokens", type=int, default=64)
    p3.add_argument("--planner_temperature", type=float, default=0.0)
    p3.add_argument("--planner_top_p", type=float, default=1.0)
    p3.add_argument("--planner_rel_topm", type=int, default=120, help="Top-M candidate relations by freq")
    p3.add_argument("--planner_cache_dir", default="cache/planner", help="Planner cache directory")
    p3.add_argument(
        "--planner_separate",
        action="store_true",
        help="Load a separate planner model instead of reusing reader",
    )
    p3.add_argument("--path_topk", type=int, default=5)
    p3.add_argument("--path_max_len", type=int, default=2)
    p3.add_argument("--path_direction", default="out", choices=["out", "in", "both"])
    p3.add_argument("--hybrid_alpha", type=float, default=0.7)
    p3.add_argument("--path_bonus", type=float, default=0.3)
    p3.set_defaults(func=cmd_phase3)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
