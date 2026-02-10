fgimport argparse
from typing import List

from dataloader import load_gnnrag_split
from idmap import IDMap


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PHR-QA experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p0 = sub.add_parser("phase0", help="Load data and print sample info")
    p0.add_argument("--data_dir", required=True, help="Dataset directory containing JSONL and id maps")
    p0.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p0.add_argument("--limit", type=int, default=10)
    p0.set_defaults(func=cmd_phase0)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

