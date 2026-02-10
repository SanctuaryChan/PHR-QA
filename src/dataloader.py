import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


class DataFormatError(ValueError):
    pass


def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise DataFormatError(f"JSON parse error in {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise DataFormatError(
                    f"Expected a JSON object in {path}:{line_no}, got {type(obj).__name__}"
                )
            yield obj


def iter_jsonl_with_lineno(path: Union[str, Path]) -> Iterator[Tuple[int, Dict[str, Any]]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise DataFormatError(f"JSON parse error in {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise DataFormatError(
                    f"Expected a JSON object in {path}:{line_no}, got {type(obj).__name__}"
                )
            yield line_no, obj


def _find_split_file(data_dir: Union[str, Path], split: str) -> Path:
    data_dir = Path(data_dir)
    candidates = [data_dir / f"{split}.json", data_dir / f"{split}.jsonl"]
    for cand in candidates:
        if cand.exists():
            return cand
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Split file not found for split='{split}'. Tried: {tried}")


def _coerce_int(value: Any, *, ctx: str) -> int:
    if isinstance(value, bool):
        raise DataFormatError(f"Expected int for {ctx}, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    raise DataFormatError(f"Expected int-like value for {ctx}, got {value!r}")


def load_gnnrag_split(
    data_dir: Union[str, Path],
    split: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load WebQSP/CWQ in the "GNN-RAG" JSONL format described in docs/设计文档.md.

    Returns a list of internal Samples:
      {
        "id": str,
        "question": str,
        "topic_entities": list[int],
        "gold_answer_texts": list[str],
        "triples": list[dict(h=int, r=int, t=int)],
      }
    """
    split_path = _find_split_file(data_dir, split)
    samples: List[Dict[str, Any]] = []

    for line_no, obj in iter_jsonl_with_lineno(split_path):
        sample_id = obj.get("id")
        if sample_id is None:
            raise DataFormatError(f"Missing field 'id' in {split_path}:{line_no}")
        question = obj.get("question")
        if question is None:
            raise DataFormatError(
                f"Missing field 'question' for id={sample_id!r} in {split_path}:{line_no}"
            )

        entities = obj.get("entities", [])
        if entities is None:
            entities = []
        if not isinstance(entities, list):
            raise DataFormatError(
                f"Expected 'entities' to be a list for id={sample_id!r} in {split_path}:{line_no}, "
                f"got {type(entities).__name__}"
            )
        topic_entities = [_coerce_int(e, ctx=f"id={sample_id!r}.entities[]") for e in entities]

        gold_answer_texts: List[str] = []
        answers = obj.get("answers", [])
        if isinstance(answers, list):
            for ans in answers:
                if isinstance(ans, dict) and "text" in ans and ans["text"] is not None:
                    gold_answer_texts.append(str(ans["text"]))

        subgraph = obj.get("subgraph")
        if not isinstance(subgraph, dict):
            raise DataFormatError(
                f"Missing or invalid 'subgraph' for id={sample_id!r} in {split_path}:{line_no}"
            )
        tuples = subgraph.get("tuples")
        if not isinstance(tuples, list):
            raise DataFormatError(
                f"Missing or invalid 'subgraph.tuples' for id={sample_id!r} in {split_path}:{line_no}"
            )

        triples: List[Dict[str, int]] = []
        for idx, t in enumerate(tuples):
            if not (isinstance(t, list) and len(t) == 3):
                raise DataFormatError(
                    f"Expected triple [h,r,t] at id={sample_id!r} in {split_path}:{line_no} "
                    f"(subgraph.tuples[{idx}])"
                )
            h = _coerce_int(t[0], ctx=f"id={sample_id!r}.subgraph.tuples[{idx}][0]")
            r = _coerce_int(t[1], ctx=f"id={sample_id!r}.subgraph.tuples[{idx}][1]")
            tail = _coerce_int(t[2], ctx=f"id={sample_id!r}.subgraph.tuples[{idx}][2]")
            triples.append({"h": h, "r": r, "t": tail})

        samples.append(
            {
                "id": str(sample_id),
                "question": str(question),
                "topic_entities": topic_entities,
                "gold_answer_texts": gold_answer_texts,
                "triples": triples,
            }
        )

        if limit is not None and len(samples) >= limit:
            break

    return samples
