import csv
import re
import string
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em_f1(prediction: str, gold_answers: Iterable[str]) -> Tuple[float, float]:
    gold_list = list(gold_answers)
    if not gold_list:
        return 0.0, 0.0
    em = 0.0
    f1 = 0.0
    for gold in gold_list:
        em = max(em, exact_match_score(prediction, gold))
        f1 = max(f1, f1_score(prediction, gold))
    return em, f1


def write_summary_csv(
    path: Path,
    dataset: str,
    split: str,
    method: str,
    num_samples: int,
    em: float,
    f1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "split", "method", "num_samples", "em", "f1"])
        writer.writerow([dataset, split, method, num_samples, f"{em:.6f}", f"{f1:.6f}"])
