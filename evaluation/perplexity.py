"""Evaluate checkpoint perplexity on minimal pairs or the generated test corpus.
Run with:
python evaluation/perplexity.py eval <MODEL_CHECKPOINT_PATH_1> <MODEL_CHECKPOINT_PATH_2> ...
or
python evaluation/perplexity.py test <MODEL_CHECKPOINT_PATH_1> <MODEL_CHECKPOINT_PATH_2> ...
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from minicons import scorer
from scipy.stats import ttest_rel
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]


def get_model_size(path):
    path = path.lower()
    for size in ("110", "52", "350"):
        if size in path:
            return size
    return "unknown"


def get_perplexity(paths, sentences, skip_long_sentences=False):
    results = defaultdict(list)
    for path in paths:
        print(f"\nLoading model: {path}")
        model = scorer.IncrementalLMScorer(path, device="cuda")
        model_size = get_model_size(path)

        for index, sentence in enumerate(
            tqdm(sentences, desc=f"Scoring {model_size}")
        ):
            tokens = model.tokenizer.tokenize(sentence)
            if not tokens:
                print(f"Skipping sentence {index} (0 tokens)")
                continue
            if skip_long_sentences and len(tokens) > 500:
                print(f"Skipping sentence {index} ({len(tokens)} tokens)")
                continue

            log_prob = model.sequence_score(
                [sentence], reduction=lambda values: values.sum().item()
            )[0]
            results[model_size].append(np.exp(-log_prob / len(tokens)))
    return results


def perplexity_ttest(results):
    sizes = sorted(results, key=int)
    test_results = {}
    for index, size1 in enumerate(sizes):
        for size2 in sizes[index + 1 :]:
            t_stat, p_value = ttest_rel(
                np.log(results[size1]), np.log(results[size2])
            )
            test_results[f"{size1}_vs_{size2}"] = {
                "group1": size1,
                "group2": size2,
                "mean_ppl1": float(np.mean(results[size1])),
                "mean_ppl2": float(np.mean(results[size2])),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
            }
    return test_results


def load_sentences(mode):
    if mode == "test":
        test_path = ROOT_DIR / "outputs/test.txt"
        with test_path.open(encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]

    eval_path = ROOT_DIR / "outputs/eval.jsonl"
    with eval_path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    good = [record[next(iter(record))] for record in records]
    bad = [record[list(record)[1]] for record in records]
    return {"good": good, "bad": bad}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("eval", "test"),
        help="Score minimal-pair eval sentences or the generated test corpus.",
    )
    parser.add_argument(
        "model_paths",
        nargs="+",
        type=Path,
        help="Paths to trained model checkpoint directories.",
    )
    args = parser.parse_args()

    output_dir = ROOT_DIR / "evaluation/data"
    output_dir.mkdir(parents=True, exist_ok=True)
    sentences = load_sentences(args.mode)

    if args.mode == "test":
        results = get_perplexity(args.model_paths, sentences, skip_long_sentences=True)
        output_path = output_dir / "perplexity_test_results.txt"
        sections = [("Pairwise T-tests for Test Set", results)]
    else:
        output_path = output_dir / "perplexity_eval_results.txt"
        sections = [
            (
                "Pairwise T-Tests for Eval Set Good Sentences",
                get_perplexity(args.model_paths, sentences["good"]),
            ),
            (
                "Pairwise T-Tests for Eval Set Bad Sentences",
                get_perplexity(args.model_paths, sentences["bad"]),
            ),
        ]

    with output_path.open("w", encoding="utf-8") as file:
        for index, (title, results) in enumerate(sections):
            if index:
                file.write("\n\n")
            file.write(title + "\n")
            file.write(json.dumps(perplexity_ttest(results), indent=2))

    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()