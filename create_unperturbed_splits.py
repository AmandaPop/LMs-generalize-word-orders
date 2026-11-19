import argparse
import random
import re
from pathlib import Path

SEED = 42
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def read_sentences(corpus_path: Path) -> list[str]:
    text = corpus_path.read_text(encoding="utf-8")
    return [sentence.strip() for sentence in SENTENCE_SPLIT.split(text) if sentence.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shuffle a raw corpus and split it into 80% train, 10% val, and 10% test."
    )
    parser.add_argument(
        "corpus_file",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "outputs" / "wikipedia_corpus.txt",
        help="Raw Wiki corpus text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "unperturbed_outputs",
        help="Directory for train.txt, val.txt, and test.txt.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.corpus_file.is_file():
        parser.error(f"Corpus file not found: {args.corpus_file}")

    sentences = read_sentences(args.corpus_file)
    if not sentences:
        parser.error("The corpus contains no sentences.")

    rng = random.Random(args.seed)

    splits = {"train": [], "val": [], "test": []}
    for sentence in sentences:
        rand_val = rng.random()
        if rand_val < 0.8:
            split_name = "train"
        elif rand_val < 0.9:
            split_name = "val"
        else:
            split_name = "test"

        splits[split_name].append(sentence)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_sentences in splits.items():
        output_path = args.output_dir / f"{split_name}.txt"
        output_path.write_text(
            "\n".join(split_sentences) + ("\n" if split_sentences else ""),
            encoding="utf-8",
        )

    print(f"Read {len(sentences)} sentences from {args.corpus_file}")
    print(f"Train: {len(splits['train'])}")
    print(f"Val: {len(splits['val'])}")
    print(f"Test: {len(splits['test'])}")
    print(f"Wrote splits to {args.output_dir}")


if __name__ == "__main__":
    main()
