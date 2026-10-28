import argparse
import json
import random
from pathlib import Path

from scripts.perturbator import SimpleNPFilter

SEED = 42


def collect_orders(pairs):
    dem_num, dem_adj, num_adj = [], [], []

    for pair in pairs:
        if "Dem-Num" in list(pair.keys())[0]:
            dem_num.append(pair)
        elif "Dem-Adj" in list(pair.keys())[0]:
            dem_adj.append(pair)
        elif "Num-Adj" in list(pair.keys())[0]:
            num_adj.append(pair)

    return dem_num, dem_adj, num_adj


def main():
    parser = argparse.ArgumentParser(
        description="Create perturbed corpus splits and minimal-pair evaluations."
    )
    parser.add_argument(
        "pickled_file", type=Path, help="Serialized Stanza corpus to perturb."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for generated corpus files and evaluation JSONL files.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    output_files = {
        "accept": output_dir / "dummy_accept_out.txt",
        "train": output_dir / "train.txt",
        "val": output_dir / "val.txt",
        "test": output_dir / "test.txt",
        "rejected": output_dir / "dummy_rejected.txt",
    }
    data_file = output_dir / "perturb_wiki_corpus_data.txt"
    minimal_pair_file = output_dir / "minimal_pairs.jsonl"
    minimal_3_pair_file = output_dir / "minimal_3_pairs.jsonl"

    with SimpleNPFilter(
        f_in=args.pickled_file,
        f_accept_out_path=output_files["accept"],
        f_accept_out_train=output_files["train"],
        f_accept_out_val=output_files["val"],
        f_accept_out_test=output_files["test"],
        f_reject_out=output_files["rejected"],
    ) as perturbator:
        perturbator.filter_write()

        with data_file.open("w", encoding="utf-8") as file:
            file.write("===== Perturbation Stats =====\n")
            file.write(f"Total sentences processed: {perturbator.total_sentences}\n")
            file.write(f"Sentences with complex NPs: {perturbator.complex_sentences}\n")
            file.write(f"Sentences with only simple NPs: {perturbator.simple_sentences}\n")
            file.write(
                f"Total perturbed sentences created: {perturbator.perturbed_sentences}\n"
            )
            file.write(f"Sentences with CITO: {perturbator.cito_sentences}\n")

            for sentence_group in perturbator.perturbed_sentence_txt:
                for sentence in sentence_group:
                    file.write(sentence + "\n")

        train_pairs_2 = [
            pair for pair in perturbator.minimal_pairs if pair.get("split") == "train"
        ]
        test_pairs_2 = [
            pair for pair in perturbator.minimal_pairs if pair.get("split") == "test"
        ]
        train_pairs_3 = [
            pair
            for pair in perturbator.minimal_3_pairs
            if pair.get("split") == "train"
        ]
        test_pairs_3 = [
            pair for pair in perturbator.minimal_3_pairs if pair.get("split") == "test"
        ]

        train_dem_num, train_dem_adj, train_num_adj = collect_orders(train_pairs_2)
        test_dem_num, test_dem_adj, test_num_adj = collect_orders(test_pairs_2)
        sample_size = 6
        combined_pairs = (
            random.sample(train_dem_num, min(len(train_dem_num), sample_size))
            + random.sample(train_dem_adj, min(len(train_dem_adj), sample_size))
            + random.sample(train_num_adj, min(len(train_num_adj), sample_size))
            + random.sample(test_dem_num, min(len(test_dem_num), sample_size))
            + random.sample(test_dem_adj, min(len(test_dem_adj), sample_size))
            + random.sample(test_num_adj, min(len(test_num_adj), sample_size))
        )

        sample_size_3 = 5
        combined_pairs_3 = random.sample(
            train_pairs_3, min(len(train_pairs_3), sample_size_3)
        ) + random.sample(test_pairs_3, min(len(test_pairs_3), sample_size_3))

    with minimal_pair_file.open("w", encoding="utf-8") as file:
        for pair in combined_pairs:
            file.write(json.dumps(pair) + "\n")

    with minimal_3_pair_file.open("w", encoding="utf-8") as file:
        for pair in combined_pairs_3:
            file.write(json.dumps(pair) + "\n")


if __name__ == "__main__":
    main()