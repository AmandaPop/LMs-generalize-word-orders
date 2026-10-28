"""Score one or more trained checkpoints passed on the command line."""

import json
import sys
import csv
from pathlib import Path
from minicons import scorer

MODEL_PATHS = sys.argv[1:]
ROOT_DIR = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT_DIR / "outputs/eval.jsonl"
CSV_OUT_FILE = ROOT_DIR / "evaluation/data/eval_results.csv"
TEST_PATH = ROOT_DIR / "outputs/test.txt"
CSV_OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def reduce_log_probs(log_probs):
    return log_probs.sum().item()

def get_model_size(path):
    p = path.lower()
    if "110" in p:
        return "110"
    if "52" in p:
        return "52"
    if "350" in p:
        return "350"

def get_deltas(model_paths, eval_file, csv_out_file):

    grouped = {}
    for p in model_paths:
        s = get_model_size(p)
        if s not in grouped:
            grouped[s] = []
        grouped[s].append(p)

    with open(csv_out_file, 'w', newline='') as csvfile:
        fieldnames = [
            'item', 'sentence_good', 'sentence_bad',
            'np_good', 'np_bad', 'type', 'split',
            'ctx_delta', 'iso_delta',
            'model_num', 'model_size', 'model_path'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_size, paths in grouped.items():

            for model_num, model in enumerate(paths):

                print(f"\nProcessing {model_size} MODEL {model_num}: {model}")

                lm = scorer.IncrementalLMScorer(
                    model,
                    device="cuda"
                    
                )

                with open(eval_file, "r") as f:
                    item_num = 0

                    for line in f:
                        obj = json.loads(line)

                        keys = list(obj.keys())
                        type_key = keys[0]
                        bad_key = keys[1]

                        split = obj["split"]

                        good_sent = obj[type_key]
                        bad_sent = obj[bad_key]

                        good_np = obj["np_good"]
                        bad_np = obj["np_bad"]

                        lp_good_ctx, lp_bad_ctx = lm.sequence_score(
                            [good_sent, bad_sent],
                            reduction=reduce_log_probs
                        )
                        delta_ctx = lp_good_ctx - lp_bad_ctx

                        lp_good_iso, lp_bad_iso = lm.sequence_score(
                            [good_np, bad_np],
                            reduction=reduce_log_probs
                        )
                        delta_iso = lp_good_iso - lp_bad_iso

                        writer.writerow({
                            'sentence_good': good_sent,
                            'sentence_bad': bad_sent,
                            'np_good': good_np,
                            'np_bad': bad_np,
                            'type': type_key,
                            'split': split,
                            'ctx_delta': delta_ctx,
                            'iso_delta': delta_iso,
                            'model_num': model_num,
                            'model_size': model_size,
                            'model_path': model,
                            'item': item_num
                        })

                        item_num += 1
                        

if __name__ == "__main__":
    get_deltas(MODEL_PATHS, EVAL_FILE, CSV_OUT_FILE)
