import json
import random
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT_DIR / "outputs/minimal_pairs.jsonl"
OUTPUT_FILE = ROOT_DIR / "outputs/eval.jsonl"

random.seed(42) 

# Nested buckets: buckets[type][split] -> list of lines
buckets = defaultdict(lambda: defaultdict(list))

# Read and group
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        
        type_key = next(iter(obj))  # e.g., "Dem-Num"
        split = obj.get("split")    # "train" or "test"
        
        if type_key in {"Dem-Num", "Num-Adj", "Dem-Adj"} and split in {"train", "test"}:
            buckets[type_key][split].append(line)

# Sample and write
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for type_key in ["Dem-Num", "Num-Adj", "Dem-Adj"]:
        for split in ["train", "test"]:
            lines = buckets[type_key][split]
            
            if len(lines) < 25:
                raise ValueError(f"Not enough {split} samples for {type_key}: {len(lines)}")
            
            sampled = random.sample(lines, 25)
            out.writelines(sampled)

print("Done! Wrote eval.jsonl")