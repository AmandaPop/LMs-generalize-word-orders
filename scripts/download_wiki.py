"""
Script for downloading the wiki corpus up to X number of tokens,
writing the entire corpus as a single line.
To run: change TARGET_TOKENS if needed, then run:
    python scripts/download_wiki.py
"""
import os
import re
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

TARGET_TOKENS = 300_000
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTFILE = "wikipedia_corpus.txt"
BUFFER_SIZE = 10_000
SEED = 42

#paths
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
outfile_path = OUTPUT_DIR / OUTFILE

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def split_sentences(text):
    return SENT_SPLIT.split(text)

def count_tokens(sent):
    return len(tokenizer(sent, add_special_tokens=False)["input_ids"])

#stream and shuffle 
wiki = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    streaming=True
)["train"].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)

token_count = 0

with open(outfile_path, "w", encoding="utf-8") as f:
    for entry in wiki:
        text = entry.get("text", "")
        if not text:
            continue

        for sent in split_sentences(text):
            sent = sent.strip()
            if not sent:
                continue

            n_tokens = count_tokens(sent)

            if token_count + n_tokens > TARGET_TOKENS:
                print(f"Reached token budget: {token_count:,}")
                break

            f.write(sent + " ")
            token_count += n_tokens

        if token_count >= TARGET_TOKENS:
            break

print(f"Done. Wrote {token_count:,} tokens to {outfile_path}")
