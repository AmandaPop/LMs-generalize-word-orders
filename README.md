# Corpus Perturbation Pipeline

This repository contains the code used to run the experiments and complete the data analysis for "[Language Models Generalize to Human-Like Word Order Preferences](https://arxiv.org/abs/2608.05028)" paper. The upstream `corpus-filtering` and `lm-training` repositories provide the filtering framework and model/tokenizer training code; this repository extends them with the perturbation-specific code here as well as includes the evaluation steps.

## Pipeline

1. Download a raw corpus sample.
2. Annotate it with Stanza and serialize the documents.
3. Create simple/complex corpus splits and minimal-pair evaluation data.
4. Train the word-level tokenizer using `lm-training`.
5. Train models using `lm-training`.
6. Score trained checkpoints with the evaluation scripts.

Generated corpora, tokenizers, checkpoints, logs, and evaluation tables are ignored by Git.

## Setup

The included `environment.yml` creates the GPU Conda environment for corpus preprocessing, perturbation, tokenizer/model training, and evaluation.

### 1. Clone the repository and submodules

```sh
git clone --recurse-submodules <your-repository-url>
cd <your-repository>
```

If the repository was cloned without `--recurse-submodules`, initialize the submodules with:

```sh
git submodule update --init --recursive
```

The submodules are checked out as `corpus-filtering/` and `lm-training/`. The environment file includes the dependencies used by both upstream repositories and by this pipeline.

### 2. Create and activate the environment

```sh
conda env create -f environment.yml
conda activate corpus-perturbation
```

Because `scripts/perturbator.py` imports the upstream filtering package, install that submodule into the active environment:

```sh
pip install -e corpus-filtering
```

## Run

Run commands from this repository's root. These commands first download the corpus, next serialize the corpus, finally do the work of creating the perturbed corpus. 

```sh
python scripts/download_wiki.py
python scripts/stanza_serialize.py w outputs/wikipedia_corpus.txt outputs/wiki_corpus.pkl -t
python perturb_corpus.py outputs/wiki_corpus.pkl
```

`perturb_corpus.py` writes `train.txt`, `val.txt`, `test.txt`, `minimal_pairs.jsonl`, and `minimal_3_pairs.jsonl` under `outputs/`. Create the final sampled evaluation set before scoring or just score on all evaluation data created:

```sh
python scripts/sample_eval.py
```

Train the tokenizer with `lm-training/src/train_tokenizer.py`, using `outputs/` as its `data.base_dir` and writing the tokenizer into `lm-training/models/tokenizer/`. Then train the desired model configurations with `lm-training/src/train_lm_filter.py`, pointing them at the generated corpus and tokenizer.

## Evaluation and Data Analysis

The perplexity script takes a mode and one or more checkpoint paths. Use `eval` for the sampled minimal-pair dataset or `test` for the generated test corpus:

```sh
python evaluation/eval_deltas.py <MODEL_CHECKPOINT_PATH_1> <MODEL_CHECKPOINT_PATH_2> ...
python evaluation/perplexity.py eval <MODEL_CHECKPOINT_PATH_1> <MODEL_CHECKPOINT_PATH_2> ...
python evaluation/perplexity.py test <MODEL_CHECKPOINT_PATH_1> <MODEL_CHECKPOINT_PATH_2> ...
```

Evaluation outputs are written under `evaluation/data/`.

After running `eval_deltas.py` and `stats_from_eval.py`, generate the plots with `plot.py`.

```sh
python evaluation/mem.py
```
This reads `evaluation/data/eval_results.csv` and writes `evaluation/data/mixed_effects_results.csv`.


## Important paths

- `scripts/`: corpus download, Stanza serialization, perturbation support, and evaluation sampling.
- `corpus-filtering/`: Git submodule containing the filtering package used by `perturbator.py`.
- `lm-training/`: Git submodule containing tokenizer and language-model training code.
- `outputs/`: generated corpus splits, minimal pairs, checkpoints, and run logs.
- `evaluation/`: model scoring scripts and generated evaluation tables.
