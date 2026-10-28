"""
Script to compute statistics from the eval_deltas.py CSV output.
"""

import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "evaluation/data"
CSV_PATH = OUT_DIR / "eval_results.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def add_stat(name, stat, p, rows):
    rows.append({
        "section": "ttest",
        "test": name,
        "stat": stat,
        "p_value": p
    })

def add_stats(groups, grouping):
    stat_rows = []
    for g, group in groups.groupby(grouping):
        stat, p = ttest_1samp(group['ctx_delta'], 0)
        add_stat(f"{grouping} {g} (ctx_delta)", stat, p, stat_rows)

    for g, group in groups.groupby(grouping):
        stat, p = ttest_1samp(group['iso_delta'], 0)
        add_stat(f"{grouping} {g} (iso_delta)", stat, p, stat_rows)

    for g, group in groups.groupby(grouping):
        stat, p = ttest_1samp(group['ctx_accuracy'], 0.5)
        add_stat(f"{grouping} {g} (ctx_accuracy)", stat, p, stat_rows)

    for g, group in groups.groupby(grouping):
        stat, p = ttest_1samp(group['iso_accuracy'], 0.5)
        add_stat(f"{grouping} {g} (iso_accuracy)", stat, p, stat_rows)

    stats_df = pd.DataFrame(stat_rows)
    return stats_df

def compute_stats_dataset(df, model_size):
    # collapse per item × seed
    item_seed = df.groupby(['item', 'model_num']).agg(
        ctx_delta=('ctx_delta', 'mean'),
        iso_delta=('iso_delta', 'mean')
    ).reset_index()

    dataset_level = item_seed.groupby('item').agg(
        ctx_delta=('ctx_delta', 'mean'),
        iso_delta=('iso_delta', 'mean'),
        sd_ctx_delta=('ctx_delta', 'std'),
        sd_iso_delta=('iso_delta', 'std'),
        n=('ctx_delta', 'count')
    ).reset_index()

    dataset_seed = item_seed.groupby('model_num').agg(
        ctx_delta=('ctx_delta', 'mean'),
        iso_delta=('iso_delta', 'mean')
    ).reset_index()

    seed_accuracy = item_seed.groupby('model_num').agg(
        ctx_accuracy=('ctx_delta', lambda x: (x > 0).mean()),
        iso_accuracy=('iso_delta', lambda x: (x > 0).mean())
    ).reset_index()

    #summary stats
    dataset_summary = pd.DataFrame([{
        "section": "summary",
        "model_size": model_size,
        "ctx_delta": dataset_level['ctx_delta'].mean(),
        "iso_delta": dataset_level['iso_delta'].mean(),
        "ctx_accuracy": (dataset_level['ctx_delta'] > 0).mean(),
        "iso_accuracy": (dataset_level['iso_delta'] > 0).mean(),
        "num_items": len(dataset_level),

        "se_ctx_delta": dataset_seed['ctx_delta'].sem(),
        "se_iso_delta": dataset_seed['iso_delta'].sem(),
        "se_ctx_accuracy": seed_accuracy['ctx_accuracy'].sem(),
        "se_iso_accuracy": seed_accuracy['iso_accuracy'].sem(),
    }])

    #add section for the stats that are run
    stat_rows = []
    #calculate and record all the t-tests
    stat, p = ttest_1samp(dataset_seed['ctx_delta'], 0)
    add_stat("ctx_delta", stat, p, stat_rows)

    stat, p = ttest_1samp(dataset_seed['iso_delta'], 0)
    add_stat("iso_delta", stat, p, stat_rows)

    stat, p = ttest_1samp(seed_accuracy['ctx_accuracy'], 0.5)
    add_stat("ctx_accuracy", stat, p, stat_rows)

    stat, p = ttest_1samp(seed_accuracy['iso_accuracy'], 0.5)
    add_stat("iso_accuracy", stat, p, stat_rows)

    stats_df = pd.DataFrame(stat_rows)

    final_df = pd.concat([dataset_summary, stats_df], ignore_index=True)

    return final_df

def compute_stats_type(df, model_size):
    #collapse per item × seed, keeping type
    item_seed = df.groupby(['type', 'item', 'model_num']).agg(
    ctx_delta=('ctx_delta', 'mean'),
    iso_delta=('iso_delta', 'mean')
    ).reset_index()

    #average across seeds within item and type
    item_level = item_seed.groupby(['type', 'item']).agg(
    ctx_delta=('ctx_delta', 'mean'),
    iso_delta=('iso_delta', 'mean')
    ).reset_index()

    #equivalent to the ICS-% score
    type_accuracy = item_level.groupby('type').agg(
    ctx_accuracy=('ctx_delta', lambda x: (x > 0).mean()),
    iso_accuracy=('iso_delta', lambda x: (x > 0).mean()),
    num_items=('item', 'count')
    ).reset_index()

    #to be used to calculate the SE accuracy across seeds within type
    type_seed_accuracy = item_seed.groupby(['type', 'model_num']).agg(
    ctx_accuracy=('ctx_delta', lambda x: (x > 0).mean()),
    iso_accuracy=('iso_delta', lambda x: (x > 0).mean())
    ).reset_index()
    #now compute SE
    type_acc_se = type_seed_accuracy.groupby('type').agg(
    se_ctx_accuracy=('ctx_accuracy', 'sem'),
    se_iso_accuracy=('iso_accuracy', 'sem')
    ).reset_index()

    #collapse item but keep type and seed for averaging across items within type and seed
    type_seed = item_seed.groupby(['type', 'model_num']).agg(
        ctx_delta=('ctx_delta', 'mean'),
        iso_delta=('iso_delta', 'mean'),
        ctx_accuracy=('ctx_delta', lambda x: (x > 0).mean()),
        iso_accuracy=('iso_delta', lambda x: (x > 0).mean())
    ).reset_index()

    #compute stats within each type (over items)
    type_stats = type_seed.groupby('type').agg(
        avg_ctx_delta=('ctx_delta', 'mean'),
        sd_ctx_delta=('ctx_delta', 'std'),
        n=('ctx_delta', 'count'),
        avg_iso_delta=('iso_delta', 'mean'),
        sd_iso_delta=('iso_delta', 'std'),
        n_iso=('iso_delta', 'count'),
    ).reset_index()
    #get the standard error for each type
    type_stats['se_ctx_delta'] = type_stats['sd_ctx_delta'] / (type_stats['n'] ** 0.5)
    type_stats['se_iso_delta'] = type_stats['sd_iso_delta'] / (type_stats['n_iso'] ** 0.5)

    type_stats = type_stats.merge(type_accuracy, on='type')
    type_stats = type_stats.merge(type_acc_se, on='type')

    type_stats['section'] = 'summary'

    #run and add stats to df 
    stats_df = add_stats(type_seed, "type")
    final_df = pd.concat([type_stats, stats_df], ignore_index=True)
    
    return final_df

def compute_stats_split(df, model_size):
    

    #average per item (across models), keeping split
    item_seed = df.groupby(['split', 'item', 'model_num']).agg(
        ctx_delta=('ctx_delta', 'mean'),
        iso_delta=('iso_delta', 'mean')
    ).reset_index()

    item_level = item_seed.groupby(['split', 'item']).agg(
        ctx_delta=('ctx_delta', 'mean'),
        iso_delta=('iso_delta', 'mean')
    ).reset_index()

    #equivalent to the ICS-% score
    split_accuracy = item_level.groupby('split').agg(
        ctx_accuracy=('ctx_delta', lambda x: (x > 0).mean()),
        iso_accuracy=('iso_delta', lambda x: (x > 0).mean()),
        num_items=('item', 'count')
    ).reset_index()

    #for SE of accuracy across seeds within split
    split_seed_accuracy = item_seed.groupby(['split', 'model_num']).agg(
    ctx_accuracy=('ctx_delta', lambda x: (x > 0).mean()),
    iso_accuracy=('iso_delta', lambda x: (x > 0).mean())
    ).reset_index()
    split_acc_se = split_seed_accuracy.groupby('split').agg(
    se_ctx_accuracy=('ctx_accuracy', 'sem'),
    se_iso_accuracy=('iso_accuracy', 'sem')
    ).reset_index()
    

    #collapse item but keep split and seed for averaging across items within split and seed
    split_seed = item_seed.groupby(['split', 'model_num']).agg(
        ctx_delta=('ctx_delta', 'mean'),
        iso_delta=('iso_delta', 'mean'),
        ctx_accuracy=('ctx_delta', lambda x: (x > 0).mean()),
        iso_accuracy=('iso_delta', lambda x: (x > 0).mean())
    ).reset_index()

    #compute stats within each split (over items)
    split_stats = split_seed.groupby('split').agg(
        avg_ctx_delta=('ctx_delta', 'mean'),
        sd_ctx_delta=('ctx_delta', 'std'),
        n=('ctx_delta', 'count'),
        avg_iso_delta=('iso_delta', 'mean'),
        sd_iso_delta=('iso_delta', 'std'),
        n_iso=('iso_delta', 'count'),
    ).reset_index()
    #get the standard error for each split
    split_stats['se_ctx_delta'] = split_stats['sd_ctx_delta'] / (split_stats['n'] ** 0.5)
    split_stats['se_iso_delta'] = split_stats['sd_iso_delta'] / (split_stats['n_iso'] ** 0.5)

    split_stats = split_stats.merge(split_accuracy, on='split')
    split_stats = split_stats.merge(split_acc_se, on='split')

    split_stats['section'] = 'summary'

    #run and add stats to the df
    stats_df = add_stats(split_seed, "split")
    final_df = pd.concat([stats_df, split_stats], ignore_index=True)

    return final_df


def compute_item_level_by_size(df):
    #mean across seeds
    item_seed = df.groupby(['model_size', 'item', 'model_num']).agg(
        ctx_delta=('ctx_delta', 'mean'),
        iso_delta=('iso_delta', 'mean')
    ).reset_index()
    #mean across items within one size 
    item_level = item_seed.groupby(['model_size', 'item']).agg(
        ctx_delta=('ctx_delta', 'mean'),
        iso_delta=('iso_delta', 'mean'),
        ctx_accuracy=('ctx_delta', lambda x: (x > 0).mean()),
        iso_accuracy=('iso_delta', lambda x: (x > 0).mean())
    ).reset_index()

    return item_level


def main():
    #csv path needs to be the path to the CSV output from eval_deltas.py
    df = pd.read_csv(CSV_PATH)
    #split by model size and run all stats separately for each model size
    df_110 = df[df["model_size"] == 110] 
    df_52 = df[df["model_size"] == 52]
    df_350 = df[df["model_size"] == 350]
    dataset_summary_110 = compute_stats_dataset(df_110, "110")
    dataset_summary_110.to_csv(OUT_DIR / "110_dataset_summary.csv", index=False)

    type_stats_110 = compute_stats_type(df_110, "110")
    type_stats_110.to_csv(OUT_DIR / "110_type_summary.csv", index=False)

    split_stats_110 =compute_stats_split(df_110, "110")
    split_stats_110.to_csv(OUT_DIR / "110_split_summary.csv", index=False)

    dataset_summary_52 = compute_stats_dataset(df_52, "52")
    dataset_summary_52.to_csv(OUT_DIR / "52_dataset_summary.csv", index=False)

    type_stats_52 = compute_stats_type(df_52, "52")
    type_stats_52.to_csv(OUT_DIR / "52_type_summary.csv", index=False)

    split_stats_52 = compute_stats_split(df_52, "52")
    split_stats_52.to_csv(OUT_DIR / "52_split_summary.csv", index=False)

    dataset_summary_350 = compute_stats_dataset(df_350, "350")
    dataset_summary_350.to_csv(OUT_DIR / "350_dataset_summary.csv", index=False)

    type_stats_350 = compute_stats_type(df_350, "350")
    type_stats_350.to_csv(OUT_DIR / "350_type_summary.csv", index=False)

    split_stats_350 = compute_stats_split(df_350, "350")
    split_stats_350.to_csv(OUT_DIR / "350_split_summary.csv", index=False)

    #needed for plotting sizes together 
    model_size_accuracy = pd.concat(
        [dataset_summary_110, dataset_summary_52, dataset_summary_350], ignore_index=True
    )
    model_size_accuracy.to_csv(OUT_DIR / "model_size_accuracy.csv", index=False)


if __name__ == "__main__":
    main()