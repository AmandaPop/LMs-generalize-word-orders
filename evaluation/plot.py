import sys
import os
import pandas as pd
import plotnine as pn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "evaluation", "data")


def probability_plot(delta_file: str, output_dir: str, model_size: str = None) -> None:

    all_data = pd.read_csv(delta_file)
    #filter into model_size if needed
    model_size = int(model_size)
    if model_size:
        all_data = all_data[all_data["model_size"] == model_size]
        title_suffix = f" (Model Size {model_size})"
        filename = f"plot_{model_size}.png"
    else:
        title_suffix = ""
        filename = "plot.png"
    
    all_data_long = all_data.melt(
        id_vars=all_data.columns.difference(["iso_delta", "ctx_delta"]),
        value_vars=["iso_delta", "ctx_delta"],
        var_name="eval_type",
        value_name="delta",
    )
    all_data_long["eval_type"] = all_data_long["eval_type"].map(
        {"iso_delta": "iso", "ctx_delta": "ctx"}
    )
    all_data_long = all_data_long.dropna(subset=["delta"])

    summary = all_data_long.groupby(["type", "eval_type"], as_index=False).agg(
        mean_delta=("delta", "mean"), sd_delta=("delta", "std"), n=("delta", "count")
    )
    summary["se"] = summary["sd_delta"].fillna(0) / summary["n"].pow(0.5)
    summary["ci95"] = 1.96 * summary["se"]
    summary["ymin"] = summary["mean_delta"] - summary["ci95"]
    summary["ymax"] = summary["mean_delta"] + summary["ci95"]
    plot = (
        pn.ggplot(all_data_long, pn.aes(x="type", y="delta"))
        + pn.geom_violin(
            pn.aes(fill="eval_type"),
            color="grey",
            alpha=0.5,
            size=0.2,
            # draw_quantiles=(0.25, 0.5, 0.75),
        )
        + pn.geom_point(
            pn.aes(fill="eval_type"),
            alpha=0.2,
            stroke=0.0,
            position=pn.position_dodge(width=0.9),
        )
        + pn.geom_pointrange(
            summary,
            pn.aes(
                x="type",
                y="mean_delta",
                ymin="ymin",
                ymax="ymax",
                group="eval_type",
            ),
            color="black",
            shape="D",
            position=pn.position_dodge(width=0.9),
            fatten=1.5,
            alpha=0.7,
        )
        + pn.scale_fill_discrete(
            # values={"ctx": "orchid", "iso": "plum"},
            breaks=["ctx", "iso"],
            labels=["Context", "Isolation"],
            name="Evaluation",
        )
        # + pn.scale_color_manual( values={"ctx": "orchid", "iso": "plum"}, breaks=["ctx", "iso"], labels=["Context", "Isolation"], name="Evaluation",)
        # + pn.facet_wrap("~eval_type")
        + pn.labs(
            title=f"Probability Deltas by Phrase Type{title_suffix}",
            y="Log Probability Delta",
            x="",
        )
        + pn.theme_minimal()
    )
    plot.save(os.path.join(output_dir, filename))


def accuracy_plot(output_dir: str, model_size: str = None) -> None:

    if model_size:
        acc_file = os.path.join(DATA_DIR, f"{model_size}_type_summary.csv")
        title_suffix = f" (Model Size {model_size})"
        filename = f"type_level_results_{model_size}.png"
    else:
        acc_file = os.path.join(DATA_DIR, "type_stats.csv")
        title_suffix = ""
        filename = "type_level_results.png"

    data = pd.read_csv(acc_file)[
        ["type", "ctx_accuracy", "iso_accuracy", "se_ctx_accuracy", "se_iso_accuracy"]
    ]
    acc_long = data.melt(
        id_vars=["type"],
        value_vars=["ctx_accuracy", "iso_accuracy"],
        var_name="eval_type",
        value_name="accuracy",
    )
    acc_long["eval_type"] = acc_long["eval_type"].map(
        {"ctx_accuracy": "ctx", "iso_accuracy": "iso"}
    )

    se_long = data.melt(
        id_vars=["type"],
        value_vars=["se_ctx_accuracy", "se_iso_accuracy"],
        var_name="se_type",
        value_name="se",
    )
    se_long["eval_type"] = se_long["se_type"].map(
        {"se_ctx_accuracy": "ctx", "se_iso_accuracy": "iso"}
    )
    data_long = acc_long.merge(
        se_long[["type", "eval_type", "se"]], on=["type", "eval_type"]
    )
    data_long = data_long.dropna(subset=["accuracy", "se"])
    data_long["ci95"] = 1.96 * data_long["se"]
    data_long["ymin"] = data_long["accuracy"] - data_long["ci95"]
    data_long["ymax"] = data_long["accuracy"] + data_long["ci95"]

    dodge = pn.position_dodge(width=0.75)
    plot = (
        pn.ggplot(data_long, pn.aes(x="type", y="accuracy", fill="eval_type"))
        + pn.geom_col(position=dodge, width=0.65, alpha=0.5)
        + pn.geom_errorbar(
            pn.aes(ymin="ymin", ymax="ymax"),
            position=dodge,
            width=0.2,
        )
        + pn.geom_hline(yintercept=0.5, linetype="dashed", alpha=0.5)
        + pn.scale_fill_discrete(
            # values={"ctx": "orchid", "iso": "plum"},
            breaks=["ctx", "iso"],
            labels=["Context", "Isolation"],
            name="Evaluation",
        )
        + pn.labs(
            title=f"Homomorphic Choices by Phrase Type{title_suffix}",
            y="Proportion Scope-Homorphic Choices",
            x="",
        )
        + pn.theme_minimal()
    )
    plot.save(os.path.join(output_dir, filename))


def split_plot(output_dir: str, model_size: str = None) -> None:

    if model_size:
        split_file = os.path.join(DATA_DIR, f"{model_size}_split_summary.csv")
        filename = f"split_results_{model_size}.png"
    else:
        split_file = os.path.join(DATA_DIR, "split_stats.csv")
        filename = "split_results.png"

    data = pd.read_csv(split_file)[
        [
            "split",
            "avg_ctx_delta",
            "se_ctx_delta",
            "avg_iso_delta",
            "se_iso_delta",
            "ctx_accuracy",
            "iso_accuracy",
            "se_ctx_accuracy",
            "se_iso_accuracy",
        ]
    ]

    # Melt deltas
    delta_long = data[
        ["split", "avg_ctx_delta", "avg_iso_delta", "se_ctx_delta", "se_iso_delta"]
    ].melt(
        id_vars=["split"],
        value_vars=["avg_ctx_delta", "avg_iso_delta"],
        var_name="eval_type",
        value_name="value",
    )
    delta_long["eval_type"] = delta_long["eval_type"].map(
        {"avg_ctx_delta": "ctx", "avg_iso_delta": "iso"}
    )
    se_delta = data[["split", "se_ctx_delta", "se_iso_delta"]].melt(
        id_vars=["split"],
        value_vars=["se_ctx_delta", "se_iso_delta"],
        var_name="se_type",
        value_name="se",
    )
    se_delta["eval_type"] = se_delta["se_type"].map(
        {"se_ctx_delta": "ctx", "se_iso_delta": "iso"}
    )
    delta_long = delta_long.merge(
        se_delta[["split", "eval_type", "se"]], on=["split", "eval_type"]
    )
    delta_long["metric"] = "Probability Delta"

    # Melt accuracy
    acc_long = data[
        ["split", "ctx_accuracy", "iso_accuracy", "se_ctx_accuracy", "se_iso_accuracy"]
    ].melt(
        id_vars=["split"],
        value_vars=["ctx_accuracy", "iso_accuracy"],
        var_name="eval_type",
        value_name="value",
    )
    acc_long["eval_type"] = acc_long["eval_type"].map(
        {"ctx_accuracy": "ctx", "iso_accuracy": "iso"}
    )
    se_acc = data[["split", "se_ctx_accuracy", "se_iso_accuracy"]].melt(
        id_vars=["split"],
        value_vars=["se_ctx_accuracy", "se_iso_accuracy"],
        var_name="se_type",
        value_name="se",
    )
    se_acc["eval_type"] = se_acc["se_type"].map(
        {"se_ctx_accuracy": "ctx", "se_iso_accuracy": "iso"}
    )
    acc_long = acc_long.merge(
        se_acc[["split", "eval_type", "se"]], on=["split", "eval_type"]
    )
    acc_long["metric"] = "Accuracy"

    # Combine both
    data_long = pd.concat([delta_long, acc_long], ignore_index=True)
    data_long = data_long.dropna(subset=["value", "se"])
    data_long["ci95"] = 1.96 * data_long["se"]
    data_long["ymin"] = data_long["value"] - data_long["ci95"]
    data_long["ymax"] = data_long["value"] + data_long["ci95"]

    dodge = pn.position_dodge(width=0.75)
    plot = (
        pn.ggplot(data_long, pn.aes(x="split", y="value", fill="eval_type"))
        + pn.geom_col(position=dodge, width=0.65, alpha=0.5)
        + pn.geom_errorbar(
            pn.aes(ymin="ymin", ymax="ymax"),
            position=dodge,
            width=0.2,
        )
        + pn.geom_hline(yintercept=0, linetype="dashed", alpha=0.5)
        + pn.facet_wrap("~metric", scales="free_y")
        + pn.scale_fill_discrete(
            breaks=["ctx", "iso"],
            labels=["Context", "Isolation"],
            name="Evaluation",
        )
        + pn.labs(
            # title="Delta and Accuracy by Split",
            y="",
            x="",
        )
        + pn.theme_minimal()
    )
    plot.save(os.path.join(output_dir, filename))


def delta_violin_by_type_and_model_size(delta_file: str, output_dir: str, model_sizes: list[str] = None, bar_colors: list[str] = None) -> None:
    """
    Create a single violin plot of deltas by modifier type with different colors for each model size.
    
    Args:
        delta_file: Path to the CSV file with delta data
        output_dir: Directory to save the plot
        model_sizes: List of model sizes to include (e.g., ["52", "110", "350"])
    """
    if model_sizes is None:
        model_sizes = ["52", "110", "350"]
    # Default palette (override by passing bar_colors)
    if bar_colors is None:
        bar_colors = ["#4C78A8", "#ED0F2C", "#A24B9B"]

    if len(bar_colors) != len(model_sizes):
        raise ValueError("bar_colors must have the same length as model_sizes")
    
    all_data = pd.read_csv(delta_file)
    all_data["model_size"] = all_data["model_size"].astype(str)
    
    # Filter to selected model sizes
    all_data = all_data[all_data["model_size"].isin(model_sizes)]

    # If caller provided an explicit ordering for model sizes, force it
    if model_sizes is not None:
        all_data["model_size"] = pd.Categorical(
            all_data["model_size"], categories=[str(s) for s in model_sizes], ordered=True
        )
    
    # Melt to long format for both delta types
    all_data_long = all_data.melt(
        id_vars=all_data.columns.difference(["iso_delta", "ctx_delta"]),
        value_vars=["iso_delta", "ctx_delta"],
        var_name="eval_type",
        value_name="delta",
    )
    all_data_long["eval_type"] = all_data_long["eval_type"].map(
        {"iso_delta": "iso", "ctx_delta": "ctx"}
    )
    all_data_long = all_data_long.dropna(subset=["delta"])
    
    # Compute summary statistics for overlays
    summary = all_data_long.groupby(["type", "model_size", "eval_type"], as_index=False).agg(
        mean_delta=("delta", "mean"), sd_delta=("delta", "std"), n=("delta", "count")
    )
    summary["se"] = summary["sd_delta"].fillna(0) / summary["n"].pow(0.5)
    summary["ci95"] = 1.96 * summary["se"]
    summary["ymin"] = summary["mean_delta"] - summary["ci95"]
    summary["ymax"] = summary["mean_delta"] + summary["ci95"]
    
    plot = (
        pn.ggplot(all_data_long, pn.aes(x="type", y="delta", fill="model_size"))
        + pn.geom_violin(
            color="grey",
            alpha=0.5,
            size=0.2,
        )
        + pn.geom_point(
            alpha=0.2,
            stroke=0.0,
            position=pn.position_dodge(width=0.9),
        )
        + pn.labs(
            title="Delta Distribution by Phrase Type and Model Size",
            y="Log Probability Delta",
            x="Phrase Type",
        )
        + pn.theme_minimal()
    )

    # map model sizes to colors
    color_map = {str(size): color for size, color in zip(model_sizes, bar_colors)}
    plot = plot + pn.scale_fill_manual(values=color_map, name="Model Size")
    plot.save(os.path.join(output_dir, "delta_violin_by_type.png"))


def ctx_accuracy_by_type_and_model_size(
    output_dir: str,
    model_sizes: list[str] = None,
    bar_colors: list[str] = None,
) -> None:
    """
    Plot context accuracy by modifier type, with bars for each model size.

    Args:
        output_dir: Directory to save the plot
        model_sizes: List of model sizes to include (e.g., ["52", "110", "350"])
        bar_colors: Optional list of colors to use for each model size in the same order
    """
    if model_sizes is None:
        model_sizes = ["52", "110", "350"]
    # Default color palette (can be overridden by caller)
    if bar_colors is None:
        bar_colors = ["#4C78A8", "#ED0F2C", "#A24B9B"]

    if len(bar_colors) != len(model_sizes):
        raise ValueError("bar_colors must have the same length as model_sizes")

    # Read type summary files for each model size
    dfs = []
    for size in model_sizes:
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f"{size}_type_summary.csv"))
            df["model_size"] = str(size)
            dfs.append(df[["type", "ctx_accuracy", "se_ctx_accuracy", "model_size"]])
        except FileNotFoundError:
            continue

    if not dfs:
        print("No type summary files found")
        return

    data = pd.concat(dfs, ignore_index=True)
    data["model_size"] = data["model_size"].astype(str)
    if model_sizes is not None:
        data["model_size"] = pd.Categorical(
            data["model_size"], categories=[str(s) for s in model_sizes], ordered=True
        )
    data = data.rename(columns={"se_ctx_accuracy": "se"})
    data = data.dropna(subset=["ctx_accuracy", "se"])
    data["ci95"] = 1.96 * data["se"]
    data["ymin"] = data["ctx_accuracy"] - data["ci95"]
    data["ymax"] = data["ctx_accuracy"] + data["ci95"]

    dodge = pn.position_dodge(width=0.75)
    plot = (
        pn.ggplot(data, pn.aes(x="type", y="ctx_accuracy", fill="model_size"))
        + pn.geom_col(position=dodge, width=0.65, alpha=0.5)
        + pn.geom_errorbar(
            pn.aes(ymin="ymin", ymax="ymax"),
            position=dodge,
            width=0.2,
        )
        + pn.geom_hline(yintercept=0.5, linetype="dashed", alpha=0.5)
        + pn.labs(
            title="Accuracy by Phrase Type and Model Size",
            x="Phrase Type",
            y="Proportion Scope-Homomorphic Choices",
        )
        + pn.theme_minimal()
    )

    if bar_colors is None:
        plot = plot + pn.scale_fill_discrete(name="Model Size")
    else:
        color_map = {
            str(size): color for size, color in zip(model_sizes, bar_colors)
        }
        plot = plot + pn.scale_fill_manual(values=color_map, name="Model Size")

    plot.save(os.path.join(output_dir, "ctx_accuracy_by_type_and_model_size.png"))


def probability_violin_by_model_size(
    delta_file: str,
    output_dir: str,
    model_size_order: list[str] = None,
    bar_colors: list[str] = None,
) -> None:
    all_data = pd.read_csv(delta_file)
    all_data["model_size"] = all_data["model_size"].astype(str)

    if model_size_order is not None:
        all_data["model_size"] = pd.Categorical(
            all_data["model_size"], categories=model_size_order, ordered=True
        )

    all_data_long = all_data[["model_size", "ctx_delta"]].rename(
        columns={"ctx_delta": "delta"}
    )
    all_data_long = all_data_long.dropna(subset=["delta"])

    ordered_model_sizes = [
        str(size) for size in all_data_long["model_size"].dropna().unique()
    ]
    if model_size_order is not None:
        ordered_model_sizes = [
            size for size in model_size_order if size in ordered_model_sizes
        ] + [
            size for size in ordered_model_sizes if size not in model_size_order
        ]

    if bar_colors is None:
        bar_colors = ["#4C78A8", "#ED0F2C", "#A24B9B"]
    if len(bar_colors) != len(ordered_model_sizes):
        raise ValueError("bar_colors must have the same length as the number of model sizes")

    color_map = {
        str(size): color for size, color in zip(ordered_model_sizes, bar_colors)
    }

    summary = all_data_long.groupby(["model_size"], as_index=False).agg(
        mean_delta=("delta", "mean"),
        sd_delta=("delta", "std"),
        n=("delta", "count"),
    )
    summary["se"] = summary["sd_delta"].fillna(0) / summary["n"].pow(0.5)
    summary["ci95"] = 1.96 * summary["se"]
    summary["ymin"] = summary["mean_delta"] - summary["ci95"]
    summary["ymax"] = summary["mean_delta"] + summary["ci95"]

    plot = (
        pn.ggplot(all_data_long, pn.aes(x="model_size", y="delta", fill="model_size"))
        + pn.geom_violin(
            color="grey",
            alpha=0.5,
            size=0.2,
        )
        + pn.geom_point(
            alpha=0.2,
            stroke=0.0,
            position=pn.position_dodge(width=0.9),
        )
        + pn.geom_pointrange(
            summary,
            pn.aes(
                x="model_size",
                y="mean_delta",
                ymin="ymin",
                ymax="ymax",
            ),
            color="black",
            shape="D",
            position=pn.position_dodge(width=0.9),
            fatten=1.5,
            alpha=0.7,
        )
        + pn.scale_fill_manual(values=color_map, name="Model Size")
        + pn.labs(
            title="Delta Distribution by Model Size",
            x="Model Size",
            y="Log Probability Delta",
        )
        + pn.theme_minimal()
    )
    plot.save(os.path.join(output_dir, "probability_violin_by_model_size.png"))

def plot_accuracy_by_model_size(
    data_file: str,
    output_dir: str,
) -> None:
    data = pd.read_csv(data_file)
    data["model_size"] = pd.Categorical(
    data["model_size"].astype(str),
    categories=["52", "110", "350"],
    ordered=True,
    )

    data = data.sort_values("model_size")
    # Order model sizes from smallest to largest
    model_size_order = ["52", "110", "350"]
    bar_colors = ["#4C78A8", "#ED0F2C", "#A24B9B"]

    data["ci95"] = 1.96 * data["se_ctx_accuracy"]
    data["ymin"] = data["ctx_accuracy"] - data["ci95"]
    data["ymax"] = data["ctx_accuracy"] + data["ci95"]

    color_map = dict(zip(model_size_order, bar_colors))
    plot = (
        pn.ggplot(data, pn.aes(x="model_size", y="ctx_accuracy", fill="model_size"))
        + pn.geom_col(width=0.65, alpha=0.5)
        + pn.geom_errorbar(
            pn.aes(ymin="ymin", ymax="ymax"),
            width=0.2,
        )
        + pn.geom_hline(yintercept=0.5, linetype="dashed", alpha=0.5)
        + pn.scale_fill_manual(values=color_map)
        + pn.labs(
            title="Overall Accuracy by Model Size",
            x="Model Size",
            y="Proportion Scope-Homomorphic Choices",
        )
        + pn.theme_minimal()
    )

    plot.save(os.path.join(output_dir, "accuracy_by_model_size.png"))

if __name__ == "__main__":
    eval_file = os.path.join(DATA_DIR, "eval_results.csv")
    output_dir = os.path.join(ROOT_DIR, "evaluation", "plots")
    model_size_accuracy = os.path.join(DATA_DIR, "model_size_accuracy.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots for each model size
    # probability_plot(eval_file, output_dir, model_size="110")
    # probability_plot(eval_file, output_dir, model_size="52")
    # probability_plot(eval_file, output_dir, model_size="350")
    
    # accuracy_plot(output_dir, model_size="110")
    # accuracy_plot(output_dir, model_size="52")
    # accuracy_plot(output_dir, model_size="350")
    
    # split_plot(output_dir, model_size="110")
    # split_plot(output_dir, model_size="52")
    # split_plot(output_dir, model_size="350")

    plot_accuracy_by_model_size(model_size_accuracy, output_dir)
    probability_violin_by_model_size(eval_file, output_dir, model_size_order=["52", "110", "350"])

    #new plots with more shown in one image
    ctx_accuracy_by_type_and_model_size(
        output_dir,
        model_sizes=["52", "110", "350"],
    )
    delta_violin_by_type_and_model_size(eval_file, output_dir, model_sizes=["52", "110", "350"])