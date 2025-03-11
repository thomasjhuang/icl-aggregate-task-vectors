import os
import pickle
from typing import List
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from core.config import FIGURES_DIR

from scripts.figures.helpers import (
    MODEL_DISPLAY_NAME_MAPPING,
    load_main_results,
    extract_accuracies,
    create_accuracies_df,
    create_grouped_accuracies_df,
)


def filter_tasks_with_low_icl_accuracy(grouped_accuracies_df, regular_accuracy_threshold=0.20):
    mask = (grouped_accuracies_df["Regular"] - grouped_accuracies_df["Baseline"]) >= regular_accuracy_threshold
    filtered_task_accuracies_df = grouped_accuracies_df[mask].copy()

    # print the excluded model,task pairs, Hypothesis by commas
    if not mask.all():
        print(
            "Excluded:",
            grouped_accuracies_df[~mask][["model", "task_name", "Regular"]].apply(
                lambda x: f"({x['model']}, {x['task_name']}): {x['Regular']:.2f}", axis=1
            ),
        )
    print("Num excluded / total:", (~mask).sum(), "/", len(grouped_accuracies_df))

    return filtered_task_accuracies_df


def plot_avg_accuracies_per_model(grouped_accuracies_df):
    filtered_task_accuracies_df = filter_tasks_with_low_icl_accuracy(grouped_accuracies_df)

    # Update columns to include MeanTV if it exists
    columns_to_plot = ["Baseline", "Task Vector", "Regular"]
    if "Mean Task Vector" in filtered_task_accuracies_df.columns:
        columns_to_plot.append("Mean Task Vector")

    # Calculate average accuracy and std deviation for each model
    df_agg = filtered_task_accuracies_df.groupby("model")[columns_to_plot].agg("mean")

    # Plotting
    # Sort the model names, firsts by the base name, then by the size
    model_names = df_agg.index.unique()
    num_models = len(model_names)
    model_names = sorted(model_names, key=lambda x: (x.split(" ")[0], float(x.split(" ")[1][:-1])))

    plt.rcParams.update({"font.size": 14})  # Set font size

    fig, ax = plt.subplots(figsize=(6, 6))

    bar_width = 0.3 if len(columns_to_plot) <= 3 else 0.25
    hatches = ["/", "\\", "|", "x"]
    colors = ["grey", "blue", "green", "purple"]
    
    for j, column in enumerate(columns_to_plot):
        means = df_agg[column]
        y_positions = np.arange(len(means)) + (j - (len(columns_to_plot)-1)/2) * bar_width
        ax.barh(
            y_positions,
            means,
            height=bar_width,
            capsize=2,
            color=colors[j % len(colors)],
            edgecolor="white",
            hatch=hatches[j % len(hatches)] * 2,
        )

    # set the y ticks to be the model names
    ax.set_yticks(np.arange(num_models))
    ax.set_yticklabels([model_name for model_name in model_names])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Accuracy")
    ax.set_xlim([0.0, 1.0])

    # show legend below the plot
    legend_elements = [
        Patch(facecolor=colors[i], edgecolor="white", hatch=hatches[i % len(hatches)] * 2, label=column)
        for i, column in enumerate(columns_to_plot)
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(columns_to_plot))

    # save the figure
    save_path = os.path.join(FIGURES_DIR, "main_experiment_results_per_model.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_accuracy_by_layer(results, model_names: List[str], normalize_x_axis: bool = False, filename_suffix: str = ""):
    plt.figure(figsize=(10, 5))

    plt.rc("font", size=16)

    plt.title(f"Average Accuracy by Layer")

    regular_accuracy_threshold = 0.75

    num_layers = {
        "llama_7B": 32,
        "llama_13B": 40,
        "gpt-j_6B": 28,
        "pythia_2.8B": 32,
        "pythia_6.9B": 32,
    }

    # Define different markers for each model
    markers = ["o", "^", "s", "P", "X", "D", "v"]

    for idx, model_name in enumerate(model_names):
        min_num_layers = min(
            len(results[model_name][task_name]["tv_dev_accruacy_by_layer"]) for task_name in results[model_name]
        )
        all_tv_dev_accruacy_by_layer = np.array(
            [
                np.array(list(results[model_name][task_name]["tv_dev_accruacy_by_layer"].values())[:min_num_layers])
                for task_name in results[model_name]
            ]
        )

        all_tv_dev_accruacy_by_layer = all_tv_dev_accruacy_by_layer[
            all_tv_dev_accruacy_by_layer.max(axis=-1) > regular_accuracy_threshold
        ]

        mean_tv_dev_accruacy_by_layer = np.mean(all_tv_dev_accruacy_by_layer, axis=0)
        std_tv_dev_accruacy_by_layer = np.std(all_tv_dev_accruacy_by_layer, axis=0)

        layers = np.array(list(list(results[model_name].values())[0]["tv_dev_accruacy_by_layer"].keys()))
        layers_fraction = layers / (max(layers) / 0.9)

        x_values = layers
        if normalize_x_axis:
            x_values = x_values / num_layers[model_name]

        # Use different marker for each model and increase the marker size
        plt.plot(
            x_values,
            mean_tv_dev_accruacy_by_layer,
            marker=markers[idx],
            markersize=10,
            label=MODEL_DISPLAY_NAME_MAPPING[model_name],
            alpha=0.8,
        )
        plt.fill_between(
            x_values,
            mean_tv_dev_accruacy_by_layer - std_tv_dev_accruacy_by_layer,
            mean_tv_dev_accruacy_by_layer + std_tv_dev_accruacy_by_layer,
            alpha=0.1,
        )

    plt.xlabel("Layer")
    plt.ylabel("Accuracy")

    plt.ylim(0.0, 1.0)

    # place the legend on the top right corner
    plt.legend(loc="upper right")

    # save the figure
    save_path = os.path.join(FIGURES_DIR, f"accuracy_per_layer{filename_suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def create_results_latex_table(grouped_accuracies_df):
    prefix = "\\onecolumn\n\\begin{center}\n\\small"
    suffix = "\\end{center}\n\\twocolumn"

    caption_and_label = (
        r"""\caption{Complete results of the main experiment for all tasks and models.} \label{table:main_results} \\"""
    )

    table_df = grouped_accuracies_df.copy()

    table_df = table_df.sort_values(by=["model", "task_type", "task_name"])

    table_df["task_name"] = table_df["task_name"].str.replace("_", " ").str.capitalize()
    table_df["task_type"] = table_df["task_type"].str.capitalize()

    table_df.columns = table_df.columns.str.replace("_", " ").str.capitalize()

    # set ["model", "task_type", "task_name"] as index
    table_df = table_df.set_index(["Model", "Task type", "Task name"])

    # sort the index by the model name. make sure to have "Pythia 2.9B" before "Pythia 7B" etc.
    table_df = table_df.sort_index(level=0)

    table_latex = table_df.to_latex(index=True, multirow=True, float_format="%.2f")

    table_latex = table_latex.replace("tabular", "longtable")

    original_head = "\n".join(table_latex.split("\n")[1:5])

    new_head = (
        original_head
        + r"""
    \endfirsthead

    \multicolumn{3}{c}
    {{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\
    """
        + original_head
        + r"""
    \endhead
    """
    )

    table_latex = table_latex.replace(original_head, new_head)

    # add the caption and label after the first line
    table_latex = (
        "\n".join(table_latex.split("\n")[:1])
        + "\n\n"
        + caption_and_label
        + "\n\n"
        + "\n".join(table_latex.split("\n")[1:])
    )

    final_latex = prefix + "\n" + table_latex + "\n" + suffix

    save_path = os.path.join(FIGURES_DIR, "main_experiment_results_table.tex")
    with open(save_path, "w") as f:
        f.write(final_latex)


def create_top_tokens_table(results):
    # Top Tokens
    task_names = [
        "algorithmic_prev_letter",
        "translation_fr_en",
        "linguistic_present_simple_gerund",
        "knowledge_country_capital",
    ]

    model_names = ["llama_13B", "pythia_6.9B", "gpt-j_6B"]

    df_data = {}

    for model_name in model_names:
        df_data[model_name] = {}
        model_results = results[model_name]

        def remove_duplicates_ignore_case(lst):
            seen = set()
            output = []
            for s in lst:
                if s.lower() not in seen:
                    output.append(s)
                    seen.add(s.lower())
            return output

        top_words_per_task = {}
        for task_name in task_names:
            # Skip if task doesn't exist in results
            if task_name not in model_results:
                print(f"Warning: Task {task_name} not found in results for model {model_name}")
                top_words_per_task[task_name] = "N/A"
                continue
                
            task_results = model_results[task_name]
            
            # Skip if necessary keys don't exist
            if "tv_dev_accruacy_by_layer" not in task_results or "tv_ordered_tokens_by_layer" not in task_results:
                print(f"Warning: Missing required data for task {task_name} in model {model_name}")
                top_words_per_task[task_name] = "N/A"
                continue

            dev_accuracy_by_layer = task_results["tv_dev_accruacy_by_layer"]
            best_layer_index = max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get)
            
            # Calculate the layer key, but handle potential offsets safely
            available_layers = list(task_results["tv_ordered_tokens_by_layer"].keys())
            if not available_layers:
                print(f"Warning: No layers found in tv_ordered_tokens_by_layer for {task_name}")
                top_words_per_task[task_name] = "N/A"
                continue
                
            # First try best_layer_index + 2
            best_layer = best_layer_index + 2
            
            # If that doesn't exist, try direct mapping or nearest available
            if best_layer not in task_results["tv_ordered_tokens_by_layer"]:
                if best_layer_index in task_results["tv_ordered_tokens_by_layer"]:
                    best_layer = best_layer_index
                else:
                    # Find closest available layer
                    best_layer = min(available_layers, key=lambda x: abs(x - best_layer_index))
                    print(f"Warning: Using nearest layer {best_layer} instead of {best_layer_index + 2} for {task_name}")
            
            try:
                top_words = task_results["tv_ordered_tokens_by_layer"][best_layer]
                
                top_words = [x.strip() for x in top_words]
                
                # filter tokens that are only a-z or A-Z
                top_words = [w for w in top_words if re.match("^[a-zA-Z]+$", w)]
                
                # remove duplicates
                top_words = remove_duplicates_ignore_case(top_words)
                
                top_words_per_task[task_name] = ", ".join(top_words[:20])
            except (KeyError, IndexError) as e:
                print(f"Error processing top tokens for {task_name} in {model_name}: {e}")
                top_words_per_task[task_name] = "Error"

        df_data[model_name] = top_words_per_task

    # create a dataframe with 2 indexes: model and task, and 1 column: top tokens
    df = pd.DataFrame.from_dict(df_data, orient="index").stack().to_frame()

    # save the table as a latex table
    save_path = os.path.join(FIGURES_DIR, "top_tokens_table.tex")
    with open(save_path, "w") as f:
        f.write(df.to_latex())


def create_all_figures(experiment_id: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = load_main_results(experiment_id)
    accuracies = extract_accuracies(results)
    accuracies_df = create_accuracies_df(results)
    grouped_accuracies_df = create_grouped_accuracies_df(accuracies_df)

    plot_avg_accuracies_per_model(grouped_accuracies_df)
    plot_accuracy_by_layer(results, model_names=["llama_7B", "llama_13B"])
    plot_accuracy_by_layer(
        results, model_names=["pythia_2.8B", "pythia_6.9B", "gpt-j_6B"], filename_suffix="_appendix"
    )
    create_results_latex_table(grouped_accuracies_df)
    create_top_tokens_table(results)


if __name__ == "__main__":
    experiment_id = "52"

    create_all_figures(experiment_id)
