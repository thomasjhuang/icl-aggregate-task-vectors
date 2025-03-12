from dotenv import load_dotenv

load_dotenv(".env")

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

from core.config import FIGURES_DIR
from core.data.task_helpers import get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.task_vectors import get_task_hiddens, get_multi_context_task_hiddens, task_vector_accuracy_by_layer
from core.utils.misc import limit_gpus, seed_everything
from scripts.experiments.main import TASKS_TO_EVALUATE


def create_task_vectors(model, tokenizer):
    standard_task_vectors = {}
    mean_task_vectors = {}

    for task_name in tqdm(TASKS_TO_EVALUATE):
        num_examples = 4

        task = get_task_by_name(tokenizer, task_name)

        test_datasets = task.create_datasets(num_datasets=50, num_examples=num_examples)
        dev_datasets = task.create_datasets(num_datasets=50, num_examples=num_examples)

        # Get standard task vectors
        standard_task_hiddens = get_task_hiddens(model, tokenizer, task, test_datasets)
        
        dev_accuracy_by_layer = task_vector_accuracy_by_layer(
            model, tokenizer, task, dev_datasets,
            layers_to_test=range(10, 20)
        )
        best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))
        standard_task_vectors[task_name] = standard_task_hiddens[:, best_intermediate_layer]
        
        # Get mean task vectors
        multi_context_task_hiddens = get_multi_context_task_hiddens(model, tokenizer, task, test_datasets)
        # Use the same best layer for consistency
        mean_task_vectors[task_name] = multi_context_task_hiddens[:, best_intermediate_layer]

    return standard_task_vectors, mean_task_vectors


def create_tsne_plot(standard_task_vectors, mean_task_vectors):
    # Apply t-SNE separately to each type of vectors for clearer visualization
    all_standard_vectors = torch.cat(list(standard_task_vectors.values()), dim=0)
    all_mean_vectors = torch.cat(list(mean_task_vectors.values()), dim=0)
    
    # Create task labels for each set
    standard_task_labels = []
    mean_task_labels = []
    
    for i, task_name in enumerate(standard_task_vectors.keys()):
        standard_task_labels.extend([i] * standard_task_vectors[task_name].shape[0])
        mean_task_labels.extend([i] * mean_task_vectors[task_name].shape[0])
    
    # Apply t-SNE to all vectors combined to keep them in the same space
    combined_vectors = torch.cat([all_standard_vectors, all_mean_vectors], dim=0)
    dim_reduction = TSNE(n_components=2, random_state=41, perplexity=30)
    all_vectors_2d = dim_reduction.fit_transform(combined_vectors)
    
    # Split the projected vectors back into standard and mean
    standard_count = all_standard_vectors.shape[0]
    standard_vectors_2d = all_vectors_2d[:standard_count]
    mean_vectors_2d = all_vectors_2d[standard_count:]

    # Create a figure with larger size
    plt.figure(figsize=(14, 10))
    
    # Visualization parameters
    task_colors = plt.cm.tab20(np.linspace(0, 1, len(TASKS_TO_EVALUATE)))
    standard_marker = 'o'  # circle
    mean_marker = '*'      # star (very distinctive)
    
    # Plot standard vectors with circles
    for task_idx, task_name in enumerate(standard_task_vectors.keys()):
        mask = np.array(standard_task_labels) == task_idx
        plt.scatter(
            standard_vectors_2d[mask, 0], 
            standard_vectors_2d[mask, 1],
            marker=standard_marker,
            s=80,           # Larger size
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=f"{task_name} (Standard)" if task_idx == 0 else None,
            color=task_colors[task_idx]
        )
    
    # Plot mean vectors with stars
    for task_idx, task_name in enumerate(mean_task_vectors.keys()):
        mask = np.array(mean_task_labels) == task_idx
        plt.scatter(
            mean_vectors_2d[mask, 0], 
            mean_vectors_2d[mask, 1],
            marker=mean_marker,
            s=150,          # Even larger for stars
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5,
            label=f"{task_name} (Mean)" if task_idx == 0 else None,
            color=task_colors[task_idx]
        )
    
    # Create a more distinctive legend at the top
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker=standard_marker, color='gray', label='Standard TV',
               markerfacecolor='gray', markersize=12, linestyle='None'),
        Line2D([0], [0], marker=mean_marker, color='gray', label='Mean TV',
               markerfacecolor='gray', markersize=16, linestyle='None')
    ]
    
    # Add task color legend elements
    for i, task_name in enumerate(TASKS_TO_EVALUATE):
        legend_elements.append(
            Line2D([0], [0], marker='s', color=task_colors[i], label=task_name,
                   markerfacecolor=task_colors[i], markersize=10, linestyle='None')
        )
    
    # Create two legends - one for vector types and one for tasks
    vector_legend = plt.legend(handles=legend_elements[:2], loc='upper left', 
                              title="Vector Type", fontsize=12, title_fontsize=14)
    plt.gca().add_artist(vector_legend)
    
    # Place tasks legend on the right side
    plt.legend(handles=legend_elements[2:], loc='center left', bbox_to_anchor=(1.01, 0.5),
              title="Tasks", fontsize=10, title_fontsize=12)
    
    plt.title("t-SNE Visualization: Standard vs Mean Task Vectors", fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(FIGURES_DIR, "task_vectors_tsne_comparison.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)


def create_histograms_plot(standard_task_vectors, mean_task_vectors):
    # Create a figure with 2 rows (standard and mean) and 2 columns (within-task and between-tasks)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Row titles
    row_titles = ["Standard Task Vectors", "Mean Task Vectors"]
    
    # Process both types of vectors
    for row, (title, vectors) in enumerate(zip(row_titles, [standard_task_vectors, mean_task_vectors])):
        # Calculate within-task distances
        within_task_distances = []
        for task_name in vectors.keys():
            task_vecs = vectors[task_name]
            if task_vecs.shape[0] > 1:  # Need at least 2 vectors to compute distances
                distances = cdist(task_vecs, task_vecs, metric="cosine")
                # Remove self-comparisons (diagonal)
                mask = ~np.eye(distances.shape[0], dtype=bool)
                within_task_distances.append(distances[mask])
        
        within_task_flat = np.concatenate(within_task_distances)
        
        # Calculate between-task distances
        between_task_distances = []
        task_names = list(vectors.keys())
        for i, task1 in enumerate(task_names):
            for task2 in task_names[i+1:]:
                distances = cdist(vectors[task1], vectors[task2], metric="cosine")
                between_task_distances.append(distances.flatten())
        
        between_task_flat = np.concatenate(between_task_distances)
        
        # Plot histograms
        axs[row, 0].hist(within_task_flat, bins=50, alpha=0.7, density=True)
        axs[row, 0].set_title(f"{title}: Within-Task Distances")
        axs[row, 0].set_xlabel("Cosine Distance")
        axs[row, 0].set_ylabel("Density")
        
        axs[row, 1].hist(between_task_flat, bins=50, alpha=0.7, density=True)
        axs[row, 1].set_title(f"{title}: Between-Task Distances")
        axs[row, 1].set_xlabel("Cosine Distance")
        axs[row, 1].set_ylabel("Density")
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "task_vectors_histograms_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Also create a combined histogram showing all distributions
    plt.figure(figsize=(10, 6))
    
    within_task_flat_std = np.concatenate([cdist(standard_task_vectors[task_name], standard_task_vectors[task_name], 
                                               metric="cosine")[~np.eye(len(standard_task_vectors[task_name]), dtype=bool)].flatten() 
                                        for task_name in standard_task_vectors.keys()])
    
    between_task_flat_std = np.concatenate([cdist(standard_task_vectors[task1], standard_task_vectors[task2], metric="cosine").flatten()
                                         for i, task1 in enumerate(task_names)
                                         for task2 in task_names[i+1:]])
    
    within_task_flat_mean = np.concatenate([cdist(mean_task_vectors[task_name], mean_task_vectors[task_name], 
                                                metric="cosine")[~np.eye(len(mean_task_vectors[task_name]), dtype=bool)].flatten() 
                                         for task_name in mean_task_vectors.keys()])
    
    between_task_flat_mean = np.concatenate([cdist(mean_task_vectors[task1], mean_task_vectors[task2], metric="cosine").flatten()
                                          for i, task1 in enumerate(task_names)
                                          for task2 in task_names[i+1:]])
    
    plt.hist(within_task_flat_std, bins=50, alpha=0.5, label="Standard TV: Within-Task", density=True)
    plt.hist(between_task_flat_std, bins=50, alpha=0.5, label="Standard TV: Between-Task", density=True)
    plt.hist(within_task_flat_mean, bins=50, alpha=0.5, label="Mean TV: Within-Task", density=True)
    plt.hist(between_task_flat_mean, bins=50, alpha=0.5, label="Mean TV: Between-Task", density=True)
    
    plt.xlabel("Cosine Distance")
    plt.ylabel("Density")
    plt.title("Distance Distributions: Standard vs Mean Task Vectors")
    plt.legend()
    
    save_path = os.path.join(FIGURES_DIR, "task_vectors_combined_histogram.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return {
        "standard": {
            "within": within_task_flat_std,
            "between": between_task_flat_std
        },
        "mean": {
            "within": within_task_flat_mean,
            "between": between_task_flat_mean
        }
    }


def print_histogram_stats(distance_data):
    stats = {
        "Method": ["Standard TV", "Standard TV", "Mean TV", "Mean TV"],
        "Comparison": ["Within-Task", "Between-Task", "Within-Task", "Between-Task"],
        "Mean": [
            np.mean(distance_data["standard"]["within"]),
            np.mean(distance_data["standard"]["between"]),
            np.mean(distance_data["mean"]["within"]),
            np.mean(distance_data["mean"]["between"])
        ],
        "Std": [
            np.std(distance_data["standard"]["within"]),
            np.std(distance_data["standard"]["between"]),
            np.std(distance_data["mean"]["within"]),
            np.std(distance_data["mean"]["between"])
        ],
        "Min": [
            np.min(distance_data["standard"]["within"]),
            np.min(distance_data["standard"]["between"]),
            np.min(distance_data["mean"]["within"]),
            np.min(distance_data["mean"]["between"])
        ],
        "Max": [
            np.max(distance_data["standard"]["within"]),
            np.max(distance_data["standard"]["between"]),
            np.max(distance_data["mean"]["within"]),
            np.max(distance_data["mean"]["between"])
        ]
    }
    
    df = pd.DataFrame(stats)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    
    # Calculate separation metrics
    for method in ["standard", "mean"]:
        within_mean = np.mean(distance_data[method]["within"])
        between_mean = np.mean(distance_data[method]["between"])
        separation = (between_mean - within_mean) / within_mean * 100
        print(f"\n{method.title()} Task Vector Separation: {separation:.2f}%")


def main():
    seed_everything(41)
    limit_gpus(range(0, 1))

    model_type, model_variant = "llama", "7B"
    model, tokenizer = load_model_and_tokenizer(model_type, model_variant)

    standard_task_vectors, mean_task_vectors = create_task_vectors(model, tokenizer)

    create_tsne_plot(standard_task_vectors, mean_task_vectors)
    distance_data = create_histograms_plot(standard_task_vectors, mean_task_vectors)
    print_histogram_stats(distance_data)


if __name__ == "__main__":
    main()
