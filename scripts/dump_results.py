#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
from tabulate import tabulate
import pickle
import os

from scripts.figures.helpers import load_main_results, load_overriding_results, extract_accuracies
from scripts.utils import main_experiment_results_dir, overriding_experiment_results_dir


def format_accuracies(results, experiment_type="main"):
    """Extract and format accuracy metrics from results."""
    if experiment_type == "main":
        # For main experiment, use the helper to extract accuracies in a clean format
        accuracies = extract_accuracies(results)
        
        # Convert to a DataFrame for easier display
        rows = []
        for model_name, model_results in accuracies.items():
            for task_name, task_results in model_results.items():
                row = {
                    "Model": model_name,
                    "Task": task_name,
                    "Baseline": f"{task_results['bl']:.4f}",
                    "ICL": f"{task_results['icl']:.4f}",
                    "Task Vector": f"{task_results['tv']:.4f}"
                }
                # Add TV Mean if available
                if "tv_mean" in task_results:
                    row["TV Mean"] = f"{task_results['tv_mean']:.4f}"
                rows.append(row)
        return pd.DataFrame(rows)
    
    else:  # overriding experiment
        rows = []
        for model_name, model_results in results.items():
            for experiment_name, experiment_results in model_results.items():
                rows.append({
                    "Model": model_name,
                    "Experiment": experiment_name,
                    "ICL Original": f"{experiment_results['icl_accuracy_original']:.4f}",
                    "ICL Patched": f"{experiment_results['icl_accuracy_patched']:.4f}",
                    "Standard TV Original": f"{experiment_results['standard_tv_accuracy_original']:.4f}",
                    "Standard TV Patched": f"{experiment_results['standard_tv_accuracy_patched']:.4f}",
                    "Mean TV Original": f"{experiment_results['mean_tv_accuracy_original']:.4f}",
                    "Mean TV Patched": f"{experiment_results['mean_tv_accuracy_patched']:.4f}"
                })
        return pd.DataFrame(rows)


def check_vector_data(results, experiment_type="main"):
    """Check if there's any vector data in the results"""
    vector_info = {}
    
    if experiment_type == "main":
        for model_name, model_results in results.items():
            vector_info[model_name] = {}
            for task_name, task_results in model_results.items():
                vector_data = {
                    "task": task_name,
                    "has_ordered_tokens": "tv_ordered_tokens_by_layer" in task_results,
                    "best_layer": None,
                    "tokens_in_layer": [],
                    "has_mean_tv": "tv_mean_accuracy" in task_results
                }
                
                # Check for best layer information
                if "tv_dev_accruacy_by_layer" in task_results:  # Note the typo in the key name
                    best_layer = max(task_results["tv_dev_accruacy_by_layer"], 
                                    key=task_results["tv_dev_accruacy_by_layer"].get)
                    vector_data["best_layer"] = best_layer
                
                # Check if we have ordered tokens
                if "tv_ordered_tokens_by_layer" in task_results:
                    for layer, tokens in task_results["tv_ordered_tokens_by_layer"].items():
                        vector_data["tokens_in_layer"].append(layer)
                
                vector_info[model_name][task_name] = vector_data
    
    return vector_info


def main():
    parser = argparse.ArgumentParser(description="Display accuracy metrics and vector data from experiments")
    parser.add_argument("--experiment-type", choices=["main", "overriding"], default="main",
                      help="Type of experiment (main or overriding)")
    parser.add_argument("--experiment-id", default="camera_ready",
                      help="Experiment ID (default: camera_ready)")
    parser.add_argument("--model", help="Optional: Filter by model name (e.g., llama_13B)")
    parser.add_argument("--task", help="Optional: Filter by task name")
    parser.add_argument("--sort-by", choices=["model", "task", "baseline", "icl", "tv", "tv_mean"], default="model",
                      help="Sort results by this column")
    parser.add_argument("--show-vectors", action="store_true", help="Show information about stored task vectors")
    parser.add_argument("--dump-raw", action="store_true", help="Dump raw pickled data for the specified model/task")
    
    args = parser.parse_args()
    
    # Load results based on experiment type
    if args.experiment_type == "main":
        results = load_main_results(args.experiment_id)
    else:
        results = load_overriding_results(args.experiment_id)
    
    # Filter by model if specified
    if args.model:
        if args.model in results:
            results = {args.model: results[args.model]}
        else:
            print(f"Model '{args.model}' not found. Available models: {list(results.keys())}")
            return
    
    if args.dump_raw and args.model and args.task:
        if args.task in results[args.model]:
            print(f"Raw data for {args.model}, task: {args.task}:")
            for key, value in results[args.model][args.task].items():
                if isinstance(value, (np.ndarray, list)) and len(str(value)) > 100:
                    print(f"{key}: [Data too large to display]")
                else:
                    print(f"{key}: {value}")
            return
    
    # Show vector information if requested
    if args.show_vectors:
        vector_info = check_vector_data(results, args.experiment_type)
        
        # Create a table with vector data information
        rows = []
        for model_name, model_vector_info in vector_info.items():
            for task_name, task_vector_info in model_vector_info.items():
                if args.task and args.task not in task_name:
                    continue
                    
                rows.append({
                    "Model": model_name,
                    "Task": task_name,
                    "Has Tokens": "Yes" if task_vector_info["has_ordered_tokens"] else "No",
                    "Has Mean TV": "Yes" if task_vector_info["has_mean_tv"] else "No",
                    "Best Layer": task_vector_info["best_layer"],
                    "Layers with Tokens": ",".join(map(str, task_vector_info["tokens_in_layer"]))
                })
        
        if rows:
            df = pd.DataFrame(rows)
            print("\nTask Vector Information:")
            print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
        else:
            print("\nNo vector information found in the results.")
        
        return
    
    # Format the accuracies
    df = format_accuracies(results, args.experiment_type)
    
    # Filter by task if specified
    if args.task and args.experiment_type == "main":
        df = df[df["Task"].str.contains(args.task, case=False)]
        if df.empty:
            print(f"No tasks matching '{args.task}' found.")
            return
    elif args.task and args.experiment_type == "overriding":
        df = df[df["Experiment"].str.contains(args.task, case=False)]
        if df.empty:
            print(f"No experiments matching '{args.task}' found.")
            return
    
    # Sort the results
    if args.sort_by == "model":
        df = df.sort_values("Model")
    elif args.sort_by == "task" and "Task" in df.columns:
        df = df.sort_values("Task")
    elif args.sort_by == "baseline" and "Baseline" in df.columns:
        df = df.sort_values("Baseline", ascending=False)
    elif args.sort_by == "icl" and "ICL" in df.columns:
        df = df.sort_values("ICL", ascending=False)
    elif args.sort_by == "tv" and "Task Vector" in df.columns:
        df = df.sort_values("Task Vector", ascending=False)
    elif args.sort_by == "tv_mean" and "TV Mean" in df.columns:
        df = df.sort_values("TV Mean", ascending=False)
    
    # Display the table
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    
    # Add summary statistics
    if not df.empty:
        print("\nSummary Statistics:")
        
        # Identify numeric columns
        numeric_cols = []
        for col in df.columns:
            if col in ["Baseline", "ICL", "Task Vector", "TV Mean", "ICL Original", "ICL Patched", "TV Original", "TV Patched"]:
                numeric_cols.append(col)
        
        # Convert string values back to floats for calculations
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Calculate means
        means = df[numeric_cols].mean()
        print("\nMean Accuracies:")
        for col, mean in means.items():
            print(f"{col}: {mean:.4f}")


if __name__ == "__main__":
    main()