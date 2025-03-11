from typing import List

import numpy as np

from core.data.datasets.few_shot_dataset import FewShotDataset
from core.data.tasks.task import Task


def calculate_accuracy(task: Task, predictions: List[str], expected_outputs: List[str]) -> List[bool]:
    correct = _evaluate_predictions(task, predictions, expected_outputs)
    accuracy = correct.mean()
    return accuracy


def calculate_accuracy_on_datasets(task: Task, predictions: List[str], datasets: List[FewShotDataset]) -> List[bool]:
    expected_outputs = [dataset.test_output for dataset in datasets]
    return calculate_accuracy(task, predictions, expected_outputs)


def print_evaluation_summary(task: Task, predictions: List[str], datasets: List[FewShotDataset]) -> None:
    expected_outputs = [dataset.test_output for dataset in datasets]
    inputs = [dataset.test_input for dataset in datasets]
    correct = _evaluate_predictions(task, predictions, expected_outputs)
    accuracy = correct.mean()

    print("Out:\t", predictions)
    print("Exp.:\t", expected_outputs)

    print(f"Accuracy: {accuracy:.2f}")
    if accuracy < 1:
        print("Error cases:")

        # print as a table, with header. The column width is the length of the longest string in the column or the header
        headers = ["Input", "Output", "Expected"]

        column_width = (
            max(max(len(str(x)) for x in column) for column in zip(headers, inputs, predictions, expected_outputs)) + 4
        )

        # sort all lists by the input
        inputs, predictions, expected_outputs, correct = zip(
            *sorted(zip(inputs, predictions, expected_outputs, correct), key=lambda x: x[0])
        )

        print(f"{'Input':{column_width}}{'Output':{column_width}}{'Expected':{column_width}}")
        for inp, prediction, expected_output, corr in zip(inputs, predictions, expected_outputs, correct):
            if not corr:
                print(f"{inp:{column_width}}{prediction:{column_width}}{expected_output:{column_width}}")


def _evaluate_predictions(task: Task, predictions: List[str], expected_outputs: List[str]) -> List[bool]:
    predictions = _strip_whitespace(predictions)
    expected_outputs = _strip_whitespace(expected_outputs)

    vectorized_compare = np.vectorize(task.compare_outputs)
    correct = vectorized_compare(predictions, expected_outputs)

    return correct


def _strip_whitespace(lst: List[str]) -> List[str]:
    return [x.strip() for x in lst]


def _compare(prediction: str, expected_output: str) -> bool:
    return prediction == expected_output


def run_overriding_experiment_on_task_pair(
    model, 
    tokenizer, 
    task_name, 
    overriding_task_name,
    num_vectors=5
):
    seed_everything(41)

    num_examples = 4

    task = get_task_by_name(tokenizer, task_name)
    overriding_task = get_task_by_name(tokenizer, overriding_task_name)

    test_datasets = task.create_datasets(num_datasets=1000, num_examples=num_examples)
    overriding_datasets = overriding_task.create_datasets(num_datasets=100, num_examples=num_examples)

    # filter only test_datasets that are valid inputs for the overriding task
    test_datasets = [dataset for dataset in test_datasets if is_valid_input(overriding_task, dataset.test_input)]
    test_datasets = test_datasets[: len(overriding_datasets)]

    assert len(test_datasets) == len(overriding_datasets)

    # Run standard ICL for comparison
    icl_predictions = run_icl(model, tokenizer, task, test_datasets)
    
    # Get the best layer for task vector injection
    dev_accuracy_by_layer = task_vector_accuracy_by_layer(
        model,
        tokenizer,
        task,
        overriding_datasets,
    )
    best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))
    
    # 1. Get standard single task vector
    standard_task_hiddens = get_task_hiddens(model, tokenizer, task, overriding_datasets)
    
    # 2. Get multiple task vectors and calculate mean
    multiple_task_hiddens = get_multiple_task_vectors(
        model, 
        tokenizer, 
        task, 
        overriding_datasets, 
        num_vectors=num_vectors
    )
    print(f"multiple_task_hiddens shape: {multiple_task_hiddens.shape}")
    mean_task_hiddens = aggregate_task_vectors(multiple_task_hiddens, method="mean")
    
    # Generate predictions with standard task vector
    standard_tv_predictions = modulated_generate(
        model,
        tokenizer,
        task,
        test_datasets,
        task_hiddens=standard_task_hiddens,
        intermediate_layer=best_intermediate_layer,
    )
    
    # Generate predictions with mean task vector
    mean_tv_predictions = modulated_generate(
        model,
        tokenizer,
        task,
        test_datasets,
        task_hiddens=mean_task_hiddens,
        intermediate_layer=best_intermediate_layer,
    )

    expected_outputs_original = [dataset.test_output for dataset in test_datasets]
    expected_outputs_patched = [overriding_task.calc_output(dataset.test_input) for dataset in test_datasets]

    # Calculate accuracies for all methods
    icl_accuracy_original = calculate_accuracy(task, icl_predictions, expected_outputs_original)
    icl_accuracy_patched = calculate_accuracy(task, icl_predictions, expected_outputs_patched)
    
    standard_tv_accuracy_original = calculate_accuracy(task, standard_tv_predictions, expected_outputs_original)
    standard_tv_accuracy_patched = calculate_accuracy(task, standard_tv_predictions, expected_outputs_patched)
    
    mean_tv_accuracy_original = calculate_accuracy(task, mean_tv_predictions, expected_outputs_original)
    mean_tv_accuracy_patched = calculate_accuracy(task, mean_tv_predictions, expected_outputs_patched)

    # Print comparative results
    print(f"\n{'=' * 50}")
    print(f"Results for {task_name} -> {overriding_task_name}:")
    print(f"{'Method':<15} {'Original Task':<15} {'Patched Task':<15}")
    print(f"{'-' * 15} {'-' * 15} {'-' * 15}")
    print(f"{'ICL':<15} {icl_accuracy_original:.4f} {icl_accuracy_patched:.4f}")
    print(f"{'Standard TV':<15} {standard_tv_accuracy_original:.4f} {standard_tv_accuracy_patched:.4f}")
    print(f"{'Mean TV':<15} {mean_tv_accuracy_original:.4f} {mean_tv_accuracy_patched:.4f}")
    print(f"{'=' * 50}\n")

    return {
        "icl_accuracy_original": icl_accuracy_original,
        "icl_accuracy_patched": icl_accuracy_patched,
        "standard_tv_accuracy_original": standard_tv_accuracy_original,
        "standard_tv_accuracy_patched": standard_tv_accuracy_patched,
        "mean_tv_accuracy_original": mean_tv_accuracy_original,
        "mean_tv_accuracy_patched": mean_tv_accuracy_patched,
        "num_vectors": num_vectors
    }
