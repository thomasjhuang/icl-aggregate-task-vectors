from typing import Dict, List, Optional, Tuple, Union, Iterable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.data.datasets.few_shot_dataset import FewShotDataset
from core.data.tasks.task import Task
from core.models.context_managers.forward_modifiers.hidden_injector import HiddenInjector
from core.models.utils.inference import (
    batch_forward,
    batch_generate,
    decode_predictions,
    get_input_type,
    modified_forward,
    tokenize_datasets,
    traced_forward,
)
from core.models.utils.llm_layers import get_layers
from core.utils.nested import nested_apply
import random


def run_icl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    include_train: bool = True,
) -> List[str]:
    format_dataset_kwargs = {"include_train": include_train}
    inputs = tokenize_datasets(tokenizer, test_datasets, format_dataset_kwargs=format_dataset_kwargs)
    new_ids = batch_generate(model, tokenizer, inputs=inputs, generate_kwargs={"max_new_tokens": 1})
    predictions = decode_predictions(new_ids, tokenizer)

    return predictions


def run_task_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    dev_datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
    multi_context: bool = False,
    use_fusion: bool = False,
    fusion_method: str = "mean",
    num_vectors: int = 5,
):
    dev_accuracy_by_layer = task_vector_accuracy_by_layer(
        model,
        tokenizer,
        task,
        dev_datasets,
        layers_to_test=layers_to_test,
        multi_context=multi_context,
    )
    best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))

    if use_fusion:
        # Get multiple task vectors and aggregate them
        multiple_task_hiddens = get_multiple_task_vectors(
            model, 
            tokenizer, 
            task, 
            test_datasets, 
            num_vectors=num_vectors,
            multi_context=multi_context
        )
        task_hiddens = aggregate_task_vectors(multiple_task_hiddens, method=fusion_method)
    else:
        # Standard single task vector
        task_hiddens = get_task_hiddens(model, tokenizer, task, test_datasets, multi_context=multi_context)
    
    predictions = modulated_generate(
        model,
        tokenizer,
        task,
        test_datasets,
        task_hiddens=task_hiddens,
        intermediate_layer=best_intermediate_layer,
    )

    return predictions, dev_accuracy_by_layer, task_hiddens


def run_overriding_task_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    overriding_datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
    use_aggregation: bool = True,
    aggregation_method: str = "mean",
    num_vectors: int = 5,
):
    """
    Run task vector overriding with vector aggregation.
    
    Args:
        model: The language model
        tokenizer: The model's tokenizer
        task: The task to be evaluated
        test_datasets: Datasets used for testing (inputs from Task A)
        overriding_datasets: Datasets used for overriding (from Task B)
        layers_to_test: Layers to test for best performance
        use_aggregation: Whether to use aggregation of multiple task vectors
        aggregation_method: Method to aggregate vectors ("mean")
        num_vectors: Number of task vectors to aggregate
    """
    dev_accuracy_by_layer = task_vector_accuracy_by_layer(
        model,
        tokenizer,
        task,
        overriding_datasets,
        layers_to_test=layers_to_test,
    )
    best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))

    if use_aggregation:
        multiple_task_hiddens = get_multiple_task_vectors(
            model, 
            tokenizer, 
            task, 
            overriding_datasets, 
            num_vectors=num_vectors
        )
        print("multiple_task_hiddens.shape: %s", multiple_task_hiddens.shape)
        task_hiddens = aggregate_task_vectors(multiple_task_hiddens, method=aggregation_method)
    else:
        task_hiddens = get_task_hiddens(model, tokenizer, task, overriding_datasets)

    predictions = modulated_generate(
        model,
        tokenizer,
        task,
        test_datasets,
        task_hiddens=task_hiddens,
        intermediate_layer=best_intermediate_layer,
        include_train=True,
    )

    return predictions, dev_accuracy_by_layer, task_hiddens


def get_multi_context_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
) -> torch.Tensor:
    inputs = tokenize_datasets(tokenizer, datasets)

    outputs, forward_trace = traced_forward(model, inputs=inputs)

    # This is exactly where the task vector is extracted, it is at the position where the model needs to 
    # make a prediction.
    task_hiddens = forward_trace.residual_stream.hidden[:, :, -1, :]

    # for each dataset, average task hiddens from other datasets that did not include the test_input from the current dataset
    mask = torch.ones(len(datasets), len(datasets))
    for i, dataset in enumerate(datasets):
        for j, other_dataset in enumerate(datasets):
            if dataset.test_input in other_dataset.train_inputs or dataset.test_input == other_dataset.test_input:
                mask[i, j] = 0

    task_hiddens = torch.cat([task_hiddens[mask[i].bool()].mean(dim=0).unsqueeze(0) for i in range(len(datasets))])

    task_hiddens = task_hiddens[:, 1:]  # the first one is the embedding layer

    return task_hiddens  # (num_datasets, num_layers, hidden_size)


def get_single_context_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    num_test_inputs_to_avg: int = 2,
) -> torch.Tensor:
    new_datasets = [
        FewShotDataset(
            train_inputs=dataset.train_inputs,
            train_outputs=dataset.train_outputs,
            test_input=test_input,
            test_output=task.calc_output(test_input),
        )
        for dataset in datasets
        for test_input in task.sample_inputs(num_test_inputs_to_avg, exclude=(dataset.test_input,))
    ]

    inputs = tokenize_datasets(tokenizer, new_datasets)

    # TODO: replace traced forward with a regular forward and rely on huggingface's saved hidden states
    outputs, forward_trace = traced_forward(model, inputs=inputs)

    task_hiddens = forward_trace.residual_stream.hidden[:, :, -1, :]
    _, num_layers, hidden_size = task_hiddens.shape
    task_hiddens = task_hiddens.view(len(datasets), num_test_inputs_to_avg, num_layers, hidden_size).mean(dim=1)

    task_hiddens = task_hiddens[:, 1:]  # the first one is the embedding layer

    return task_hiddens  # (num_datasets, num_layers, hidden_size)


def get_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    multi_context: bool = False,
) -> torch.Tensor:
    if multi_context:
        return get_multi_context_task_hiddens(model, tokenizer, task, datasets)
    else:
        return get_single_context_task_hiddens(model, tokenizer, task, datasets)


def modulated_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    task_hiddens: torch.tensor,
    intermediate_layer: Union[int, torch.Tensor],
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    return_task_hiddens: bool = False,
    include_train: bool = False,
) -> List[str]:
    inputs = tokenize_datasets(tokenizer, test_datasets, format_dataset_kwargs={"include_train": include_train})

    first_forward_outputs = modulated_forward(
        model,
        inputs=inputs,
        task_hiddens=task_hiddens,
        intermediate_layer=intermediate_layer,
        past_key_values=past_key_values,
    )
    first_predicted_token_ids = first_forward_outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1)
    answers = decode_predictions(first_predicted_token_ids, tokenizer)

    if return_task_hiddens:
        return answers, task_hiddens
    return answers


def modulated_forward(
    model: PreTrainedModel,
    inputs: Dict,
    task_hiddens: torch.Tensor,
    intermediate_layer: int,
    batch_size: Optional[int] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
):
    # TODO: move all this to the HiddenInjector class
    if isinstance(intermediate_layer, int):
        intermediate_layer = torch.tensor(intermediate_layer).repeat(len(inputs["input_ids"]))
    injection_positions = -1 * torch.ones_like(intermediate_layer, dtype=torch.long)
    task_hiddens = task_hiddens[torch.arange(len(intermediate_layer)), intermediate_layer]

    forward_modifiers = [
        HiddenInjector(
            model,
            injection_layers=intermediate_layer,
            injection_positions=injection_positions,
            hiddens_to_inject=task_hiddens,
        )
    ]

    if past_key_values is not None:
        inputs[get_input_type(inputs)] = inputs[get_input_type(inputs)][:, -1].unsqueeze(1)

    first_forward_outputs = modified_forward(
        model,
        inputs=inputs,
        forward_kwargs={"past_key_values": past_key_values},
        forward_modifiers=forward_modifiers,
        batch_size=len(inputs["input_ids"]),  # TODO: need to enable batched forward with HiddenInjector
    )

    return first_forward_outputs


def task_vector_accuracy_by_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
    multi_context: bool = False,
) -> Dict[int, float]:
    if layers_to_test is None:
        num_layers = len(get_layers(model))
        layers_to_test = range(num_layers)

    # Get task hiddens
    task_hiddens = get_task_hiddens(model, tokenizer, task, datasets, multi_context=multi_context)

    # Get input past_key_values
    inputs = tokenize_datasets(tokenizer, datasets, format_dataset_kwargs={"include_train": False})
    outputs = batch_forward(model, inputs=inputs, forward_kwargs={"use_cache": True})
    past_key_values = outputs.past_key_values
    past_key_values = nested_apply(past_key_values, lambda x: x[:, :, :-1])  # remove last token from past_key_values
    inputs["input_ids"] = inputs["input_ids"][:, -1].unsqueeze(1)

    # Find best intermediate layer using dev set
    accuracies = []
    for layer_num in layers_to_test:
        answers = modulated_generate(
            model,
            tokenizer,
            task,
            datasets,
            intermediate_layer=layer_num,
            task_hiddens=task_hiddens,
            past_key_values=past_key_values,
        )

        accuracy = calculate_accuracy_on_datasets(task, answers, datasets)
        accuracies.append(accuracy)
    accuracy_by_layer = {layer: accuracy for layer, accuracy in zip(layers_to_test, accuracies)}

    return accuracy_by_layer


def continue_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: Dict,
    first_forward_outputs: CausalLMOutputWithPast,
    test_datasets: List[FewShotDataset],
) -> List[str]:
    """
    Continue generation after the first token. This is currently not supported.
    """
    first_predicted_token_ids = first_forward_outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1)

    new_input_ids = first_predicted_token_ids
    new_attention_mask = torch.ones_like(new_input_ids)

    full_input_ids = torch.cat([inputs["input_ids"], new_input_ids], dim=-1)
    full_attention_mask = torch.cat([inputs["attention_mask"], new_attention_mask], dim=-1)

    # full_input_ids = new_input_ids
    # full_attention_mask = new_attention_mask

    past_key_values = first_forward_outputs.past_key_values

    max_new_tokens = 1  # Right now we don't support multi-token outputs

    if max_new_tokens > 0:
        output_ids = model.generate(
            **{"input_ids": full_input_ids, "attention_mask": full_attention_mask},
            do_sample=False,
            max_new_tokens=max_new_tokens,
            past_key_values=past_key_values,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        output_ids = full_input_ids

    new_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
    answers = decode_predictions(new_ids, tokenizer)

    return answers


def get_multiple_task_vectors(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    num_vectors: int = 5,
    multi_context: bool = False,
) -> torch.Tensor:
    """
    Extract multiple task vectors for the same task using different demonstrations.
    Attempts to use non-overlapping examples when possible.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        task: The task to extract vectors for
        datasets: List of few-shot datasets
        num_vectors: Number of task vectors to extract
        multi_context: Whether to use multi-context extraction
        
    Returns:
        A tensor of shape (num_vectors, num_datasets, num_layers, hidden_size)
    """
    all_task_vectors = []
    
    # For each dataset, prepare all available examples upfront
    all_examples_by_dataset = []
    for dataset_idx, dataset in enumerate(datasets):
        # Get the test input to exclude from demonstrations
        test_input = dataset.test_input
        examples_per_vector = len(dataset.train_inputs)
        # Get all potential examples, excluding the test input
        total_available = task.num_examples() - 1  # -1 for test example
        # Don't try to sample more than available
        all_examples = task.sample_inputs(min(total_available, examples_per_vector * num_vectors), exclude=[test_input])
        all_examples_by_dataset.append(all_examples)
    
    # Now create vectors with as much disjoint sampling as possible
    for i in range(num_vectors):
        # Create new datasets with different demonstrations
        new_datasets = []
        
        for dataset_idx, dataset in enumerate(datasets):
            num_examples = len(dataset.train_inputs)
            available_examples = all_examples_by_dataset[dataset_idx]
            
            # If we have enough examples for fully disjoint sets, use them
            if len(available_examples) >= num_examples * num_vectors:
                start_idx = i * num_examples
                end_idx = start_idx + num_examples
                examples_to_use = available_examples[start_idx:end_idx]
            else:
                # Not enough for disjoint sets - randomly sample with different seed for each vector
                random.seed(i * 100 + 42)
                # Shuffle all available examples differently for each vector
                shuffled = available_examples.copy()
                random.shuffle(shuffled)
                examples_to_use = shuffled[:num_examples]
            
            # Create a dataset with the selected examples
            new_dataset = FewShotDataset(
                train_inputs=[str(x) for x in examples_to_use],
                train_outputs=[str(task.calc_output(x)) for x in examples_to_use],
                test_input=dataset.test_input,
                test_output=dataset.test_output
            )
            
            new_datasets.append(new_dataset)
        
        # Extract task vector with this set of demonstrations
        task_vector = get_task_hiddens(model, tokenizer, task, new_datasets, multi_context=multi_context)
        all_task_vectors.append(task_vector)
        
    return torch.stack(all_task_vectors)  # (num_vectors, num_datasets, num_layers, hidden_size)


def aggregate_task_vectors(task_vectors: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """
    Aggregate multiple task vectors using the specified method.
    
    Args:
        task_vectors: Tensor of shape (num_vectors, num_datasets, num_layers, hidden_size)
        method: Aggregation method ("mean" or "median")
        
    Returns:
        Aggregated task vectors of shape (num_datasets, num_layers, hidden_size)
    """
    if method == "mean":
        return torch.mean(task_vectors, dim=0)
    elif method == "median":
        return torch.median(task_vectors, dim=0).values
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def run_task_vector_with_fusion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    dev_datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
    multi_context: bool = False,
    num_vectors: int = 5,
):
    """
    Run task vector evaluation with vector fusion (aggregation of multiple vectors).
    """
    # Get standard task vector accuracy by layer (for comparison)
    dev_accuracy_by_layer = task_vector_accuracy_by_layer(
        model,
        tokenizer,
        task,
        dev_datasets,
        layers_to_test=layers_to_test,
        multi_context=multi_context,
    )
    best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))
    
    # Get multiple task vectors
    multiple_task_hiddens = get_multiple_task_vectors(
        model, 
        tokenizer, 
        task, 
        test_datasets, 
        num_vectors=num_vectors,
        multi_context=multi_context
    )
    
    # Get standard task vector (for comparison)
    standard_task_hiddens = get_task_hiddens(model, tokenizer, task, test_datasets, multi_context=multi_context)
    
    # Aggregate task vectors
    mean_task_hiddens = aggregate_task_vectors(multiple_task_hiddens, method="mean")
    # median_task_hiddens = aggregate_task_vectors(multiple_task_hiddens, method="median")
    
    # Generate predictions with standard task vector
    standard_predictions = modulated_generate(
        model,
        tokenizer,
        task,
        test_datasets,
        task_hiddens=standard_task_hiddens,
        intermediate_layer=best_intermediate_layer,
    )
    
    # Generate predictions with mean task vector
    mean_predictions = modulated_generate(
        model,
        tokenizer,
        task,
        test_datasets,
        task_hiddens=mean_task_hiddens,
        intermediate_layer=best_intermediate_layer,
    )
    
    # Generate predictions with median task vector
    # median_predictions = modulated_generate(
    #     model,
    #     tokenizer,
    #     task,
    #     test_datasets,
    #     task_hiddens=median_task_hiddens,
    #     intermediate_layer=best_intermediate_layer,
    # )
    
    # Calculate accuracies
    standard_accuracy = calculate_accuracy_on_datasets(task, standard_predictions, test_datasets)
    mean_accuracy = calculate_accuracy_on_datasets(task, mean_predictions, test_datasets)
    # median_accuracy = calculate_accuracy_on_datasets(task, median_predictions, test_datasets)
    
    return {
        "standard_predictions": standard_predictions,
        "mean_predictions": mean_predictions,
        # "median_predictions": median_predictions,
        "dev_accuracy_by_layer": dev_accuracy_by_layer,
        "best_intermediate_layer": best_intermediate_layer,
        "standard_accuracy": standard_accuracy,
        "mean_accuracy": mean_accuracy,
        # "median_accuracy": median_accuracy,
        "task_hiddens": {
            "standard": standard_task_hiddens,
            "mean": mean_task_hiddens,
            # "median": median_task_hiddens
        }
    }
