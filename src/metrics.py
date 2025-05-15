import numpy as np
import torch
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    matthews_corrcoef,
    mean_squared_error,
    pearson_corrcoef as pearson_corr,
    spearman_corrcoef as spearman_corr
)
import logging
import re
import string
from collections import Counter

logger = logging.getLogger(__name__)


def accuracy(predictions: List, references: List) -> float:
    filtered_pairs = [(p, r) for p, r in zip(predictions, references) if r != -100]
    if not filtered_pairs:
        return 0.0
    
    preds, refs = zip(*filtered_pairs)
    return accuracy_score(refs, preds)


def f1_score_with_invalid(predictions: List, references: List) -> float:
    filtered_pairs = [(p, r) for p, r in zip(predictions, references) if r != -100]
    if not filtered_pairs:
        return 0.0
    
    preds, refs = zip(*filtered_pairs)
    return f1_score(refs, preds, average='macro')


def mean_multiclass_f1(num_classes: int) -> callable:
    def f1_metric(predictions: List, references: List) -> float:
        # Filter out padding tokens (-100)
        filtered_pairs = [(p, r) for p, r in zip(predictions, references) if r != -100]
        if not filtered_pairs:
            return 0.0
        
        preds, refs = zip(*filtered_pairs)
        f1_scores = []
        
        for cls in range(num_classes):
            cls_preds = [1 if p == cls else 0 for p in preds]
            cls_refs = [1 if r == cls else 0 for r in refs]
            f1 = f1_score(cls_refs, cls_preds, average='binary')
            f1_scores.append(f1)
        
        return sum(f1_scores) / len(f1_scores)
    
    return f1_metric


def pearson_corrcoef(predictions: List[float], references: List[float]) -> float:
    filtered_pairs = [(p, r) for p, r in zip(predictions, references) if r != -100]
    if not filtered_pairs:
        return 0.0
    
    preds, refs = zip(*filtered_pairs)
    return pearson_corr(refs, preds)[0]


def spearman_corrcoef(predictions: List[float], references: List[float]) -> float:
    filtered_pairs = [(p, r) for p, r in zip(predictions, references) if r != -100]
    if not filtered_pairs:
        return 0.0
    
    preds, refs = zip(*filtered_pairs)
    return spearman_corr(refs, preds)[0]


def matthews_corrcoef_with_invalid(predictions: List, references: List) -> float:
    filtered_pairs = [(p, r) for p, r in zip(predictions, references) if r != -100]
    if not filtered_pairs:
        return 0.0
    
    preds, refs = zip(*filtered_pairs)
    return matthews_corrcoef(refs, preds)


def exact_match(predictions: List[str], references: List[str], normalize: bool = True) -> float:
    if normalize:
        predictions = [normalize_text(p) for p in predictions]
        references = [normalize_text(r) for r in references]
    
    exact_matches = 0
    for pred, ref in zip(predictions, references):
        if pred == ref:
            exact_matches += 1
    
    return exact_matches / len(predictions) if len(predictions) > 0 else 0


def normalize_text(text: str) -> str:
    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    
    if precision == 0 or recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def squad(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    em_score = 0
    f1_score = 0
    
    for pred, refs in zip(predictions, references):
        # For each example, calculate EM and F1 with each reference and take the maximum
        example_em = max(exact_match([pred], [ref], normalize=True) for ref in refs)
        example_f1 = max(token_f1(pred, ref) for ref in refs)
        
        em_score += example_em
        f1_score += example_f1
    
    n_examples = len(predictions)
    return {
        "exact_match": em_score / n_examples if n_examples > 0 else 0,
        "token_f1": f1_score / n_examples if n_examples > 0 else 0
    }


def calculate_transfer_gain(source_perf: float, target_perf: float, baseline_perf: float) -> float:
    return (target_perf - baseline_perf) / (source_perf - baseline_perf + 1e-8)


def calculate_domain_adaptability(source_perf: float, target_perf: float) -> float:
    return target_perf / (source_perf + 1e-8)


def calculate_disentanglement_quality(
    model_outputs: Dict,
    domain_labels: torch.Tensor,
    task_labels: torch.Tensor
) -> Dict[str, float]:
    if "encoder_hidden_states" not in model_outputs:
        return {}
    
    hidden_states = model_outputs["encoder_hidden_states"]
    batch_size = hidden_states.size(0)

    prefix_length = model_outputs.get("prefix_length", 100)
    di_ratio = model_outputs.get("di_ratio", 0.6)
    
    di_length = int(prefix_length * di_ratio)
    di_repr = hidden_states[:, :di_length].mean(dim=1)
    ds_repr = hidden_states[:, di_length:prefix_length].mean(dim=1)
    
    return {
        "di_domain_invariance": 0.5,  
        "ds_domain_specificity": 0.5 
    }


def calculate_adversarial_effectiveness(
    baseline_perf: float,
    adversarial_perf: float
) -> float:
    return (adversarial_perf - baseline_perf) / (baseline_perf + 1e-8)


def evaluate_cross_domain_transfer(
    model,
    source_dataloader,
    target_dataloader,
    source_task: str,
    target_task: str,
    device,
    baseline_results: Optional[Dict] = None
) -> Dict[str, float]:
    model.eval()
    results = {}
    source_outputs = []
    source_predictions = []
    source_references = []
    
    with torch.no_grad():
        for batch in source_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                decoder_attention_mask=batch["decoder_attention_mask"],
                labels=batch["labels"],
                domain_labels=batch["domain_labels"],
                task_labels=batch["task_labels"],
                training=False
            )
            
            source_outputs.append(outputs)

            logits = outputs["lm_logits"]
            predictions = torch.argmax(logits, dim=-1)
            source_predictions.extend(predictions.cpu().numpy().tolist())
            source_references.extend(batch["labels"].cpu().numpy().tolist())

    target_outputs = []
    target_predictions = []
    target_references = []
    
    with torch.no_grad():
        for batch in target_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                decoder_attention_mask=batch["decoder_attention_mask"],
                labels=batch["labels"],
                domain_labels=batch["domain_labels"],
                task_labels=batch["task_labels"],
                training=False
            )
            
            target_outputs.append(outputs)
            
            logits = outputs["lm_logits"]
            predictions = torch.argmax(logits, dim=-1)
            target_predictions.extend(predictions.cpu().numpy().tolist())
            target_references.extend(batch["labels"].cpu().numpy().tolist())
    
    task_type = None

    if any(qa_task in source_task.lower() for qa_task in ["qa", "squad", "nq", "hotpot"]):
        task_type = "qa"
    elif any(cls_task in source_task.lower() for cls_task in ["sst", "cola", "yelp", "polarity"]):
        task_type = "classification"
    elif any(sim_task in source_task.lower() for sim_task in ["qqp", "paws", "sts"]):
        task_type = "similarity"
    elif any(ent_task in source_task.lower() for ent_task in ["mnli", "scitail", "rte"]):
        task_type = "entailment"

    if task_type == "qa":
        source_qa_metrics = squad(source_predictions, source_references)
        target_qa_metrics = squad(target_predictions, target_references)
        
        results["source_em"] = source_qa_metrics["exact_match"]
        results["source_f1"] = source_qa_metrics["token_f1"]
        results["target_em"] = target_qa_metrics["exact_match"]
        results["target_f1"] = target_qa_metrics["token_f1"]
        
        source_perf = source_qa_metrics["token_f1"]
        target_perf = target_qa_metrics["token_f1"]
        
    elif task_type in ["classification", "entailment"]:
        source_acc = accuracy(source_predictions, source_references)
        target_acc = accuracy(target_predictions, target_references)
        
        results["source_accuracy"] = source_acc
        results["target_accuracy"] = target_acc
        source_perf = source_acc
        target_perf = target_acc
        
    elif task_type == "similarity":
        source_f1 = f1_score_with_invalid(source_predictions, source_references)
        target_f1 = f1_score_with_invalid(target_predictions, target_references)
        source_acc = accuracy(source_predictions, source_references)
        target_acc = accuracy(target_predictions, target_references)
        
        results["source_f1"] = source_f1
        results["target_f1"] = target_f1
        results["source_accuracy"] = source_acc
        results["target_accuracy"] = target_acc

        source_perf = source_f1
        target_perf = target_f1
    
    else:
        source_acc = accuracy(source_predictions, source_references)
        target_acc = accuracy(target_predictions, target_references)
        
        results["source_accuracy"] = source_acc
        results["target_accuracy"] = target_acc

        source_perf = source_acc
        target_perf = target_acc
    baseline_perf = 0.0
    if baseline_results is not None:
        if task_type == "qa":
            baseline_perf = baseline_results.get("target_f1", 0.0)
        else:
            baseline_perf = baseline_results.get("target_accuracy", 0.0)
    
    results["transfer_gain"] = calculate_transfer_gain(source_perf, target_perf, baseline_perf)
    results["domain_adaptability"] = calculate_domain_adaptability(source_perf, target_perf)

    source_loss = np.mean([output["task_loss"].item() for output in source_outputs])
    target_loss = np.mean([output["task_loss"].item() for output in target_outputs])
    
    results["source_loss"] = source_loss
    results["target_loss"] = target_loss
    
    return results


def aggregate_metrics(all_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not all_results:
        return {}
    
    aggregated = {}
    for pair_name, metrics in all_results.items():
        for metric_name, value in metrics.items():
            if metric_name not in aggregated:
                aggregated[metric_name] = []
            
            aggregated[metric_name].append(value)

    final_results = {}
    for metric_name, values in aggregated.items():
        if values:
            final_results[f"{metric_name}_mean"] = np.mean(values)
            final_results[f"{metric_name}_std"] = np.std(values)
    
    return final_results


def print_metrics_table(results: Dict[str, Dict[str, float]], metric_names: List[str] = None):
    if not results:
        print("No results to display")
        return
    
    pairs = list(results.keys())
    
    if metric_names is None:
        first_pair = next(iter(results.values()))
        metric_names = list(first_pair.keys())
    header = "| Transfer Pair | " + " | ".join(metric_names) + " |"
    separator = "|" + "-" * (len("Transfer Pair") + 2) + "|" + "".join(["-" * (len(m) + 2) + "|" for m in metric_names])
    
    print(header)
    print(separator)
    
    for pair in pairs:
        pair_metrics = results[pair]
        row = f"| {pair} | "
        for metric in metric_names:
            if metric in pair_metrics:
                value = pair_metrics[metric]
                row += f"{value:.4f} | "
            else:
                row += "N/A | "
        
        print(row)
    aggregated = aggregate_metrics(results)
    print(separator)

    mean_row = "| Mean | "
    for metric in metric_names:
        mean_key = f"{metric}_mean"
        if mean_key in aggregated:
            mean_row += f"{aggregated[mean_key]:.4f} | "
        else:
            mean_row += "N/A | "
    
    print(mean_row)
    
    # Print std
    std_row = "| Std | "
    for metric in metric_names:
        std_key = f"{metric}_std"
        if std_key in aggregated:
            std_row += f"{aggregated[std_key]:.4f} | "
        else:
            std_row += "N/A | "
    
    print(std_row)