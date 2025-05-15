import os
import argparse
import torch
import numpy as np
import json
import logging
from transformers import T5Tokenizer
from dataloader import ARPODataLoader
from arpo_model import ARPOModel
from metrics import evaluate_cross_domain_transfer, print_metrics_table, aggregate_metrics
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def run_cross_domain_evaluation(args):
    """Run cross-domain evaluation on trained models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    data_loader = ARPODataLoader(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        setting=args.setting,
        batch_size=args.batch_size,
        max_source_length=args.max_seq_length,
        max_target_length=args.max_target_length,
        train_n_obs=None, 
        val_n_obs=None,
        test_n_obs=None,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    all_dataloaders = data_loader.get_all_dataloaders()
    all_results = {}

    results_file = os.path.join(args.model_dir, f"evaluation_{args.setting}.json")
    if os.path.exists(results_file) and not args.overwrite:
        with open(results_file, "r") as f:
            all_results = json.load(f)
        logger.info(f"Loaded existing results from {results_file}")
    
    for pair_name, loaders in all_dataloaders.items():
        logger.info(f"Evaluating {pair_name}")
        source_task = loaders["source_task"]
        target_task = loaders["target_task"]
        
        if pair_name in all_results and not args.overwrite:
            logger.info(f"Skipping {pair_name} as it was already evaluated")
            continue
        
        model_path = os.path.join(args.model_dir, f"best_model_{pair_name.replace('→', '_to_')}.pt")
        if not os.path.exists(model_path):
            logger.warning(f"Model for {pair_name} not found at {model_path}, skipping...")
            continue

        model = ARPOModel(
            base_model_name=args.model_name,
            prefix_length=args.prefix_length,
            prefix_dim=args.prefix_dim,
            di_ratio=args.di_ratio,
            device=device
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        # Evaluate transfer performance
        results = evaluate_cross_domain_transfer(
            model=model,
            source_dataloader=loaders["source_val"],
            target_dataloader=loaders["target_test"],
            source_task=source_task,
            target_task=target_task,
            device=device
        )
        
        all_results[pair_name] = results

        logger.info(f"Results for {pair_name}:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")

    if args.model_dir:
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved evaluation results to {results_file}")
    
    logger.info("Cross-domain transfer evaluation results:")
    print_metrics_table(
        all_results, 
        metric_names=["target_accuracy", "target_f1", "domain_adaptability", "transfer_gain"]
    )

    aggregated = aggregate_metrics(all_results)
    logger.info("Aggregated metrics:")
    for metric, value in aggregated.items():
        logger.info(f"  {metric}: {value:.4f}")

    if args.plot_results:
        plot_cross_domain_results(all_results, args.setting, args.model_dir)
    
    return all_results


def plot_cross_domain_results(results, setting, output_dir):
    if not results:
        logger.warning("No results to plot")
        return

    metrics = ["target_accuracy", "target_f1", "domain_adaptability", "transfer_gain"]
    available_metrics = []
    
    first_pair = next(iter(results.values()))
    for metric in metrics:
        if metric in first_pair:
            available_metrics.append(metric)
    
    if not available_metrics:
        logger.warning("No metrics available for plotting")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for metric in available_metrics:
        plt.figure(figsize=(10, 6))

        pairs = list(results.keys())
        values = [results[pair].get(metric, 0) for pair in pairs]
        

        plt.bar(pairs, values)
        plt.xlabel("Transfer Pair")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} for {setting}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, f"{setting}_{metric}.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved plot to {plot_path}")
    
    plt.figure(figsize=(12, 8))
    
    pairs = list(results.keys())
    x = np.arange(len(pairs))
    width = 0.2
    offset = -width * (len(available_metrics) - 1) / 2
    
    for i, metric in enumerate(available_metrics):
        values = [results[pair].get(metric, 0) for pair in pairs]
        plt.bar(x + offset + i * width, values, width, label=metric.replace("_", " ").title())
    
    plt.xlabel("Transfer Pair")
    plt.ylabel("Score")
    plt.title(f"Combined Metrics for {setting}")
    plt.xticks(x, pairs, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    combined_plot_path = os.path.join(plots_dir, f"{setting}_combined.png")
    plt.savefig(combined_plot_path)
    plt.close()
    
    logger.info(f"Saved combined plot to {combined_plot_path}")


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the data files")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing trained models")
    parser.add_argument("--setting", type=str, default="setting1", choices=["setting1", "setting2"],
                        help="Cross-domain transfer setting to evaluate")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length for source")
    parser.add_argument("--max_target_length", type=int, default=128,
                        help="Maximum sequence length for target")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for dataloader")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument("--model_name", type=str, default="t5-base",
                        help="T5 model name")
    parser.add_argument("--prefix_length", type=int, default=100,
                        help="Length of prefix")
    parser.add_argument("--prefix_dim", type=int, default=768,
                        help="Dimension of prefix")
    parser.add_argument("--di_ratio", type=float, default=0.6,
                        help="Ratio of domain-invariant prefix length to total prefix length")

    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing evaluation results")
    parser.add_argument("--plot_results", action="store_true",
                        help="Plot evaluation results")
    
    args = parser.parse_args()

    _ = run_cross_domain_evaluation(args)


if __name__ == "__main__":
    main()