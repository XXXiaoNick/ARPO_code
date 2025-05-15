import os
import argparse
import torch
import logging
from transformers import T5Tokenizer
from dataloader import ARPODataLoader
from arpo_model import ARPOModel
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def evaluate_model(model, dataloader, device, task_name=None):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating on {task_name}" if task_name else "Evaluating"):
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
            
            loss = outputs["task_loss"]
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_samples += batch["input_ids"].size(0)
            
            logits = outputs["lm_logits"]
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
    
    avg_loss = total_loss / total_samples

    if hasattr(dataloader.dataset, 'task_id') and dataloader.dataset.task_id == 1:  # Classification
        correct = 0
        total = 0
        for pred, label in zip(all_predictions, all_labels):
            if label != -100: 
                if pred == label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        return {"eval_loss": avg_loss, "accuracy": accuracy}
    
    return {"eval_loss": avg_loss}


def run_experiment(args):
    """Run cross-domain transfer experiment"""
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
        train_n_obs=args.train_n_obs,
        val_n_obs=args.val_n_obs,
        test_n_obs=args.test_n_obs,
        num_workers=args.num_workers,
        seed=args.seed,
        source_domain_ratio=args.source_domain_ratio,
        include_target_train=args.include_target_train
    )
    
    all_dataloaders = data_loader.get_all_dataloaders()
 
    results = {}
    
    for pair_name, loaders in all_dataloaders.items():
        logger.info(f"Running experiment for {pair_name}")
        source_task = loaders["source_task"]
        target_task = loaders["target_task"]
        
        model = ARPOModel(
            base_model_name=args.model_name,
            prefix_length=args.prefix_length,
            prefix_dim=args.prefix_dim,
            di_ratio=args.di_ratio,
            beta1=args.beta1,
            beta2=args.beta2,
            lambda0=args.lambda0,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            lambda3=args.lambda3,
            lambda4=args.lambda4,
            lambda5=args.lambda5,
            temp=args.temp,
            alpha=args.alpha,
            theta0=args.theta0,
            gamma=args.gamma,
            window_size=args.window_size,
            device=device
        )
        
        model.to(device)
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )

        num_training_steps = len(loaders["train"]) * args.num_epochs

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_decay_steps,
            gamma=args.lr_decay_rate
        )

        best_target_val_loss = float('inf')
        best_target_test_metrics = None

        current_strategy = {
            'token_fgsm': args.use_fgsm,
            'token_fgsm_strength': args.fgsm_strength,
            'token_pgd': args.use_pgd,
            'token_pgd_strength': args.pgd_strength,
            'token_pgd_steps': args.pgd_steps,
            'phrase_swap_a': args.use_phrase_swap,
            'phrase_swap_a_prob': args.phrase_swap_prob,
            'phrase_swap_a_strength': args.phrase_swap_strength,
            'task_cross_domain': args.use_task_cross_domain,
            'task_cross_domain_prob': args.task_cross_domain_prob,
            'task_cross_domain_strength': args.task_cross_domain_strength,
            'threshold': args.theta0,
            'alpha': args.alpha
        }

        logger.info(f"Starting training for {pair_name}")
        global_step = 0
        
        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0
            epoch_task_loss = 0
            epoch_disent_loss = 0
            epoch_adv_loss = 0

            progress_bar = tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{args.num_epochs}")
            
            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                    labels=batch["labels"],
                    domain_labels=batch["domain_labels"],
                    task_labels=batch["task_labels"],
                    strategy=current_strategy,
                    training=True
                )

                loss = outputs["total_loss"]

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "task_loss": outputs["task_loss"].item() if outputs["task_loss"] is not None else 0,
                    "disent_loss": outputs["disent_loss"].item() if outputs["disent_loss"] is not None else 0,
                    "adv_loss": outputs["adv_loss"].item() if outputs["adv_loss"] is not None else 0,
                    "threshold": outputs["dynamic_threshold"] if outputs["dynamic_threshold"] is not None else 0,
                    "improvement": outputs["loss_improvement"] if outputs["loss_improvement"] is not None else 0
                })

                epoch_loss += loss.item()
                epoch_task_loss += outputs["task_loss"].item() if outputs["task_loss"] is not None else 0
                epoch_disent_loss += outputs["disent_loss"].item() if outputs["disent_loss"] is not None else 0
                epoch_adv_loss += outputs["adv_loss"].item() if outputs["adv_loss"] is not None else 0
                
                global_step += 1

                if args.use_bayesian_opt and global_step % args.bo_update_steps == 0:
                    performance_metrics = {
                        "task_accuracy": 1.0 - outputs["task_loss"].item(),  # Approximate accuracy from loss
                        "adversarial_robustness": outputs["adv_loss"].item() if outputs["adv_loss"] is not None else 0,
                        "computational_cost": 1.0  # Placeholder for actual computational cost
                    }

                    new_strategy = model.update_bayesian_optimization(
                        current_strategy, performance_metrics
                    )
                    
                    if new_strategy is not None:
                        logger.info(f"Updated strategy: {new_strategy}")
                        current_strategy = new_strategy
            
            avg_loss = epoch_loss / len(loaders["train"])
            avg_task_loss = epoch_task_loss / len(loaders["train"])
            avg_disent_loss = epoch_disent_loss / len(loaders["train"])
            avg_adv_loss = epoch_adv_loss / len(loaders["train"])
            
            logger.info(f"Epoch {epoch+1}/{args.num_epochs} - "
                    f"Loss: {avg_loss:.4f}, "
                    f"Task Loss: {avg_task_loss:.4f}, "
                    f"Disent Loss: {avg_disent_loss:.4f}, "
                    f"Adv Loss: {avg_adv_loss:.4f}")

            logger.info("Evaluating on source validation set")
            source_val_results = evaluate_model(model, loaders["source_val"], device, source_task)
            
            logger.info("Evaluating on target validation set")
            target_val_results = evaluate_model(model, loaders["target_val"], device, target_task)
            
            logger.info(f"Source Val Loss: {source_val_results['eval_loss']:.4f}")
            logger.info(f"Target Val Loss: {target_val_results['eval_loss']:.4f}")
            
            if 'accuracy' in source_val_results:
                logger.info(f"Source Val Accuracy: {source_val_results['accuracy']:.4f}")
            
            if 'accuracy' in target_val_results:
                logger.info(f"Target Val Accuracy: {target_val_results['accuracy']:.4f}")
            
            if target_val_results["eval_loss"] < best_target_val_loss:
                best_target_val_loss = target_val_results["eval_loss"]
                logger.info(f"New best target validation loss: {best_target_val_loss:.4f}")
                
                logger.info("Evaluating best model on target test set")
                target_test_results = evaluate_model(model, loaders["target_test"], device, target_task)
                best_target_test_metrics = target_test_results
                
                logger.info(f"Target Test Loss: {target_test_results['eval_loss']:.4f}")
                if 'accuracy' in target_test_results:
                    logger.info(f"Target Test Accuracy: {target_test_results['accuracy']:.4f}")

                if args.output_dir:
                    os.makedirs(args.output_dir, exist_ok=True)
                    model_save_path = os.path.join(args.output_dir, f"best_model_{pair_name.replace('→', '_to_')}.pt")

                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_target_val_loss": best_target_val_loss,
                        "args": args,
                        "current_strategy": current_strategy
                    }, model_save_path)
                    
                    logger.info(f"Saved model to {model_save_path}")
        
        results[pair_name] = best_target_test_metrics
        
        logger.info(f"Final results for {pair_name}:")
        if best_target_test_metrics is not None:
            for metric_name, value in best_target_test_metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
    
    avg_results = {}
    for metric in ["eval_loss", "accuracy"]:
        if all(metric in result for result in results.values()):
            avg_results[metric] = np.mean([result[metric] for result in results.values()])
    
    logger.info("Average results across all pairs:")
    for metric_name, value in avg_results.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the data files")
    parser.add_argument("--setting", type=str, default="setting1", choices=["setting1", "setting2"],
                        help="Cross-domain transfer setting to run")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the model")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length for source")
    parser.add_argument("--max_target_length", type=int, default=128,
                        help="Maximum sequence length for target")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for dataloader")
    parser.add_argument("--train_n_obs", type=int, default=None,
                        help="Number of training observations to use")
    parser.add_argument("--val_n_obs", type=int, default=None,
                        help="Number of validation observations to use")
    parser.add_argument("--test_n_obs", type=int, default=None,
                        help="Number of test observations to use")
    parser.add_argument("--source_domain_ratio", type=float, default=0.7,
                        help="Ratio of source domain data in training")
    parser.add_argument("--include_target_train", action="store_true",
                        help="Include target domain data in training")
    
    parser.add_argument("--model_name", type=str, default="t5-base",
                        help="T5 model name")
    parser.add_argument("--prefix_length", type=int, default=100,
                        help="Length of prefix")
    parser.add_argument("--prefix_dim", type=int, default=768,
                        help="Dimension of prefix")
    parser.add_argument("--di_ratio", type=float, default=0.6,
                        help="Ratio of domain-invariant prefix length to total prefix length")
    
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for Adam optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--lr_decay_steps", type=int, default=1000,
                        help="Steps for learning rate decay")
    parser.add_argument("--lr_decay_rate", type=float, default=0.9,
                        help="Learning rate decay rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    parser.add_argument("--beta1", type=float, default=0.1,
                        help="Weight for task info in DI prefix")
    parser.add_argument("--beta2", type=float, default=0.5,
                        help="Weight for task info in DS prefix")
    parser.add_argument("--lambda0", type=float, default=1.0,
                        help="HSIC orthogonality weight")
    parser.add_argument("--lambda1", type=float, default=1.0,
                        help="Weight for IB loss of DI prefix")
    parser.add_argument("--lambda2", type=float, default=1.0,
                        help="Weight for IB loss of DS prefix")
    parser.add_argument("--lambda3", type=float, default=1.0,
                        help="Weight for orthogonality constraint")
    parser.add_argument("--lambda4", type=float, default=1.0,
                        help="Weight for contrastive loss")
    parser.add_argument("--lambda5", type=float, default=1.0,
                        help="Weight for conditional independence")
    parser.add_argument("--temp", type=float, default=0.07,
                        help="Temperature for contrastive loss")
    
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for adversarial loss")
    parser.add_argument("--theta0", type=float, default=0.5,
                        help="Base threshold for adversarial training")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Threshold sensitivity")
    parser.add_argument("--window_size", type=int, default=10,
                        help="Window size for loss statistics")
    
    parser.add_argument("--use_fgsm", action="store_true",
                        help="Use FGSM attack")
    parser.add_argument("--fgsm_strength", type=float, default=0.01,
                        help="Strength of FGSM attack")
    parser.add_argument("--use_pgd", action="store_true",
                        help="Use PGD attack")
    parser.add_argument("--pgd_strength", type=float, default=0.01,
                        help="Strength of PGD attack")
    parser.add_argument("--pgd_steps", type=int, default=3,
                        help="Number of PGD steps")
    parser.add_argument("--use_phrase_swap", action="store_true",
                        help="Use phrase swapping")
    parser.add_argument("--phrase_swap_prob", type=float, default=0.2,
                        help="Probability of phrase swapping")
    parser.add_argument("--phrase_swap_strength", type=float, default=0.1,
                        help="Strength of phrase swapping")
    parser.add_argument("--use_task_cross_domain", action="store_true",
                        help="Use task cross-domain swapping")
    parser.add_argument("--task_cross_domain_prob", type=float, default=0.1,
                        help="Probability of task cross-domain swapping")
    parser.add_argument("--task_cross_domain_strength", type=float, default=0.3,
                        help="Strength of task cross-domain swapping")
    
    parser.add_argument("--use_bayesian_opt", action="store_true",
                        help="Use Bayesian optimization for adversarial strategies")
    parser.add_argument("--bo_update_steps", type=int, default=100,
                        help="Number of steps between Bayesian optimization updates")
    
    args = parser.parse_args()
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    results = run_experiment(args)

    if args.output_dir:
        import json
        results_file = os.path.join(args.output_dir, f"results_{args.setting}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")


if __name__ == "__main__":
    main()