import os
import torch
from arpo_model import ARPOModel
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import shutil
import datetime

logger = logging.getLogger(__name__)


def save_model(
    model: ARPOModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: Optional[Dict] = None,
    args: Optional[Dict] = None,
    output_dir: str = None,
    filename: str = None,
    is_best: bool = False,
    current_strategy: Optional[Dict] = None,
    save_optimizer: bool = True,
):
    if output_dir is None:
        raise ValueError("output_dir must be specified")
    
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": args,
    }
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    if save_optimizer and optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if save_optimizer and scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if current_strategy is not None:
        checkpoint["current_strategy"] = current_strategy

    if filename is None:
        filename = f"checkpoint_epoch_{epoch}_step_{global_step}.pt"

    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = os.path.join(output_dir, "best_model.pt")
        shutil.copyfile(checkpoint_path, best_path)
        best_meta = {
            "epoch": epoch,
            "global_step": global_step,
            "filename": filename,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        if metrics is not None:
            best_meta["metrics"] = metrics
        
        with open(os.path.join(output_dir, "best_model_meta.json"), "w") as f:
            json.dump(best_meta, f, indent=2)
    
    logger.info(f"Model saved to {checkpoint_path}")
    return checkpoint_path


def load_model(
    model_path: str,
    model: Optional[ARPOModel] = None,
    model_args: Optional[Dict] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
    load_optimizer: bool = True,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    if model is None:
        if model_args is None:
            if "args" in checkpoint:
                model_args = checkpoint["args"]
            else:
                raise ValueError("Either model or model_args must be provided")

        logger.info("Initializing new ARPOModel instance")
        model = ARPOModel(
            base_model_name=model_args.get("model_name", "t5-base"),
            prefix_length=model_args.get("prefix_length", 100),
            prefix_dim=model_args.get("prefix_dim", 768),
            di_ratio=model_args.get("di_ratio", 0.6),
            beta1=model_args.get("beta1", 0.1),
            beta2=model_args.get("beta2", 0.5),
            lambda0=model_args.get("lambda0", 1.0),
            lambda1=model_args.get("lambda1", 1.0),
            lambda2=model_args.get("lambda2", 1.0),
            lambda3=model_args.get("lambda3", 1.0),
            lambda4=model_args.get("lambda4", 1.0),
            lambda5=model_args.get("lambda5", 1.0),
            temp=model_args.get("temp", 0.07),
            alpha=model_args.get("alpha", 0.5),
            theta0=model_args.get("theta0", 0.5),
            gamma=model_args.get("gamma", 0.1),
            window_size=model_args.get("window_size", 10),
            device=device
        )

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.to(device)
    
    if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
        logger.info("Loading optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    if load_optimizer and scheduler is not None and "scheduler_state_dict" in checkpoint:
        logger.info("Loading scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    metrics = checkpoint.get("metrics", {})
    args = checkpoint.get("args", {})
    current_strategy = checkpoint.get("current_strategy", None)
    
    logger.info(f"Loaded model from epoch {epoch}, global step {global_step}")
    
    return model, optimizer, scheduler, {
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
        "args": args,
        "current_strategy": current_strategy
    }


def resume_training(
    model_path: str,
    output_dir: str,
    device: Optional[torch.device] = None,
    update_output_dir: bool = True
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading checkpoint from {model_path} for resuming training")
    checkpoint = torch.load(model_path, map_location=device)
    
    if "args" not in checkpoint:
        raise ValueError("Checkpoint does not contain args, cannot resume training")

    args = checkpoint["args"]

    if update_output_dir and output_dir is not None:
        args.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

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

    model.load_state_dict(checkpoint["model_state_dict"])
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
    
    # Load optimizer state if available
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_steps if hasattr(args, "lr_decay_steps") else 1000,
        gamma=args.lr_decay_rate if hasattr(args, "lr_decay_rate") else 0.9
    )
    
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    checkpoint_info = {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "metrics": checkpoint.get("metrics", {}),
        "current_strategy": checkpoint.get("current_strategy", None)
    }
    
    logger.info(f"Successfully prepared for resuming training from epoch {checkpoint_info['epoch']}, step {checkpoint_info['global_step']}")
    
    return model, optimizer, scheduler, args, checkpoint_info


def find_best_checkpoint(model_dir: str, metric: str = "target_loss", higher_better: bool = False):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Check if best model metadata exists
    meta_path = os.path.join(model_dir, "best_model_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        best_path = os.path.join(model_dir, "best_model.pt")
        if os.path.exists(best_path):
            logger.info(f"Found best model at {best_path}")
            return best_path
    
    checkpoints = []
    for filename in os.listdir(model_dir):
        if filename.endswith(".pt") and filename.startswith("checkpoint_"):
            checkpoint_path = os.path.join(model_dir, filename)
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if "metrics" in checkpoint and metric in checkpoint["metrics"]:
                    checkpoints.append({
                        "path": checkpoint_path,
                        "metric": checkpoint["metrics"][metric],
                        "epoch": checkpoint.get("epoch", 0),
                        "global_step": checkpoint.get("global_step", 0)
                    })
            except Exception as e:
                logger.warning(f"Error loading checkpoint {checkpoint_path}: {e}")
    
    if not checkpoints:
        raise ValueError(f"No valid checkpoints found in {model_dir}")

    if higher_better:
        best_checkpoint = max(checkpoints, key=lambda x: x["metric"])
    else:
        best_checkpoint = min(checkpoints, key=lambda x: x["metric"])
    
    logger.info(f"Found best checkpoint at {best_checkpoint['path']} with {metric}={best_checkpoint['metric']}")
    
    return best_checkpoint["path"]


def export_model_for_inference(
    model_path: str,
    export_dir: str,
    format: str = "pytorch",
    device: Optional[torch.device] = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, _, _, checkpoint_info = load_model(
        model_path=model_path,
        device=device,
        load_optimizer=False
    )
    os.makedirs(export_dir, exist_ok=True)
    
    if format == "pytorch":
        export_path = os.path.join(export_dir, "model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": checkpoint_info.get("args", {}),
            "current_strategy": checkpoint_info.get("current_strategy", None)
        }, export_path)
    
    elif format == "torchscript":
        model.eval()
        batch_size = 1
        seq_len = 128
        dummy_input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        dummy_attention_mask = torch.ones((batch_size, seq_len), device=device)
        dummy_decoder_input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        dummy_decoder_attention_mask = torch.ones((batch_size, seq_len), device=device)

        dummy_inputs = {
            "input_ids": dummy_input_ids,
            "attention_mask": dummy_attention_mask,
            "decoder_input_ids": dummy_decoder_input_ids,
            "decoder_attention_mask": dummy_decoder_attention_mask,
            "training": False
        }
        
        try:
            scripted_model = torch.jit.script(model)
            
            export_path = os.path.join(export_dir, "model.torchscript")
            scripted_model.save(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting to TorchScript: {e}")
            logger.info("Falling back to PyTorch format")
            
            export_path = os.path.join(export_dir, "model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": checkpoint_info.get("args", {}),
                "current_strategy": checkpoint_info.get("current_strategy", None)
            }, export_path)
    
    elif format == "onnx":
        try:
            import onnx
            import onnxruntime
            
            model.eval()
            batch_size = 1
            seq_len = 128
            dummy_input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
            dummy_attention_mask = torch.ones((batch_size, seq_len), device=device)
            dummy_decoder_input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
            dummy_decoder_attention_mask = torch.ones((batch_size, seq_len), device=device)
            
            def forward_for_export(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    training=False
                )
                return outputs["lm_logits"]

            export_path = os.path.join(export_dir, "model.onnx")
            torch.onnx.export(
                model,
                (dummy_input_ids, dummy_attention_mask, dummy_decoder_input_ids, dummy_decoder_attention_mask),
                export_path,
                input_names=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "seq_len"},
                    "attention_mask": {0: "batch_size", 1: "seq_len"},
                    "decoder_input_ids": {0: "batch_size", 1: "seq_len"},
                    "decoder_attention_mask": {0: "batch_size", 1: "seq_len"},
                    "logits": {0: "batch_size", 1: "seq_len"}
                },
                opset_version=12
            )
            
        except Exception as e:
            logger.error(f"Error exporting to ONNX: {e}")
            logger.info("Falling back to PyTorch format")
            
            export_path = os.path.join(export_dir, "model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": checkpoint_info.get("args", {}),
                "current_strategy": checkpoint_info.get("current_strategy", None)
            }, export_path)
    
    else:
        raise ValueError(f"Unsupported export format: {format}")

    config_path = os.path.join(export_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_type": "arpo",
            "base_model_name": checkpoint_info.get("args", {}).get("model_name", "t5-base"),
            "prefix_length": model.prefix_length,
            "prefix_dim": model.prefix_dim,
            "di_ratio": model.di_length / model.prefix_length if model.prefix_length > 0 else 0.6,
            "export_format": format,
            "export_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    logger.info(f"Model exported to {export_path}")
    return export_path


def load_model_for_inference(
    model_path: str,
    device: Optional[torch.device] = None,
    model_args: Optional[Dict] = None
):
    if os.path.isdir(model_path):
        for filename in ["model.pt", "best_model.pt"]:
            possible_path = os.path.join(model_path, filename)
            if os.path.exists(possible_path):
                model_path = possible_path
                break
        else:
            try:
                model_path = find_best_checkpoint(model_path)
            except Exception as e:
                raise FileNotFoundError(f"Could not find model file in directory {model_path}: {e}")
    
    model, _, _, _ = load_model(
        model_path=model_path,
        model_args=model_args,
        device=device,
        load_optimizer=False
    )
    model.eval()
    
    return model


def create_model_archive(
    model_path: str,
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
    include_optimizer: bool = False
):
    import zipfile
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = torch.load(model_path, map_location="cpu")

        if not include_optimizer and "optimizer_state_dict" in checkpoint:
            clean_checkpoint = {k: v for k, v in checkpoint.items() if k != "optimizer_state_dict"}
            clean_model_path = os.path.join(temp_dir, "model.pt")
            torch.save(clean_checkpoint, clean_model_path)
        else:
            clean_model_path = os.path.join(temp_dir, "model.pt")
            shutil.copy(model_path, clean_model_path)

        if config_path is not None and os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(temp_dir, "config.json"))
        elif "args" in checkpoint:
            config_path = os.path.join(temp_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(checkpoint["args"], f, indent=2)
        
        metadata = {
            "model_type": "arpo",
            "base_model_name": checkpoint.get("args", {}).get("model_name", "t5-base"),
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "includes_optimizer": include_optimizer
        }
        
        with open(os.path.join(temp_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        with open(os.path.join(temp_dir, "README.md"), "w") as f:
            f.write("# ARPO Model Archive\n\n")
            f.write("This archive contains an ARPO model for cross-domain transfer learning.\n\n")
            f.write("## Files\n\n")
            f.write("- `model.pt`: PyTorch model checkpoint\n")
            f.write("- `config.json`: Model configuration\n")
            f.write("- `metadata.json`: Archive metadata\n\n")
            f.write("## Usage\n\n")
            f.write("```python\n")
            f.write("from save_and_load import load_model_for_inference\n\n")
            f.write("# Load model\n")
            f.write("model = load_model_for_inference('path/to/extracted/archive')\n")
            f.write("```\n")
        
        if output_path is None:
            output_path = f"arpo_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
    
    logger.info(f"Model archive created at {output_path}")
    return output_path