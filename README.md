This repository implements ARPO, a novel approach for cross-domain transfer learning that leverages disentangled prefix learning and adaptive adversarial strategies to improve knowledge transfer between domains.

## Overview
ARPO introduces three key innovations:

1. We propose a clear **domain-information disentanglement strategy** using Information Bottleneck constraints, effectively mitigating negative transfer caused by domain-specific features; 

2. We design a **dynamic threshold-based adversarial training mechanism** to prevent premature interference in primary task training, significantly improving training stability; 

3. We introduce an **automated MOBO-based search mechanism** with adaptive dimensionality reduction to efficiently optimize complex, high-dimensional parameter spaces, substantially reducing manual tuning costs and enhancing model performance in realistic multi-domain scenarios.

## File Structure

- **Core Model Files**:
  - `arpo_model.py` - Implements the ARPO model architecture with all three key components
  - `train.py` - Basic training script for a single domain transfer experiment
  - `run_experiment.py` - Comprehensive experiment runner for multiple domain pairs

- **Data Processing**:
  - `tasks.py` - Defines task-specific data processing for various NLP tasks
  - `dataloader.py` - Implements data loading and preprocessing for cross-domain transfer

- **Evaluation**:
  - `metrics.py` - Implements evaluation metrics for cross-domain transfer
  - `evaluate.py` - Script for evaluating trained models

- **Experiment Management**:
  - `config.py` - Centralized configuration management system
  - `save_and_load.py` - Model saving, loading, and exporting utilities

## Dependencies

```
torch>=1.10.0
transformers>=4.21.0
datasets>=2.4.0
botorch>=0.6.0
numpy>=1.20.0
tqdm>=4.62.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
gpytorch>=1.6.0
```

## Running Experiments

### Basic Workflow

The general experimental workflow follows these steps:

1. **Configure the experiment**
   ```python
   from config import ARPOConfig

   config = ARPOConfig(
       profile="full_arpo",   
       setting="setting1",  
       data_dir="data/",   
       output_dir="output/"  
   )
   ```

2. **Run the experiment**
   ```python
   from run_experiment import run_experiment
   import argparse
   
   # Convert config to args
   args = argparse.Namespace(**config.get_args_dict())
   
   # Run experiment
   results = run_experiment(args)
   ```

3. **Evaluate results**
   ```python
   from evaluate import run_cross_domain_evaluation
   
   # Evaluate results
   eval_results = run_cross_domain_evaluation(args)
   ```

## Cross-Domain Transfer Settings

The framework supports two main experimental settings:

### Setting 1 Comparison of methods evaluated on cross-domain transfer on same tasks
- NQ → SearchQA (Question Answering)
- Yelp Polarity → SciTail (Sentiment/Entailment)
- NewsQA → HotpotQA (Question Answering)
- MNLI → QQP (Entailment/Paraphrase)
- CoLA → PAWS (Grammar/Paraphrase)

### Setting 2 Comparison of methods evaluated on cross-domain transfer on different tasks
- CoLA → QQP (Grammar/Paraphrase)
- RTE → SST-2 (Entailment/Sentiment)
- BoolQ → NQ (QA/QA)
- HotpotQA → SciTail (QA/Entailment)
- MNLI → SearchQA (Entailment/QA)

## Configuration System

The `config.py` file provides a comprehensive configuration system:

- **Experiment Profiles**: Predefined configurations like "baseline", "light_arpo", "full_arpo"
- **Model Configurations**: Settings for different model sizes (t5-base, t5-large)
- **Training Configurations**: Different learning rates and batch sizes
- **Disentanglement Configurations**: Various settings for prefix learning
- **Adversarial Configurations**: Different adversarial training strategies
