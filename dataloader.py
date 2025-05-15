import os
import json
import torch
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Dict, List, Tuple, Optional, Union
from transformers import T5Tokenizer
from tasks import AutoTask, TASK_MAPPING

logger = logging.getLogger(__name__)

DOMAIN_MAPPINGS = {
    "nq": 0,
    "squad": 1,
    "searchqa": 2,
    "hotpotqa": 3,
    "newsqa": 4,
    "yelp_polarity": 5,
    "scitail": 6,
    "mnli": 7,
    "qqp": 8,
    "cola": 9,
    "paws": 10,
    "rte": 11,
    "sst2": 12,
    "superglue-boolq": 13
}

TASK_MAPPINGS = {
    "qa": 0,            
    "classification": 1, 
    "similarity": 2,     
    "entailment": 3    
}

TASK_TO_TYPE = {
    "nq": "qa",
    "squad": "qa",
    "searchqa": "qa",
    "hotpotqa": "qa",
    "newsqa": "qa",
    "yelp_polarity": "classification",
    "scitail": "entailment",
    "mnli": "entailment",
    "qqp": "similarity",
    "cola": "classification",
    "paws": "similarity",
    "rte": "entailment",
    "sst2": "classification",
    "superglue-boolq": "classification"
}

TRANSFER_SETTINGS = {
    "setting1": [
        {"source": "nq", "target": "searchqa", "name": "NQ→SQA"},
        {"source": "yelp_polarity", "target": "scitail", "name": "Yelp→SciTail"},
        {"source": "newsqa", "target": "hotpotqa", "name": "News→HP"},
        {"source": "mnli", "target": "qqp", "name": "MNLI→QQP"},
        {"source": "cola", "target": "paws", "name": "CoLA→PAWS"}
    ],
    "setting2": [
        {"source": "cola", "target": "qqp", "name": "CoLA→QQP"},
        {"source": "rte", "target": "sst2", "name": "RTE→SST-2"},
        {"source": "superglue-boolq", "target": "nq", "name": "BoolQ→NQ"},
        {"source": "hotpotqa", "target": "scitail", "name": "HP→SciTail"},
        {"source": "mnli", "target": "searchqa", "name": "MNLI→SQA"}
    ]
}


class DomainTaskDataset(Dataset):
    def __init__(
        self,
        task_name: str,
        tokenizer: T5Tokenizer,
        data_dir: str,
        split: str = "train",
        max_source_length: int = 512,
        max_target_length: int = 128,
        domain_id: int = None,
        task_type: str = None,
        n_obs: int = None,
        seed: int = 42
    ):
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.split = split
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.domain_id = DOMAIN_MAPPINGS[task_name] if domain_id is None else domain_id
        task_type = TASK_TO_TYPE[task_name] if task_type is None else task_type
        self.task_id = TASK_MAPPINGS[task_type]
 
        self.task = AutoTask.get(task_name, config=None, seed=seed)
        self.data = self.load_data(n_obs=n_obs)
        
        logger.info(f"Loaded {len(self.data)} examples from {task_name} {split} split")
    
    def load_data(self, n_obs=None):
        """Load and preprocess the data"""
        task_dir = os.path.join(self.data_dir, self.task_name)
        os.makedirs(task_dir, exist_ok=True)

        cache_file = os.path.join(
            task_dir, 
            f"cached_{self.split}_{self.max_source_length}_{self.max_target_length}.pt"
        )
        
        if os.path.exists(cache_file):
            logger.info(f"Loading cached data from {cache_file}")
            dataset = torch.load(cache_file)
            if n_obs is not None and n_obs < len(dataset):
                dataset = self.task.subsample(dataset, n_obs=n_obs)
            return dataset

        logger.info(f"Loading {self.task_name} {self.split} data")
        dataset = self.task.get(
            split=self.split,
            add_prefix=True,
            n_obs=n_obs,
            split_validation_test=False
        )

        torch.save(dataset, cache_file)
        return dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get item by index"""
        example = self.data[idx]

        source_text = example["source"]
        target_text = example["target"]

        source_tokenized = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_tokenized = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": source_tokenized["input_ids"].squeeze(),
            "attention_mask": source_tokenized["attention_mask"].squeeze(),
            "decoder_input_ids": target_tokenized["input_ids"].squeeze(),
            "decoder_attention_mask": target_tokenized["attention_mask"].squeeze(),
            "labels": target_tokenized["input_ids"].squeeze(),
            "domain_labels": torch.tensor(self.domain_id, dtype=torch.long),
            "task_labels": torch.tensor(self.task_id, dtype=torch.long),
            "source_text": source_text,
            "target_text": target_text,
            "task_name": self.task_name
        }


class ARPODataLoader:
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        data_dir: str,
        setting: str = "setting1",
        batch_size: int = 16,
        max_source_length: int = 512,
        max_target_length: int = 128,
        train_n_obs: Optional[int] = None,
        val_n_obs: Optional[int] = None,
        test_n_obs: Optional[int] = None,
        num_workers: int = 4,
        seed: int = 42,
        source_domain_ratio: float = 0.7,
        include_target_train: bool = True
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.setting = setting
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.train_n_obs = train_n_obs
        self.val_n_obs = val_n_obs
        self.test_n_obs = test_n_obs
        self.num_workers = num_workers
        self.seed = seed
        self.source_domain_ratio = source_domain_ratio
        self.include_target_train = include_target_train

        self.transfer_pairs = TRANSFER_SETTINGS[setting]
        self._initialize_dataloaders()
    
    def _initialize_dataloaders(self):
        """Initialize datasets and dataloaders for each transfer pair"""
        self.dataloaders = {}
        
        for pair in self.transfer_pairs:
            source_task = pair["source"]
            target_task = pair["target"]
            pair_name = pair["name"]
            
            logger.info(f"Initializing dataloaders for {pair_name}")

            source_train = DomainTaskDataset(
                task_name=source_task,
                tokenizer=self.tokenizer,
                data_dir=self.data_dir,
                split="train",
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
                n_obs=self.train_n_obs,
                seed=self.seed
            )
            
            source_val = DomainTaskDataset(
                task_name=source_task,
                tokenizer=self.tokenizer,
                data_dir=self.data_dir,
                split="validation",
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
                n_obs=self.val_n_obs,
                seed=self.seed
            )

            target_train = None
            if self.include_target_train:
                target_train = DomainTaskDataset(
                    task_name=target_task,
                    tokenizer=self.tokenizer,
                    data_dir=self.data_dir,
                    split="train",
                    max_source_length=self.max_source_length,
                    max_target_length=self.max_target_length,
                    n_obs=self.train_n_obs,
                    seed=self.seed
                )
            
            target_val = DomainTaskDataset(
                task_name=target_task,
                tokenizer=self.tokenizer,
                data_dir=self.data_dir,
                split="validation",
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
                n_obs=self.val_n_obs,
                seed=self.seed
            )
            
            target_test = DomainTaskDataset(
                task_name=target_task,
                tokenizer=self.tokenizer,
                data_dir=self.data_dir,
                split="test",
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
                n_obs=self.test_n_obs,
                seed=self.seed
            )
            train_datasets = [source_train]
            if target_train is not None:
                train_datasets.append(target_train)
            
            if len(train_datasets) > 1:
                if self.source_domain_ratio != 0.5:
                    source_size = len(source_train)
                    target_size = len(target_train)
                    total_size = source_size + target_size
                    
                    source_ratio = self.source_domain_ratio
                    target_ratio = 1 - source_ratio
                    
                    # Calculate desired sizes
                    desired_source_size = int(total_size * source_ratio)
                    desired_target_size = int(total_size * target_ratio)
                    
                    # Subsample if necessary
                    if desired_source_size < source_size:
                        source_train = self.subsample_dataset(source_train, desired_source_size)
                    
                    if desired_target_size < target_size:
                        target_train = self.subsample_dataset(target_train, desired_target_size)
                    
                    train_datasets = [source_train, target_train]
            
            combined_train = ConcatDataset(train_datasets)
            train_loader = DataLoader(
                combined_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            source_val_loader = DataLoader(
                source_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            target_val_loader = DataLoader(
                target_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            target_test_loader = DataLoader(
                target_test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            self.dataloaders[pair_name] = {
                "train": train_loader,
                "source_val": source_val_loader,
                "target_val": target_val_loader,
                "target_test": target_test_loader,
                "source_task": source_task,
                "target_task": target_task
            }
    
    def subsample_dataset(self, dataset, size):
        """Subsample a dataset to the given size"""
        indices = torch.randperm(len(dataset))[:size]
        return torch.utils.data.Subset(dataset, indices)
    
    def get_dataloader(self, pair_name, split):
        """Get dataloader for a specific pair and split"""
        if pair_name not in self.dataloaders:
            raise ValueError(f"Pair {pair_name} not found. Available pairs: {list(self.dataloaders.keys())}")
        
        if split not in self.dataloaders[pair_name]:
            raise ValueError(f"Split {split} not found. Available splits: {list(self.dataloaders[pair_name].keys())}")
        
        return self.dataloaders[pair_name][split]
    
    def get_all_dataloaders(self):
        """Get all dataloaders"""
        return self.dataloaders