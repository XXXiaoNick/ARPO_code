import os
from typing import Dict, List, Tuple, Optional, Union
import json

MODEL_CONFIGS = {
    "t5-small": {
        "prefix_length": 100,
        "prefix_dim": 512,
        "di_ratio": 0.6
    },
    "t5-base": {
        "prefix_length": 100,
        "prefix_dim": 768,
        "di_ratio": 0.6
    },
    "t5-large": {
        "prefix_length": 100,
        "prefix_dim": 1024,
        "di_ratio": 0.6
    }
}

TRAINING_CONFIGS = {
    "default": {
        "learning_rate": 3e-5,
        "batch_size": 16,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "num_epochs": 5,
        "warmup_steps": 500,
        "lr_decay_steps": 1000,
        "lr_decay_rate": 0.9,
        "seed": 42
    },
    "high_lr": {
        "learning_rate": 5e-5,
        "batch_size": 16,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "num_epochs": 5,
        "warmup_steps": 500,
        "lr_decay_steps": 1000,
        "lr_decay_rate": 0.9,
        "seed": 42
    },
    "low_lr": {
        "learning_rate": 1e-5,
        "batch_size": 16,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "num_epochs": 5,
        "warmup_steps": 500,
        "lr_decay_steps": 1000,
        "lr_decay_rate": 0.9,
        "seed": 42
    },
    "large_batch": {
        "learning_rate": 3e-5,
        "batch_size": 32,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "num_epochs": 5,
        "warmup_steps": 500,
        "lr_decay_steps": 1000,
        "lr_decay_rate": 0.9,
        "seed": 42
    }
}

DISENTANGLEMENT_CONFIGS = {
    "default": {
        "beta1": 0.1,
        "beta2": 0.5,
        "lambda0": 1.0,
        "lambda1": 1.0,
        "lambda2": 1.0,
        "lambda3": 1.0,
        "lambda4": 1.0,
        "lambda5": 1.0,
        "temp": 0.07
    },
    "strong_di": {
        "beta1": 0.2,
        "beta2": 0.3,
        "lambda0": 2.0,
        "lambda1": 1.5,
        "lambda2": 0.8,
        "lambda3": 1.5,
        "lambda4": 1.5,
        "lambda5": 1.5,
        "temp": 0.05
    },
    "weak_di": {
        "beta1": 0.05,
        "beta2": 0.8,
        "lambda0": 0.5,
        "lambda1": 0.5,
        "lambda2": 1.5,
        "lambda3": 0.5,
        "lambda4": 0.5,
        "lambda5": 0.5,
        "temp": 0.1
    }
}

ADVERSARIAL_CONFIGS = {
    "none": {
        "alpha": 0.0,
        "theta0": 0.5,
        "gamma": 0.1,
        "window_size": 10,
        "use_fgsm": False,
        "fgsm_strength": 0.0,
        "use_pgd": False,
        "pgd_strength": 0.0,
        "pgd_steps": 0,
        "use_phrase_swap": False,
        "phrase_swap_prob": 0.0,
        "phrase_swap_strength": 0.0,
        "use_task_cross_domain": False,
        "task_cross_domain_prob": 0.0,
        "task_cross_domain_strength": 0.0
    },
    "light": {
        "alpha": 0.3,
        "theta0": 0.4,
        "gamma": 0.1,
        "window_size": 10,
        "use_fgsm": True,
        "fgsm_strength": 0.005,
        "use_pgd": False,
        "pgd_strength": 0.0,
        "pgd_steps": 0,
        "use_phrase_swap": True,
        "phrase_swap_prob": 0.1,
        "phrase_swap_strength": 0.05,
        "use_task_cross_domain": False,
        "task_cross_domain_prob": 0.0,
        "task_cross_domain_strength": 0.0
    },
    "medium": {
        "alpha": 0.5,
        "theta0": 0.5,
        "gamma": 0.1,
        "window_size": 10,
        "use_fgsm": True,
        "fgsm_strength": 0.01,
        "use_pgd": False,
        "pgd_strength": 0.0,
        "pgd_steps": 0,
        "use_phrase_swap": True,
        "phrase_swap_prob": 0.2,
        "phrase_swap_strength": 0.1,
        "use_task_cross_domain": True,
        "task_cross_domain_prob": 0.1,
        "task_cross_domain_strength": 0.2
    },
    "strong": {
        "alpha": 0.7,
        "theta0": 0.3,
        "gamma": 0.1,
        "window_size": 10,
        "use_fgsm": True,
        "fgsm_strength": 0.02,
        "use_pgd": True,
        "pgd_strength": 0.01,
        "pgd_steps": 3,
        "use_phrase_swap": True,
        "phrase_swap_prob": 0.3,
        "phrase_swap_strength": 0.2,
        "use_task_cross_domain": True,
        "task_cross_domain_prob": 0.2,
        "task_cross_domain_strength": 0.3
    },
    "custom": {
        "alpha": 0.5,
        "theta0": 0.5,
        "gamma": 0.1,
        "window_size": 10,
        "use_fgsm": True,
        "fgsm_strength": 0.01,
        "use_pgd": True,
        "pgd_strength": 0.01,
        "pgd_steps": 3,
        "use_phrase_swap": True,
        "phrase_swap_prob": 0.2,
        "phrase_swap_strength": 0.1,
        "use_task_cross_domain": True,
        "task_cross_domain_prob": 0.1,
        "task_cross_domain_strength": 0.3
    }
}

BO_CONFIGS = {
    "disabled": {
        "use_bayesian_opt": False,
        "bo_update_steps": 0
    },
    "light": {
        "use_bayesian_opt": True,
        "bo_update_steps": 500
    },
    "medium": {
        "use_bayesian_opt": True,
        "bo_update_steps": 300
    },
    "aggressive": {
        "use_bayesian_opt": True,
        "bo_update_steps": 100
    }
}

DATA_CONFIGS = {
    "default": {
        "max_seq_length": 512,
        "max_target_length": 128,
        "num_workers": 4,
        "train_n_obs": None,
        "val_n_obs": None,
        "test_n_obs": None,
        "source_domain_ratio": 0.7,
        "include_target_train": True
    },
    "small_data": {
        "max_seq_length": 512,
        "max_target_length": 128,
        "num_workers": 4,
        "train_n_obs": 1000,
        "val_n_obs": 200,
        "test_n_obs": 200,
        "source_domain_ratio": 0.7,
        "include_target_train": True
    },
    "no_target_train": {
        "max_seq_length": 512,
        "max_target_length": 128,
        "num_workers": 4,
        "train_n_obs": None,
        "val_n_obs": None,
        "test_n_obs": None,
        "source_domain_ratio": 1.0,
        "include_target_train": False
    },
    "balanced": {
        "max_seq_length": 512,
        "max_target_length": 128,
        "num_workers": 4,
        "train_n_obs": None,
        "val_n_obs": None,
        "test_n_obs": None,
        "source_domain_ratio": 0.5,
        "include_target_train": True
    }
}

EXPERIMENT_SETTINGS = {
    "setting1": {
        "description": "Primary cross-domain transfer experiments",
        "pairs": [
            {"source": "nq", "target": "searchqa", "name": "NQ→SQA"},
            {"source": "yelp_polarity", "target": "scitail", "name": "Yelp→SciTail"},
            {"source": "newsqa", "target": "hotpotqa", "name": "News→HP"},
            {"source": "mnli", "target": "qqp", "name": "MNLI→QQP"},
            {"source": "cola", "target": "paws", "name": "CoLA→PAWS"}
        ]
    },
    "setting2": {
        "description": "Secondary cross-domain transfer experiments",
        "pairs": [
            {"source": "cola", "target": "qqp", "name": "CoLA→QQP"},
            {"source": "rte", "target": "sst2", "name": "RTE→SST-2"},
            {"source": "superglue-boolq", "target": "nq", "name": "BoolQ→NQ"},
            {"source": "hotpotqa", "target": "scitail", "name": "HP→SciTail"},
            {"source": "mnli", "target": "searchqa", "name": "MNLI→SQA"}
        ]
    }
}

EXPERIMENT_PROFILES = {
    "baseline": {
        "model_config": "t5-base",
        "training_config": "default",
        "disentanglement_config": "default",
        "adversarial_config": "none",
        "bo_config": "disabled",
        "data_config": "default"
    },
    "light_arpo": {
        "model_config": "t5-base",
        "training_config": "default",
        "disentanglement_config": "default",
        "adversarial_config": "light",
        "bo_config": "disabled",
        "data_config": "default"
    },
    "full_arpo": {
        "model_config": "t5-base",
        "training_config": "default",
        "disentanglement_config": "default",
        "adversarial_config": "medium",
        "bo_config": "medium",
        "data_config": "default"
    },
    "strong_arpo": {
        "model_config": "t5-base",
        "training_config": "default",
        "disentanglement_config": "strong_di",
        "adversarial_config": "strong",
        "bo_config": "aggressive",
        "data_config": "default"
    },
    "small_data": {
        "model_config": "t5-base",
        "training_config": "default",
        "disentanglement_config": "default",
        "adversarial_config": "medium",
        "bo_config": "medium",
        "data_config": "small_data"
    },
    "large_model": {
        "model_config": "t5-large",
        "training_config": "low_lr",
        "disentanglement_config": "default",
        "adversarial_config": "medium",
        "bo_config": "medium",
        "data_config": "default"
    }
}


class ARPOConfig:
    def __init__(
        self,
        profile: str = "baseline",
        setting: str = "setting1",
        data_dir: str = None,
        output_dir: str = None,
        custom_configs: Dict = None
    ):
        self.profile = profile
        self.setting = setting
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        if profile not in EXPERIMENT_PROFILES:
            raise ValueError(f"Unknown profile: {profile}. Available profiles: {list(EXPERIMENT_PROFILES.keys())}")
        
        self.profile_config = EXPERIMENT_PROFILES[profile]

        model_config_name = self.profile_config["model_config"]
        self.model_config = MODEL_CONFIGS[model_config_name].copy()

        training_config_name = self.profile_config["training_config"]
        self.training_config = TRAINING_CONFIGS[training_config_name].copy()
        
        disent_config_name = self.profile_config["disentanglement_config"]
        self.disentanglement_config = DISENTANGLEMENT_CONFIGS[disent_config_name].copy()
        
        adv_config_name = self.profile_config["adversarial_config"]
        self.adversarial_config = ADVERSARIAL_CONFIGS[adv_config_name].copy()
        
        bo_config_name = self.profile_config["bo_config"]
        self.bo_config = BO_CONFIGS[bo_config_name].copy()
        
        data_config_name = self.profile_config["data_config"]
        self.data_config = DATA_CONFIGS[data_config_name].copy()
        
        if setting not in EXPERIMENT_SETTINGS:
            raise ValueError(f"Unknown setting: {setting}. Available settings: {list(EXPERIMENT_SETTINGS.keys())}")
        
        self.experiment_setting = EXPERIMENT_SETTINGS[setting].copy()

        if custom_configs:
            self._apply_custom_configs(custom_configs)
    
    def _apply_custom_configs(self, custom_configs):
        for config_type, config_dict in custom_configs.items():
            if config_type == "model":
                self.model_config.update(config_dict)
            elif config_type == "training":
                self.training_config.update(config_dict)
            elif config_type == "disentanglement":
                self.disentanglement_config.update(config_dict)
            elif config_type == "adversarial":
                self.adversarial_config.update(config_dict)
            elif config_type == "bo":
                self.bo_config.update(config_dict)
            elif config_type == "data":
                self.data_config.update(config_dict)
            elif config_type == "experiment":
                self.experiment_setting.update(config_dict)
    
    def get_args_dict(self):
        args_dict = {
            "profile": self.profile,
            "setting": self.setting,
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,

            "model_name": self.profile_config["model_config"],
            **self.model_config,

            **self.training_config,

            **self.disentanglement_config,

            **self.adversarial_config,

            **self.bo_config,
            
            **self.data_config
        }
        
        return args_dict
    
    def save_config(self, save_path=None):
        if save_path is None:
            if self.output_dir is None:
                raise ValueError("Either save_path or output_dir must be provided")
            
            os.makedirs(self.output_dir, exist_ok=True)
            save_path = os.path.join(self.output_dir, f"config_{self.profile}_{self.setting}.json")
        
        config_data = {
            "profile": self.profile,
            "setting": self.setting,
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "disentanglement_config": self.disentanglement_config,
            "adversarial_config": self.adversarial_config,
            "bo_config": self.bo_config,
            "data_config": self.data_config,
            "experiment_setting": self.experiment_setting
        }
        
        with open(save_path, "w") as f:
            json.dump(config_data, f, indent=2)
        
        return save_path
    
    @classmethod
    def from_json(cls, json_path):
        """Load configuration from JSON file"""
        with open(json_path, "r") as f:
            config_data = json.load(f)
        
        # Extract basic info
        profile = config_data.get("profile", "baseline")
        setting = config_data.get("setting", "setting1")
        data_dir = config_data.get("data_dir")
        output_dir = config_data.get("output_dir")
        
        # Create custom configs from loaded data
        custom_configs = {
            "model": config_data.get("model_config", {}),
            "training": config_data.get("training_config", {}),
            "disentanglement": config_data.get("disentanglement_config", {}),
            "adversarial": config_data.get("adversarial_config", {}),
            "bo": config_data.get("bo_config", {}),
            "data": config_data.get("data_config", {}),
            "experiment": config_data.get("experiment_setting", {})
        }
        
        # Create config object
        config = cls(
            profile=profile,
            setting=setting,
            data_dir=data_dir,
            output_dir=output_dir,
            custom_configs=custom_configs
        )
        
        return config