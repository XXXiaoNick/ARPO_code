import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import T5Model, T5Config
from typing import Dict, List, Tuple, Optional, Union
import math
from botorch.models import SingleTaskGP
from botorch.acquisition.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.transforms import normalize, unnormalize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning


class MINENetwork(nn.Module):
    
    def __init__(self, prefix_dim, domain_dim, hidden_dim=256):
        super(MINENetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(prefix_dim + domain_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, prefix, domain):
        if domain.dim() == 1:
            domain = F.one_hot(domain, num_classes=domain.max()+1).float()
            
        joint_input = torch.cat([prefix, domain], dim=1)
        return self.net(joint_input)


class HSIC(nn.Module):
    
    def __init__(self, sigma=1.0):
        super(HSIC, self).__init__()
        self.sigma = sigma
        
    def compute_kernel(self, x, y=None):
        """Compute RBF kernel"""
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is None:
            y = x
            y_norm = x_norm
        else:
            y_norm = (y ** 2).sum(1).view(1, -1)
        
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.exp(-dist / (2 * self.sigma ** 2))
    
    def forward(self, x, y):
        batch_size = x.size(0)
        
        # Center the kernel matrices
        K_x = self.compute_kernel(x)
        K_y = self.compute_kernel(y)
        
        # Center K_x and K_y
        H = torch.eye(batch_size, device=x.device) - 1.0/batch_size * torch.ones((batch_size, batch_size), device=x.device)
        K_x_centered = torch.matmul(torch.matmul(H, K_x), H)
        K_y_centered = torch.matmul(torch.matmul(H, K_y), H)
        
        # Compute HSIC
        hsic_value = torch.trace(torch.matmul(K_x_centered, K_y_centered)) / (batch_size - 1) ** 2
        
        return hsic_value


class ARPOModel(nn.Module):
    def __init__(
        self,
        base_model_name: str = "t5-base",
        prefix_length: int = 100,
        prefix_dim: int = 768,
        di_ratio: float = 0.6,
        beta1: float = 0.1,
        beta2: float = 0.5,
        lambda0: float = 1.0,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 1.0,
        lambda4: float = 1.0,
        lambda5: float = 1.0,
        temp: float = 0.07,
        alpha: float = 0.5,
        theta0: float = 0.5,
        gamma: float = 0.1,
        window_size: int = 10,
        epsilon: float = 1e-8,
        device = None
    ):
        super(ARPOModel, self).__init__()
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.t5_model = T5Model.from_pretrained(base_model_name)
        self.t5_config = self.t5_model.config
        
        self.prefix_length = prefix_length
        self.prefix_dim = prefix_dim
        
        self.di_length = int(prefix_length * di_ratio)
        self.ds_length = prefix_length - self.di_length
        
        self.prefix_DI = nn.Parameter(torch.randn(self.di_length, prefix_dim))
        self.prefix_DS = nn.Parameter(torch.randn(self.ds_length, prefix_dim))
        
        self.mine_di_domain = MINENetwork(prefix_dim, 1)
        self.mine_di_task = MINENetwork(prefix_dim, 1)
        self.mine_ds_domain = MINENetwork(prefix_dim, 1)
        self.mine_ds_task = MINENetwork(prefix_dim, 1)
        
        self.hsic = HSIC()
        
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.lambda0 = lambda0 
        self.lambda1 = lambda1 
        self.lambda2 = lambda2 
        self.lambda3 = lambda3 
        self.lambda4 = lambda4 
        self.lambda5 = lambda5 
        self.temp = temp  
        self.alpha = alpha  
        self.theta0 = theta0 
        self.gamma = gamma 
        self.window_size = window_size  
        self.epsilon = epsilon  
      
        self.task_loss_history = []
        self.task_grad_history = []
        self.task_loss_initial = None
        
        self.strategies = []
        self.strategy_performances = []
        
        self.to(self.device)
        
    def get_prefix_embeddings(self, batch_size):
        """Get prefix embeddings expanded to batch size"""
        prefix_di = self.prefix_DI.unsqueeze(0).expand(batch_size, -1, -1)
        prefix_ds = self.prefix_DS.unsqueeze(0).expand(batch_size, -1, -1)
        
        return prefix_di, prefix_ds
    
    def estimate_mutual_information(self, prefix, domain, task, mine_net):
        batch_size = prefix.size(0)

        joint_output = mine_net(prefix, domain)

        perm_idx = torch.randperm(batch_size, device=self.device)
        shuffled_domain = domain[perm_idx]
        marginal_output = mine_net(prefix, shuffled_domain)

        mi_estimate = joint_output.mean() - torch.log(torch.exp(marginal_output).mean() + self.epsilon)
        
        return mi_estimate
    
    def compute_ib_loss(self, prefix_di_pooled, prefix_ds_pooled, domain_labels, task_labels):
        I_di_domain = self.estimate_mutual_information(prefix_di_pooled, domain_labels, task_labels, self.mine_di_domain)
        I_di_task = self.estimate_mutual_information(prefix_di_pooled, task_labels, domain_labels, self.mine_di_task)
        I_ds_domain = self.estimate_mutual_information(prefix_ds_pooled, domain_labels, task_labels, self.mine_ds_domain)
        I_ds_task = self.estimate_mutual_information(prefix_ds_pooled, task_labels, domain_labels, self.mine_ds_task)

        ib_loss_di = I_di_domain - self.beta1 * I_di_task
        
        ib_loss_ds = -I_ds_domain + self.beta2 * I_ds_task
        
        return ib_loss_di, ib_loss_ds
    
    def compute_orthogonality_loss(self, prefix_di_pooled, prefix_ds_pooled):
        hsic_value = self.hsic(prefix_di_pooled, prefix_ds_pooled)

        orth_loss = self.lambda0 * (1 - hsic_value).pow(-1)
        
        return orth_loss
    
    def compute_contrastive_loss(self, prefix_di_pooled, prefix_ds_pooled, task_labels):
        batch_size = prefix_di_pooled.size(0)

        norm_di = F.normalize(prefix_di_pooled, p=2, dim=1)
        norm_ds = F.normalize(prefix_ds_pooled, p=2, dim=1)
        
        sim_matrix_di = torch.matmul(norm_di, norm_di.t()) / self.temp
        sim_matrix_ds = torch.matmul(norm_ds, norm_ds.t()) / self.temp

        task_eq = task_labels.unsqueeze(1) == task_labels.unsqueeze(0)
        mask_pos = task_eq & (~torch.eye(batch_size, device=self.device).bool())

        loss_di = 0
        if mask_pos.sum() > 0:
            pos_sim_di = sim_matrix_di[mask_pos]

            exp_sim_di = torch.exp(sim_matrix_di)
            exp_sim_di = exp_sim_di - torch.diag(torch.diag(exp_sim_di))
            denom_di = exp_sim_di.sum(dim=1)

            pos_exp_di = torch.exp(pos_sim_di)
            loss_di = -torch.log(pos_exp_di / denom_di.repeat_interleave(mask_pos.sum(dim=1)[mask_pos.sum(dim=1) > 0])).mean()

        loss_ds = 0
        if mask_pos.sum() > 0:
            neg_sim_ds = -sim_matrix_ds[mask_pos]
            
            exp_sim_ds = torch.exp(-sim_matrix_ds)
            exp_sim_ds = exp_sim_ds - torch.diag(torch.diag(exp_sim_ds))
            denom_ds = exp_sim_ds.sum(dim=1)

            neg_exp_ds = torch.exp(neg_sim_ds)
            loss_ds = -torch.log(neg_exp_ds / denom_ds.repeat_interleave(mask_pos.sum(dim=1)[mask_pos.sum(dim=1) > 0])).mean()

        cont_loss = loss_di + loss_ds
        
        return cont_loss
    
    def compute_conditional_independence_loss(self, prefix_di_pooled, prefix_ds_pooled, task_labels):
        unique_tasks = torch.unique(task_labels)
        cond_loss = 0
        
        for task in unique_tasks:
            task_mask = (task_labels == task)
            di_task = prefix_di_pooled[task_mask]
            ds_task = prefix_ds_pooled[task_mask]
            
            if di_task.size(0) <= 1:
                continue
            hsic_task = self.hsic(di_task, ds_task)
            cond_loss += hsic_task
            
        # Average over all tasks
        if len(unique_tasks) > 0:
            cond_loss /= len(unique_tasks)
            
        return cond_loss
    
    def compute_disentanglement_loss(self, prefix_di_pooled, prefix_ds_pooled, domain_labels, task_labels):
        ib_loss_di, ib_loss_ds = self.compute_ib_loss(
            prefix_di_pooled, prefix_ds_pooled, domain_labels, task_labels
        )
        orth_loss = self.compute_orthogonality_loss(prefix_di_pooled, prefix_ds_pooled)
        cont_loss = self.compute_contrastive_loss(prefix_di_pooled, prefix_ds_pooled, task_labels)
        cond_loss = self.compute_conditional_independence_loss(prefix_di_pooled, prefix_ds_pooled, task_labels)
        
        disent_loss = (
            self.lambda1 * ib_loss_di +
            self.lambda2 * ib_loss_ds +
            self.lambda3 * orth_loss +
            self.lambda4 * cont_loss +
            self.lambda5 * cond_loss
        )
        
        return disent_loss, {
            'ib_di': ib_loss_di.item(),
            'ib_ds': ib_loss_ds.item(),
            'orth': orth_loss.item(),
            'cont': cont_loss.item(),
            'cond': cond_loss.item()
        }
    
    def compute_dynamic_threshold(self, task_loss, task_grads=None):
        self.task_loss_history.append(task_loss.item())
        if len(self.task_loss_history) > self.window_size:
            self.task_loss_history.pop(0)
            
        if self.task_loss_initial is None:
            self.task_loss_initial = task_loss.item()
            
        loss_improvement = 1 - task_loss.item() / self.task_loss_initial

        if task_grads is not None:
            flat_grad = torch.cat([g.flatten() for g in task_grads if g is not None])
            self.task_grad_history.append(flat_grad)
            if len(self.task_grad_history) > self.window_size:
                self.task_grad_history.pop(0)
                
            if len(self.task_grad_history) > 1:
                grad_tensor = torch.stack(self.task_grad_history)
                grad_var = torch.var(grad_tensor, dim=0).mean()
                grad_mean = torch.abs(grad_tensor).mean()
                
                ratio = (grad_var / (grad_mean + self.epsilon)).item()
                theta = self.theta0 * (1 - math.exp(-self.gamma * ratio))
            else:
                theta = self.theta0
        else:
            theta = self.theta0

        if len(self.task_loss_history) >= 2:
            loss_std = torch.tensor(self.task_loss_history).std().item()
            loss_diffs = [self.task_loss_history[i] - self.task_loss_history[i+1] 
                          for i in range(len(self.task_loss_history)-1)]
            loss_diff_mean = abs(sum(loss_diffs) / len(loss_diffs) + self.epsilon)
            
            difficulty = loss_std / loss_diff_mean
            
            theta_task = theta * (1 + 0.5 * difficulty) 
        else:
            theta_task = theta
            
        return theta_task, loss_improvement
    
    def generate_pos_neg_pairs(self, h_ds, domain_labels, threshold=0.5):
        batch_size = h_ds.size(0)
        
        distances = torch.cdist(h_ds, h_ds, p=2)
        
        pos_pairs = []
        neg_pairs = []

        for i in range(batch_size):
            for j in range(i+1, batch_size):
                same_domain = domain_labels[i] == domain_labels[j]

                dist = distances[i, j].item()
                
                if same_domain and dist < threshold:
                    pos_pairs.append((i, j))
                elif not same_domain or dist >= threshold:
                    neg_pairs.append((i, j))
        
        return pos_pairs, neg_pairs
    
    def compute_adversarial_loss(self, h_ds, domain_labels, threshold=0.5):
        pos_pairs, neg_pairs = self.generate_pos_neg_pairs(h_ds, domain_labels, threshold)
        
        if len(pos_pairs) == 0 or len(neg_pairs) == 0:
            return torch.tensor(0.0, device=self.device)

        pos_loss = sum(
            torch.norm(h_ds[i] - h_ds[j], p=2) 
            for i, j in pos_pairs
        )
        
        neg_loss = sum(
            torch.norm(h_ds[i] - h_ds[j], p=2)
            for i, j in neg_pairs
        )
        
        if len(pos_pairs) > 0:
            pos_loss = pos_loss / len(pos_pairs)
        if len(neg_pairs) > 0:
            neg_loss = neg_loss / len(neg_pairs)

        adv_loss = pos_loss - neg_loss
        
        return adv_loss
    
    def apply_adversarial_strategies(self, inputs, strategy, epsilon=0.01):
        perturbed_inputs = inputs.clone()
        if strategy.get('token_fgsm', False):
            inputs.requires_grad = True
            outputs = self.t5_model.encoder(
                inputs_embeds=inputs
            )
            pseudo_loss = outputs.last_hidden_state.mean()
            pseudo_loss.backward()
            if inputs.grad is not None:
                perturbation = epsilon * inputs.grad.sign()
                perturbed_inputs = perturbed_inputs + strategy.get('token_fgsm_strength', 1.0) * perturbation
                
            inputs.requires_grad = False
            
        if strategy.get('token_pgd', False):
            inputs.requires_grad = True
            original_inputs = inputs.clone()
            perturbed_inputs = inputs.clone()
            
            steps = strategy.get('token_pgd_steps', 3)
            step_size = epsilon / steps
            
            for _ in range(steps):
                perturbed_inputs.requires_grad = True
                outputs = self.t5_model.encoder(
                    inputs_embeds=perturbed_inputs
                )
                pseudo_loss = outputs.last_hidden_state.mean()
                pseudo_loss.backward()

                if perturbed_inputs.grad is not None:
                    perturbation = step_size * perturbed_inputs.grad.sign()
                    perturbed_inputs = perturbed_inputs + strategy.get('token_pgd_strength', 1.0) * perturbation
                    
                    delta = perturbed_inputs - original_inputs
                    norm = torch.norm(delta.view(delta.size(0), -1), dim=1).view(-1, 1, 1)
                    eps_mask = (norm > epsilon).float()
                    delta = delta * (1 - eps_mask) + delta / norm * epsilon * eps_mask
                    perturbed_inputs = original_inputs + delta
                
                perturbed_inputs = perturbed_inputs.detach()
                
            inputs.requires_grad = False
            
        if strategy.get('phrase_swap_a', False):
            if np.random.random() < strategy.get('phrase_swap_a_prob', 0.2):
                batch_size, seq_len, emb_dim = perturbed_inputs.shape
                
                num_phrases = max(1, int(strategy.get('phrase_swap_a_strength', 0.1) * seq_len))
                for b in range(batch_size):
                    for _ in range(num_phrases):
                        start_pos = np.random.randint(0, seq_len - 3)
                        phrase_len = np.random.randint(2, min(5, seq_len - start_pos))
                        
                        noise = torch.randn_like(perturbed_inputs[b, start_pos:start_pos+phrase_len]) * 0.01
                        perturbed_inputs[b, start_pos:start_pos+phrase_len] += noise
        
        if strategy.get('task_cross_domain', False):
            if np.random.random() < strategy.get('task_cross_domain_prob', 0.1):
                batch_size = perturbed_inputs.size(0)

                perm_idx = torch.randperm(batch_size, device=self.device)

                mix_ratio = strategy.get('task_cross_domain_strength', 0.3)
                perturbed_inputs = (1 - mix_ratio) * perturbed_inputs + mix_ratio * perturbed_inputs[perm_idx]
        
        return perturbed_inputs
    
    def update_bayesian_optimization(self, strategy, performance_metrics):
        self.strategies.append(strategy)
        self.strategy_performances.append(performance_metrics)

        if len(self.strategies) < 5:
            return None

        X = []
        for s in self.strategies:
            x_vec = [
                s.get('token_fgsm', False) * s.get('token_fgsm_strength', 0.0),
                s.get('token_pgd', False) * s.get('token_pgd_strength', 0.0) * s.get('token_pgd_steps', 0),
                s.get('phrase_swap_a', False) * s.get('phrase_swap_a_prob', 0.0) * s.get('phrase_swap_a_strength', 0.0),
                s.get('task_cross_domain', False) * s.get('task_cross_domain_prob', 0.0) * s.get('task_cross_domain_strength', 0.0),
                s.get('threshold', 0.5),
                s.get('alpha', 0.5)
            ]
            X.append(x_vec)
        
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        Y = []
        for perf in self.strategy_performances:
            y_vec = [
                perf.get('task_accuracy', 0.0),
                perf.get('adversarial_robustness', 0.0),
                -perf.get('computational_cost', 1.0)
            ]
            Y.append(y_vec)
            
        Y_tensor = torch.tensor(Y, dtype=torch.float32, device=self.device)
        
        X_norm = normalize(X_tensor)
        
        models = []
        for i in range(Y_tensor.shape[1]):
            gp = SingleTaskGP(X_norm, Y_tensor[:, i:i+1])
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            models.append((gp, mll))
        
        ref_point = Y_tensor.min(dim=0).values - 0.1

        partitioning = DominatedPartitioning(ref_point=ref_point, Y=Y_tensor)
        acq_func = qExpectedHypervolumeImprovement(
            models[0][0], 
            ref_point=ref_point,
            partitioning=partitioning
        )
        
        bounds = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.1], 
            [1.0, 1.0, 1.0, 1.0, 0.9, 0.9]  
        ], device=self.device)

        next_x_norm = torch.rand(1, 6, device=self.device) * (bounds[1] - bounds[0]) + bounds[0]
        next_x = unnormalize(next_x_norm, bounds)

        next_strategy = {
            'token_fgsm': next_x[0, 0].item() > 0.5,
            'token_fgsm_strength': max(0.1, next_x[0, 0].item()),
            'token_pgd': next_x[0, 1].item() > 0.5,
            'token_pgd_strength': max(0.1, next_x[0, 1].item()),
            'token_pgd_steps': max(1, int(next_x[0, 1].item() * 5)),
            'phrase_swap_a': next_x[0, 2].item() > 0.5,
            'phrase_swap_a_prob': max(0.1, next_x[0, 2].item()),
            'phrase_swap_a_strength': max(0.1, next_x[0, 2].item()),
            'task_cross_domain': next_x[0, 3].item() > 0.5,
            'task_cross_domain_prob': max(0.1, next_x[0, 3].item()),
            'task_cross_domain_strength': max(0.1, next_x[0, 3].item()),
            'threshold': next_x[0, 4].item(),
            'alpha': next_x[0, 5].item()
        }
        
        return next_strategy
    
    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        domain_labels=None,
        task_labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        strategy=None,
        training=True
    ):
        batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)
        
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_ids)

        prefix_di, prefix_ds = self.get_prefix_embeddings(batch_size)
        
        combined_embeds = inputs_embeds
        
        if strategy is not None and training:
            combined_embeds = self.apply_adversarial_strategies(combined_embeds, strategy)

        encoder_outputs = self.t5_model.encoder(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=combined_embeds,
            return_dict=True
        )
        
        hidden_states = encoder_outputs.last_hidden_state
        
        h_ds = hidden_states[:, :self.ds_length]

        prefix_di_pooled = self.prefix_DI.mean(dim=0).unsqueeze(0).repeat(batch_size, 1)
        prefix_ds_pooled = self.prefix_DS.mean(dim=0).unsqueeze(0).repeat(batch_size, 1)
        
        decoder_outputs = self.t5_model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            return_dict=True
        )
        
        lm_logits = self.t5_model.lm_head(decoder_outputs.last_hidden_state)

        task_loss = None
        if labels is not None:
            task_loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100 
            )
        

        disent_loss = None
        disent_components = None
        if training and domain_labels is not None and task_labels is not None:
            disent_loss, disent_components = self.compute_disentanglement_loss(
                prefix_di_pooled, prefix_ds_pooled, domain_labels, task_labels
            )
        
        adv_loss = None
        theta = None
        loss_improvement = None
        
        if training and task_loss is not None:
            theta, loss_improvement = self.compute_dynamic_threshold(task_loss)

            if domain_labels is not None and loss_improvement >= theta:
                adv_loss = self.compute_adversarial_loss(h_ds, domain_labels)
        
        total_loss = None
        if task_loss is not None:
            total_loss = task_loss
            
            if disent_loss is not None:
                total_loss = total_loss + disent_loss
            
            if adv_loss is not None and loss_improvement >= theta:
                total_loss = total_loss + self.alpha * adv_loss
        
        output_dict = {
            'lm_logits': lm_logits,
            'encoder_hidden_states': hidden_states,
            'decoder_hidden_states': decoder_outputs.last_hidden_state,
            'task_loss': task_loss,
            'disent_loss': disent_loss,
            'disent_components': disent_components,
            'adv_loss': adv_loss,
            'total_loss': total_loss,
            'dynamic_threshold': theta,
            'loss_improvement': loss_improvement
        }
        
        return output_dict