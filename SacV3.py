import copy
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.utils import is_freq, AttrDict, soft_update_param
from utils.base import Base, Module, MainModule, Task


class SAC(MainModule):
    @Base.save_input(exclude=["qs", "policy"])
    def __init__(self, gamma:float, 
                 qs:List[Module], q_train_freq:int, q_lr:float, target_q_update_ratio:float, target_q_update_freq:int,
                 policy:Module, pi_train_freq:int, pi_lr:float, 
                 dynamic_alpha:bool, init_alpha:float=1.0, target_entropy:Optional[float]=None):
        super().__init__()
        # --- Q ---
        self.q_train_freq = q_train_freq
        self.q_lr = q_lr
        self.target_q_update_ratio = target_q_update_ratio
        self.target_q_update_freq = target_q_update_freq
        self.qs = nn.ModuleList(qs)
        self.target_qs = copy.deepcopy(self.qs).eval().requires_grad_(False)
        self.optimizer_q = optim.Adam(self.qs.parameters(), lr=self.q_lr)
        
        # --- Pi ---
        self.pi_train_freq = pi_train_freq
        self.pi_lr = pi_lr
        self.Pi = policy
        self.optimizer_pi = optim.Adam(self.Pi.parameters(), lr=self.pi_lr)
        
        # --- alpha(temperature) ---
        self.dynamic_alpha = dynamic_alpha
        if self.dynamic_alpha:
            self.target_entropy = target_entropy
            self.log_alpha = nn.Parameter(torch.tensor(init_alpha).float().log())
            self.alpha = self.log_alpha.detach().exp().item()
            self.optimizer_alpha = optim.Adam([self.log_alpha], lr=self.q_lr)
        else:
            self.alpha = init_alpha
        
        # --- etc. ---
        self.gamma = gamma
        self._step = 0
            
    def loss_critic(self, sample:List[torch.Tensor]) -> dict:
        state, action, reward, next_state, done = sample
        # --- estimate ---
        # Q(s_t, a_t)
        estimate_qs = [
            q(state, action) for q in self.qs # (d_batch, 1)
        ]
        
        # --- target ---
        with torch.no_grad():
            # a'_t+1~Pi(- | s'), log( Pi(a'_t+1 | s_t+1) )
            next_pi_action, next_log_pi = self.Pi.sample(next_state, with_log_prob=True) # (d_batch, d_action), (d_batch, 1)
            # Q'(s_t+1, a'_t+1)
            next_target_qs = [
                target_q(next_state, next_pi_action) for target_q in self.target_qs # (d_batch, 1)
            ]
            # min Q' = min{Q1'(s_t+1, a'_t+1), Q2'(s_t+1, a'_t+1), ...}
            next_target_min_q = torch.cat(next_target_qs, dim=-1).min(dim=-1, keepdim=True).values # (d_batch, q_n) -> (d_batch, 1)
            # V'(s_t+1) = min Q' - alpha * log( Pi(a'_t+1 | s_t+1) )
            next_target_v = next_target_min_q - self.alpha * next_log_pi # (d_batch, 1)
            # Q'(s_t, a_t) = r + gamma * V'(s_t+1)
            # target_q = reward + (1 - done) * self.gamma * next_target_v # (d_batch, 1)
            target_q = reward + done.logical_not() * self.gamma * next_target_v # (d_batch, 1)
            
        # --- MSE loss ---
        # MSE{ Q(s_t, a_t), Q'(s_t, a_t) }
        loss_qs = [
            F.mse_loss(estimate_q, target_q) for estimate_q in estimate_qs # scalar
        ]
        loss_q = sum(loss_qs) # scalar
        
        return AttrDict(
            loss_q=loss_q,
            **{f'loss_q{i+1}': loss_qs[i].detach().cpu().item() for i in range(len(loss_qs))},
            **{f'q{i+1}': estimate_qs[i].detach().mean().cpu().item() for i in range(len(estimate_qs))},
        )
        
    def loss_actor(self, sample:List[torch.Tensor]) -> AttrDict:
        state, action, reward, next_state, done = sample
        
        # a'_t ~ Pi(- | s), log( Pi(a'_t | s_t) )
        pi_action, log_pi = self.Pi.sample(state, with_log_prob=True) # (d_batch, d_action), (d_batch, 1)
        # q(s_t, a'_t)
        with Base.freeze_modes(*self.qs):
            estimate_qs = [
                q(state, pi_action) for q in self.qs # (d_batch, 1)
            ]
        # min q = min{ Q1(s_t, a'_t), Q2(s_t, a'_t), ... }
        estimate_min_q = torch.cat(estimate_qs, dim=-1).min(dim=-1, keepdim=True).values # (d_batch, q_n) -> (d_batch, 1)
        # (alpha * log( Pi(a'_t | s_t) )) - min q(s_t, a'_t)
        loss_pi = (self.alpha * log_pi) - estimate_min_q # (d_batch, 1)
        loss_pi = loss_pi.mean() # scalar
        
        return AttrDict(
            loss_pi=loss_pi,
            entropy_pi=-log_pi.detach().mean().cpu().item(),
        )
        
    def loss_alpha(self, sample:List[torch.Tensor]) -> AttrDict:
        state, action, reward, next_state, done = sample
        
        # log( Pi(a'_t | s_t) )
        with torch.no_grad():
            _pi_action, log_pi = self.Pi.sample(state, with_log_prob=True) # (d_batch, 1)
        loss_alpha = -self.log_alpha.exp() * (log_pi + self.target_entropy) # (d_batch, 1)
        loss_alpha = loss_alpha.mean()
        
        return AttrDict(loss_alpha=loss_alpha)
    
    def update_target(self) -> None:
        if is_freq(self._step, self.target_q_update_freq):
            soft_update_param(self.target_qs, self.qs, tau=self.target_q_update_ratio)
    
    def update(self, sample:List[torch.Tensor]) -> AttrDict:
        self._step += 1
        self.update_target()
        result = AttrDict()
        
        # --- Q ---
        if is_freq(self._step, self.q_train_freq):
            for _ in range(self.q_train_freq):
                result_critic = self.loss_critic(sample)
                self.optimizer_q.zero_grad()
                result_critic.loss_q.backward()
                self.optimizer_q.step()
                
            # for log
            result_critic.loss_q = result_critic.loss_q.detach().cpu().item()
            result.update(result_critic)
            
        if is_freq(self._step, self.pi_train_freq):
            # --- Pi ---
            for _ in range(self.pi_train_freq):
                result_actor = self.loss_actor(sample)
                self.optimizer_pi.zero_grad()
                result_actor.loss_pi.backward()
                self.optimizer_pi.step()
            
                if self.dynamic_alpha:
                    # --- alpha ---
                    result_alpha = self.loss_alpha(sample)
                    self.optimizer_alpha.zero_grad()
                    result_alpha.loss_alpha.backward()
                    self.optimizer_alpha.step()
                    self.alpha = self.log_alpha.detach().exp().item()
            
            # for logging
            result_actor.loss_pi = result_actor.loss_pi.detach().cpu().item()
            result.update(result_actor)
            if self.dynamic_alpha:
                result_alpha.loss_alpha = result_alpha.loss_alpha.detach().cpu().item()
                result.update(result_alpha)
        
        return result


def test(task:Task, policy:Module):
    with task.test_mode(), policy.inference_mode():
        task.reset()
        while True:
            action = policy(task.state, deterministic=True).cpu().numpy() # (d_action)
            task.step(action)
            if task.is_terminal():
                break
        score = task.score # NOTE: must run inside a "with" statement
    return score
