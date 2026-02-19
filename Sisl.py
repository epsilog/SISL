import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from typing import Optional, List, Tuple, Union, Callable

from utils.utils import is_freq, is_set, AttrDict, soft_update_param, to_onehot, numpy_split, \
    RunningMeanStd, TimeChecker
from utils.base import Base, Module, MainModule, Environment
from utils.distribution import Normal, kl_divergence
from utils.net import Stochastic, MLP, NormalMLP, TanhNormalMLP
from utils.buffer import Batch, Episode, TransitionBuffer, EpisodeBuffer, \
    get_multiple_sample_per_buffer, get_multiple_sample, FlatEpisodeBuffer, FlatMaxTransitionEpisodeBuffer
from meta_skill_utils import SetTransformer, GPUWorker, HierarchicalTimeLimitCollector, ConcurrentCollector
from SacV3 import SAC
from SpirlCL import SpirlCL, SkillEncoder




def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1)

from utils.buffer import uniform_sampling, prioritized_sampling, get_multiple_sample_by_index

def multiple_sample_multiple_priority_index(
    buffers:List[Union[TransitionBuffer, EpisodeBuffer]], n_sample:int, 
    task_index:int,
    sample_length:int=1, method:str="uniform", mode_episode:bool=False, **kwargs # e.g. temperature
):
    assert method in ["uniform", "prioritized"], f'unknown sample method {method}'
    n_buffer = len(buffers)
    is_episode_buffer = [isinstance(buf, EpisodeBuffer) for buf in buffers]
    if all(is_episode_buffer):
        type_buffer = "episode_buffer"
    elif not any(is_episode_buffer):
        type_buffer = "transition_buffer"
    else: 
        type_buffer = "mix_buffer"

    if mode_episode:
        assert type_buffer == "episode_buffer"
        assert sample_length == 1, "not supported sample_length option"
        n_episodes = np.array([buf.n_episode for buf in buffers])
        if method == "uniform":
            # NOTE: uniform sampling
            buffer_indices, episode_indices = uniform_sampling(
                n_sample=n_sample, array_lengths=n_episodes, sample_length=1
            )
            key = buffer_indices.argsort()
            buffer_indices = buffer_indices[key]
            episode_indices = episode_indices[key]

        elif method == "prioritized":
            # NOTE: prioritized sampling
            priorities = np.concatenate([buf.get_output_episode_priority()[:, task_index] for buf in buffers if buf.n_episode > 0])
            buffer_indices, episode_indices = prioritized_sampling(
                n_sample=n_sample, priorities=priorities, array_lengths=n_episodes,
                sample_length=1, **kwargs # e.g. temperature
            )

        else: raise NotImplementedError
        n_buffer_samples = np.array([(buffer_indices == n).sum() for n in range(n_buffer)])
        episode_indices = numpy_split(episode_indices, split_shapes=n_buffer_samples, dim=-1)
        episode_indices = [np.sort(e) if e.size > 0 else None for e in episode_indices]
        return n_buffer_samples, episode_indices, None

    else:
        assert (sample_length == 1) or (type_buffer == "episode_buffer")
        horizons = []
        for buffer_idx in range(n_buffer):
            if buffers[buffer_idx].n_transition == 0: continue
            if isinstance(buffers[buffer_idx], EpisodeBuffer):
                horizons.append(buffers[buffer_idx].get_horizon())
            else:
                horizons.append(np.array([buffers[buffer_idx].n_transition]))
        horizons = np.concatenate(horizons)

        if method == "uniform":
            pseudo_episode_indices, transition_indices = uniform_sampling(
                n_sample=n_sample, array_lengths=horizons, sample_length=sample_length
            )
            key = pseudo_episode_indices.argsort()
            pseudo_episode_indices = pseudo_episode_indices[key]
            transition_indices = transition_indices[key]

        elif method == "prioritized":
            priorities = []
            for buffer_idx in range(n_buffer):
                if buffers[buffer_idx].n_transition == 0: continue
                if isinstance(buffers[buffer_idx], EpisodeBuffer):
                    priorities.extend(buffers[buffer_idx].get_output_transition_priority())
                else:
                    priorities.append(buffers[buffer_idx].get_output_priority())
            priorities = np.concatenate(priorities)
            pseudo_episode_indices, transition_indices = prioritized_sampling(
                n_sample=n_sample, priorities=priorities, array_lengths=horizons,
                sample_length=sample_length, **kwargs # e.g. temperature
            )
        else: raise NotImplementedError

        n_pseudo_episodes = np.array([buf.n_episode if isinstance(buf, EpisodeBuffer) else 1 for buf in buffers])
        n_pseudo_episodes_cumsum = n_pseudo_episodes.cumsum()
        buffer_indices = np.searchsorted(n_pseudo_episodes_cumsum - 1, pseudo_episode_indices)
        pseudo_episode_offsets = np.concatenate(([0], n_pseudo_episodes_cumsum[:-1]))[buffer_indices]
        pseudo_episode_indices = pseudo_episode_indices - pseudo_episode_offsets

        n_buffer_samples = np.array([(buffer_indices == n).sum() for n in range(n_buffer)])
        episode_indices = numpy_split(pseudo_episode_indices, split_shapes=n_buffer_samples, dim=-1)
        episode_indices = [e if e.size > 0 else None for e in episode_indices]
        episode_indices = [e if isinstance(buf, EpisodeBuffer) else None for e, buf in zip(episode_indices, buffers)]
        transition_indices = numpy_split(transition_indices, split_shapes=n_buffer_samples, dim=-1)
        transition_indices = [t if t.size > 0 else None for t in transition_indices]
        return n_buffer_samples, episode_indices, transition_indices

def get_multiple_sample_multiple_priority(
    buffers:List[Union[TransitionBuffer, EpisodeBuffer]], n_sample:int, name:List[str]=None,
    sample_length:int=1, method:str="uniform", mode_episode:bool=False, 
    mode_info:bool=False, **kwargs
):
    n_buffer_samples, episode_indices, transition_indices = multiple_sample_multiple_priority_index(
        buffers=buffers, n_sample=n_sample, sample_length=sample_length, method=method,
        mode_episode=mode_episode, **kwargs # e.g. temperature
    )
    batch = get_multiple_sample_by_index(
        buffers=buffers, episode_indices=episode_indices, transition_indices=transition_indices,
        sample_length=sample_length, name=name
    )
    if mode_info:
        return batch, n_buffer_samples, episode_indices, transition_indices
    else:
        return batch


# NOTE: Pi(z | s) + Pi(z | s, e)
class ResidualNormalMLP(NormalMLP):
    @Base.save_input(exclude=["fixed_model"])
    def __init__(self, d_state:int, d_high_action:int, d_task:int, d_hiddens:List[int], fixed_model:Module,
                 mode_limit_std:str="hard", log_std_range:Optional[Tuple]=(-10, 2),
                 **kwargs):
        super().__init__(
            d_inputs=[d_state, d_task], d_outputs=[d_high_action], d_hiddens=d_hiddens,
            mode_limit_std=mode_limit_std, log_std_range=log_std_range, **kwargs
        )
        fixed_model = copy.deepcopy(fixed_model).eval().requires_grad_(False)
        self._fixed_model = (fixed_model, )
        
    def _to_device(self) -> None:
        if torch.device(self.device) != torch.device(self._fixed_model[0].device):
            self._fixed_model[0].to(self.device)
    
    def forward(self, s, e):
        self._to_device()
        prior_mean, prior_log_std = self._fixed_model[0](s) # (d_batch, d_high_action), (d_batch, d_high_action)
        mean, log_std = super().forward(s, e) # (d_batch, d_high_action), (d_batch, d_high_action)
        res_mean = prior_mean + mean
        res_log_std = prior_log_std + log_std
        return res_mean, res_log_std

# NOTE: task encoder: q(e | s, z, r, s', d)
class TaskEncoder(Stochastic, Module):
    @Base.save_input()
    def __init__(self, d_state:int, d_action:int, d_task:int, d_hidden:int, n_hidden:int):
        super().__init__(dist="normal")
        self.d_state, self.d_action, self.d_task, self.d_hidden = d_state, d_action, d_task, d_hidden
        self.net = SetTransformer(
            in_dim=2*d_state + d_action + 2, out_dim=2*d_task,
            hidden_dim=d_hidden, n_attention=n_hidden, n_mlp_layer=n_hidden,
            activation="relu"
        )
        self.prior = Normal(torch.zeros(d_task), torch.ones(d_task))
        self.min_scale = 0.001
    
    def forward(self, transition): # (d_batch, dTransition, d_state*2+d_action+2)
        assert (transition.shape[-1] == 2*self.d_state + self.d_action + 2) and transition.ndim == 3
        out = self.net(transition) # (d_batch, d_task*2)
        mean, scale_std = out.chunk(2, dim=-1) # (d_batch, d_task), (d_batch, d_task)
        std = self.min_scale + F.softplus(inverse_softplus(torch.tensor(1.0)) + scale_std)
        return mean, std
    
    def dist(self, *args):
        mean, std = self.forward(*args)
        return self._dist(mean, std) # (d_batch, d_task) ~ Normal
    
    @torch.no_grad()
    @Module.transform_data(3)
    def _inference(self, transition, deterministic=True): # ([d_batch], dTransition)
        return self.sample(transition, deterministic=deterministic) # ([d_batch], d_task)

class KLReg(Module):
    @Base.save_input()
    def __init__(self, init):
        super().__init__()
        self.param = nn.Parameter(
            inverse_softplus(torch.tensor(init))
        )
    
    def forward(self):
        return F.softplus(self.param)
    
    @torch.no_grad()
    def _inference(self):
        return F.softplus(self.param).detach()

# ------------------

class SISL(MainModule):
    @Base.save_input(exclude=["high_qs", "high_policy", "task_encoder", "skill_prior"])
    def __init__(self, 
                 high_qs:List[Module], high_q_lr:float, high_q_target_update_freq:int, high_q_target_update_ratio:float,
                 high_policy:Module, high_pi_lr:float, task_encoder:Module, task_encoder_lr:float,
                 high_pi_reg_train:bool, high_pi_reg_init:float, high_pi_reg_target:float, high_pi_reg_lr:float,
                 task_encoder_reg_train:bool, task_encoder_reg_init:float, task_encoder_reg_target:float, task_encoder_reg_lr:float,
                 kl_clip:int, gamma:float, n_train_task:int, skill_prior:Module,
                 d_high_action:int,
                 use_high_pi_reg_increase_only:bool,
                 high_pi_reg_max:Optional[float]=None, task_encoder_reg_max:Optional[float]=None,
                 ):
        super().__init__()
        
        # --- Q ---
        self.high_q_lr = high_q_lr
        self.high_q_target_update_freq = high_q_target_update_freq
        self.high_q_target_update_ratio = high_q_target_update_ratio
        self.high_qs = nn.ModuleList(high_qs)
        self.target_high_qs = copy.deepcopy(self.high_qs).eval().requires_grad_(False)
        self.optimizer_q_high = optim.Adam(self.high_qs.parameters(), lr=self.high_q_lr)
        
        # --- Pi ---
        self.high_pi_lr = high_pi_lr
        self.high_policy = high_policy
        self.optimizer_pi = optim.Adam(self.high_policy.parameters(), lr=self.high_pi_lr)
        
        # --- encoder ---
        self.task_encoder_lr = task_encoder_lr
        self.task_encoder = task_encoder
        self.optimizer_task_encoder = optim.Adam(self.task_encoder.parameters(), lr=self.task_encoder_lr)
        
        # --- policy reg ---
        self.high_pi_reg_train   = high_pi_reg_train
        self.high_pi_reg_init    = high_pi_reg_init
        self.high_pi_reg_target  = high_pi_reg_target
        self.high_pi_reg_lr      = high_pi_reg_lr
        self.high_pi_reg_max     = high_pi_reg_max
        if self.high_pi_reg_train: # policy KL-coefficient
            self.alpha = KLReg(self.high_pi_reg_init)
            self.optimizer_alpha = optim.Adam(self.alpha.parameters(), lr=self.high_pi_reg_lr)
        
        # --- encoder reg ---
        self.task_encoder_reg_train  = task_encoder_reg_train
        self.task_encoder_reg_init   = task_encoder_reg_init
        self.task_encoder_reg_target = task_encoder_reg_target
        self.task_encoder_reg_lr     = task_encoder_reg_lr
        self.task_encoder_reg_max    = task_encoder_reg_max
        if self.task_encoder_reg_train: # task encoder KL-coefficient
            self.betas = nn.ModuleList([KLReg(self.task_encoder_reg_init) for _ in range(n_train_task)])
            self.optimizer_beta = optim.Adam(self.betas.parameters(), lr=self.task_encoder_reg_lr)
        
        # --- etc. ---
        self._step = 0
        self.kl_clip = kl_clip
        self.gamma = gamma
        self.skill_prior = skill_prior # p(z | s_0)
        self.d_high_action = d_high_action
        
        self.use_high_pi_reg_increase_only = use_high_pi_reg_increase_only
        
    
    def get_prior_dist(self, state, e=None): # (d_batch, d_state), (d_batch, d_task)
        return self.skill_prior.dist(state) # (d_batch, d_high_action)~Normal

    def update_target(self):
        if is_freq(self._step, self.high_q_target_update_freq):
            soft_update_param(self.target_high_qs, self.high_qs, tau=self.high_q_target_update_ratio)

    def update(self, task_indices, task_dists, samples:List):
        self._step += 1
        self.update_target()
        
        # --- critic, task_encoder train ---
        output_critic = self.loss_critic(samples)
        output_taskencoder = self.loss_encoder(task_indices, task_dists)
        loss_q_taskencoder = output_critic.loss_q + output_taskencoder.loss_taskencoder
        
        self.optimizer_q_high.zero_grad()
        self.optimizer_task_encoder.zero_grad()
        loss_q_taskencoder.backward()
        self.optimizer_q_high.step()
        self.optimizer_task_encoder.step()
        
        # --- task_encoder regularizer train ---
        kl_taskencoder_std = output_taskencoder.kl_taskencoder_std # enc_kls 
        betas = output_taskencoder.betas # enc_regs
        loss_taskencoder_reg = (betas * (self.task_encoder_reg_target - kl_taskencoder_std.detach())).mean() # scalar
        
        self.optimizer_beta.zero_grad()
        loss_taskencoder_reg.backward()
        for reg in self.betas:
            if reg.param.grad is not None:
                if self.task_encoder_reg_max is not None and reg().item() >= self.task_encoder_reg_max:
                    reg.param.grad.data.clamp_(min=0)
                reg.param.grad.data.clamp_(max=0)
        self.optimizer_beta.step()
        
        # --- policy train ---
        output_actor = self.lossActor(samples)
        loss_pi = output_actor.loss_pi
        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()
        
        # --- policy regularizer train ---
        kl_pi_prior = output_actor.kl_pi_prior # (n_high_sample_task*d_batch, 1)
        alpha = self.alpha()
        loss_pi_reg = (alpha * (self.high_pi_reg_target - kl_pi_prior.detach())).mean()
        
        self.optimizer_alpha.zero_grad()
        loss_pi_reg.backward()
        if self.high_pi_reg_max is not None and self.alpha().item() >= self.high_pi_reg_max:
            self.alpha.param.grad.data.clamp_(min=0)
        if self.use_high_pi_reg_increase_only:
            self.alpha.param.grad.data.clamp_(max=0)
        self.optimizer_alpha.step()
        # ---
        
        # --- log ---
        output_critic.loss_q = output_critic.loss_q.detach().cpu().item()
        return AttrDict(
            **output_critic, # qf loss
            loss_q_taskencoder = loss_q_taskencoder.detach().cpu().item(), # qf_enc_loss
            loss_taskencoder=output_taskencoder.loss_taskencoder.detach().cpu().item(), # enc_reg_loss
            kl_taskencoder_std=kl_taskencoder_std.detach().mean().cpu().item(), # enc_post_kl
            loss_pi=output_actor.loss_pi.detach().cpu().item(), # policy_loss
            kl_pi_prior=kl_pi_prior.detach().mean().cpu().item(), # policy_post_kl
            loss_alpha=loss_pi_reg.detach().cpu().item(),
            alpha=alpha.detach().cpu().item(), # policy_post_reg
            loss_beta=loss_taskencoder_reg.detach().cpu().item(),
            beta=torch.stack([b() for b in self.betas]).detach().mean().cpu().item(),
        )
    
    def loss_critic(self, samples:list):
        state, high_action, reward, next_state, done, e = samples # (n_high_sample_task*d_batch, -)
        
        # --- estimate ---
        # Q(s, z, e)
        estimate_qs = [ # [(n_high_sample_task*d_batch, 1), (n_high_sample_task*d_batch, 1), ...]
            q(state, high_action, e) for q in self.high_qs # (n_high_sample_task*d_batch, 1)
        ]
        # --- target ---
        with torch.no_grad():
            next_pi_high_action, next_pi_dist = self.high_policy.sample(next_state, e, with_dist=True) # (n_high_sample_task*d_batch, d_high_action), (n_high_sample_task*d_batch, d_high_action)~Normal
            next_prior_dist = self.get_prior_dist(next_state).detach() # (n_high_sample_task*d_batch, d_high_action)~Normal
            next_kl_pi_prior = kl_divergence(next_pi_dist, next_prior_dist, mode_multivariate=True) # (n_high_sample_task*d_batch, 1)
            scales = next_kl_pi_prior.detach().clamp(0, self.kl_clip) / next_kl_pi_prior.detach()
            next_kl_pi_prior = next_kl_pi_prior*scales
            
            next_target_qs = [ # [(n_high_sample_task*d_batch, 1), (n_high_sample_task*d_batch, 1), ...]
                target_q(next_state, next_pi_high_action, e) for target_q in self.target_high_qs # (n_high_sample_task*d_batch, 1)
            ]
            next_target_min_q = torch.cat(next_target_qs, dim=1).min(dim=1, keepdim=True).values # (n_high_sample_task*d_batch, nQ) -> (n_high_sample_task*d_batch, 1)
            next_target_v = next_target_min_q - self.alpha().detach() * next_kl_pi_prior # (n_high_sample_task*d_batch, 1)
            # target_q = reward + (1 - done) * self.gamma * next_target_v # (n_high_sample_task*d_batch, 1)
            target_q = reward + done.logical_not() * self.gamma * next_target_v # (n_high_sample_task*d_batch, 1)
            
        loss_qs = [
            F.mse_loss(estimate_q, target_q) for estimate_q in estimate_qs # scalar
        ]
        loss_q = torch.stack(loss_qs).mean()
        return AttrDict(
            loss_q=loss_q,
            **{f'loss_q{i+1}': loss_qs[i].detach().cpu().item() for i in range(len(loss_qs))},
            **{f'q{i+1}': estimate_qs[i].detach().mean().cpu().item() for i in range(len(estimate_qs))},
        )
        
    def loss_encoder(self, task_indices, task_dists):
        kl_taskencoder_std = kl_divergence(task_dists, self.task_encoder.prior, mode_multivariate=True) # (n_high_sample_task, 1)
        # betas = torch.stack([self.betas[task_idx]() for task_idx in task_indices], dim=0).unsqueeze(dim=-1) # (n_high_sample_task, 1)
        temp_betas = [beta() for beta in self.betas] # (23)
        betas = torch.stack([temp_betas[task_idx] for task_idx in task_indices], dim=0).unsqueeze(dim=-1) # (n_high_sample_task, 1)
        loss_taskencoder = (betas.detach() * kl_taskencoder_std).mean() # scalar
        return AttrDict(
            loss_taskencoder=loss_taskencoder,        # scalar
            kl_taskencoder_std=kl_taskencoder_std,    # (n_high_sample_task, 1)
            betas=betas                     # (n_high_sample_task, 1)
            )
    
    def lossActor(self, samples:list):
        state, high_action, reward, next_state, done, e = samples
        e = e.detach()

        pi_high_action, pi_dist = self.high_policy.sample(state, e, with_dist=True) # (n_high_sample_task*d_batch, d_high_action), (n_high_sample_task*d_batch, d_high_action)~Normal
        with torch.no_grad():
            prior_dist = self.get_prior_dist(state).detach()
        kl_pi_prior = kl_divergence(pi_dist, prior_dist, mode_multivariate=True) # (n_high_sample_task*d_batch, 1)
        
        with Module.freeze_modes(*self.high_qs):
            estimate_qs = [ # [(n_high_sample_task*d_batch, 1), (n_high_sample_task*d_batch, 1), ...]
                q(state, pi_high_action, e) for q in self.high_qs # (n_high_sample_task*d_batch, 1)
            ]
        estimate_min_q = torch.cat(estimate_qs, dim=1).min(dim=1, keepdim=True).values # (n_high_sample_task*d_batch, nQ) -> (n_high_sample_task*d_batch, 1)
        loss_pi = (self.alpha().detach() * kl_pi_prior) - estimate_min_q # (n_high_sample_task*d_batch, 1)
        loss_pi = loss_pi.mean() # scalar
        return AttrDict(
            loss_pi=loss_pi,
            kl_pi_prior=kl_pi_prior,
        )


class AugmentedSAC(SAC):
    @Base.save_input(exclude=["qs", "policy"])
    def __init__(self, *args, bc_ratio:float, **kwargs):
        super().__init__(*args, **kwargs)
        self.bc_ratio = bc_ratio
    
    def loss_critic(self, sample:List[torch.Tensor], curr_conditions, next_conditions) -> dict:
        state, action, reward, next_state, done = sample
        # --- estimate ---
        # Q(s, a, e)
        estimate_qs = [
            q(state, action, *curr_conditions) for q in self.qs # (d_batch, 1)
        ]
        
        # --- target ---
        with torch.no_grad():
            # a'~Pi(- | s, e), log( Pi(a' | s, e) )
            next_pi_action, next_log_pi = self.Pi.sample(next_state, *next_conditions, with_log_prob=True) # (d_batch, d_action), (d_batch, 1)
            # Q'(s', a', e)
            next_target_qs = [
                target_q(next_state, next_pi_action, *next_conditions) for target_q in self.target_qs # (d_batch, 1)
            ]
            # min Q' = min{Q1'(s', a', e), Q2'(s', a', e), ...}
            next_target_min_q = torch.cat(next_target_qs, dim=-1).min(dim=-1, keepdim=True).values # (d_batch, q_n) -> (d_batch, 1)
            # V'(s') = min Q' - alpha * log( Pi(a' | s', e) )
            next_target_v = next_target_min_q - self.alpha * next_log_pi # (d_batch, 1)
            # Q'(s, a, e) = r + gamma * V'(s')
            # target_q = reward + (1 - done) * self.gamma * next_target_v # (d_batch, 1)
            target_q = reward + done.logical_not() * self.gamma * next_target_v # (d_batch, 1)
            
        # --- MSE loss ---
        # MSE{ Q(s, a, e), Q'(s, a, e) }
        loss_qs = [
            F.mse_loss(estimate_q, target_q) for estimate_q in estimate_qs # scalar
        ]
        loss_q = sum(loss_qs) # scalar
        
        return AttrDict(
            loss_q=loss_q,
            **{f'loss_q{i+1}': loss_qs[i].detach().cpu().item() for i in range(len(loss_qs))},
            **{f'q{i+1}': estimate_qs[i].detach().mean().cpu().item() for i in range(len(estimate_qs))},
        )
        
    def loss_actor(self, sample:List[torch.Tensor], curr_conditions, next_conditions) -> AttrDict:
        state, action, reward, next_state, done = sample
        
        # a ~ Pi(- | s, e), log( Pi(a | s, e) )
        pi_action, log_pi = self.Pi.sample(state, *curr_conditions, with_log_prob=True) # (d_batch, d_action), (d_batch, 1)
        with Base.freeze_modes(*self.qs):
            estimate_qs = [
                q(state, pi_action, *curr_conditions) for q in self.qs # (d_batch, 1)
            ]
        # min q = min{ Q1(s, a, e), Q2(s, a, e), ... }
        estimate_min_q = torch.stack(estimate_qs, dim=-1).min(dim=-1, keepdim=True).values # (d_batch, q_n) -> (d_batch, 1)
        loss_pi = (self.alpha * log_pi) - estimate_min_q # (d_batch, 1)
        loss_pi = loss_pi.mean() # scalar
        
        return AttrDict(
            loss_pi=loss_pi,
            entropy_pi=-log_pi.detach().mean().cpu().item(),
        )

    def loss_alpha(self, sample:List[torch.Tensor], curr_conditions, next_conditions) -> AttrDict:
        state, action, reward, next_state, done = sample
        
        with torch.no_grad():
            _pi_action, log_pi = self.Pi.sample(state, *curr_conditions, with_log_prob=True) # (d_batch, 1)
        loss_alpha = -self.log_alpha.exp() * (log_pi + self.target_entropy) # (d_batch, 1)
        loss_alpha = loss_alpha.mean()
        
        return AttrDict(loss_alpha=loss_alpha)
    
    def loss_bc(self, sample:List[torch.Tensor], bc_conditions):
        """KL( Pi_exp(a | s, e) || online_skill_buffer(s, a) )"""
        state, action = sample[0], sample[1]
        pi_dist = self.Pi.dist(state, *bc_conditions)
        loss_bc = -pi_dist.normal.log_prob(action).sum(dim=-1).mean() # scalar
        return loss_bc
        
    def update(
        self, rl_batch:List[torch.Tensor], bc_batch:List[torch.Tensor], use_bc:bool,
        curr_conditions, next_conditions, bc_conditions,
    ):
        self._step += 1
        self.update_target()
        result = AttrDict()
        
        # --- Q ---
        if is_freq(self._step, self.q_train_freq):
            for _ in range(self.q_train_freq):
                result_critic = self.loss_critic(rl_batch, curr_conditions, next_conditions)
                self.optimizer_q.zero_grad()
                result_critic.loss_q.backward()
                self.optimizer_q.step()
                
            # for log
            result_critic.loss_q = result_critic.loss_q.detach().cpu().item()
            result.update(result_critic)
            
        if is_freq(self._step, self.pi_train_freq):
            # --- Pi ---
            for _ in range(self.pi_train_freq):
                result_actor = self.loss_actor(rl_batch, curr_conditions, next_conditions)
                # if self.use_bc:
                if use_bc:
                    loss_bc = self.loss_bc(bc_batch, bc_conditions)
                    loss_pi = result_actor.loss_pi + loss_bc*self.bc_ratio
                else:
                    loss_bc = None
                    loss_pi = result_actor.loss_pi
                self.optimizer_pi.zero_grad()
                loss_pi.backward()
                self.optimizer_pi.step()
            
                if self.dynamic_alpha:
                    # --- alpha ---
                    result_alpha = self.loss_alpha(rl_batch, curr_conditions, next_conditions)
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
        
        return AttrDict(
            **result,
            loss_bc=loss_bc.detach().cpu().item() if loss_bc is not None else None,
        )


def rollout_skill(
    hyper_conf:AttrDict, mode_prior:bool,
    task_indices:list, tasks:list, conc_collector,
    encoder_buffers:list, task_encoder:Module, high_policy:Module, low_policy:Module,
):
    """high-policy/skill-policy rollout"""
    d_seq                       = hyper_conf.d_seq
    n_task_embedding_transition = hyper_conf.n_task_embedding_transition

    for task_idx in task_indices:
        if mode_prior:
            # NOTE: e~q(E) prior
            e = task_encoder.prior.sample().cpu() # (d_task)
        else:
            # NOTE: e~p(E|C) posterior
            batch = encoder_buffers[task_idx].get_sample(n_task_embedding_transition).torch().to(task_encoder.device)
            taskencoder_input = batch.cat() # (n_task_embedding_transition, -)
            with task_encoder.inference_mode():
                e = task_encoder(taskencoder_input, deterministic=False).cpu() # (d_task)
        conc_collector.submit(tasks[task_idx], e, high_policy, low_policy, mode="skill")
    raw_episodes = conc_collector.wait()
        
    low_episodes = []
    high_episodes = []
    for raw_episode in raw_episodes:
        # NOTE: low-level transition: (state, high_action, low_action, reward, next_state, done)
        low_episode = Episode(*raw_episode)
        horizon = low_episode.n_transition
        start_indices = np.arange(0, horizon, d_seq)
        end_indices = start_indices + d_seq - 1
        end_indices[-1] = min(end_indices[-1], horizon-1)
        
        # NOTE: high-level transition: (state, high_action, high_reward, next_state, done)
        high_episode = Episode(
            raw_episode[0][start_indices],
            raw_episode[1][start_indices],
            np.pad(raw_episode[3].flatten(), (0, (d_seq - (horizon%d_seq))%d_seq)).reshape(-1, d_seq).sum(axis=-1, keepdims=True),
            raw_episode[4][end_indices],
            raw_episode[5][end_indices],
        )
        
        low_episodes.append(low_episode)
        high_episodes.append(high_episode)
    return low_episodes, high_episodes

def rollout_exploration(
    hyper_conf:AttrDict,
    task_indices:list, tasks:list, conc_collector,
    exploration_policy:Module,
):
    n_task = len(tasks)

    for task_idx in task_indices:
        e = to_onehot(torch.tensor(task_idx), num_classes=n_task)
        conc_collector.submit(tasks[task_idx], e, exploration_policy, mode="exploration")
    # NOTE: low-level transition: (state, action, reward, next_state, done, score)
    raw_episodes = conc_collector.wait()
    return [Episode(*raw_episode) for raw_episode in raw_episodes]


@torch.no_grad()
def get_task_dist(encoder_buffers:list, n_task_embedding_transition:int, task_encoder:Module):
    n_task = len(encoder_buffers)
    device = task_encoder.device

    task_batch = get_multiple_sample_per_buffer(buffers=encoder_buffers, n_sample_per_buffer=n_task_embedding_transition)
    task_batch = Batch.stack(*task_batch).torch().to(device).cat()
    task_dists = task_encoder.dist(task_batch)
    task_dists = [type(task_dists)(task_dists.mean[i], task_dists.stddev[i]) for i in range(n_task)]
    return task_dists

def get_multiple_sample_with_task(
    task_indices:List[int],
    d_batch_per_buffer:int, n_task_embedding_transition:int,
    buffers:List, encoder_buffers:List,
    task_encoder:Module, name:Optional[List[str]]=None
):
    device = task_encoder.device

    # NOTE: batch sampling
    batch = get_multiple_sample_per_buffer(
        buffers=buffers, n_sample_per_buffer=d_batch_per_buffer, buffer_indices=task_indices, 
        name=name
    ) # [(d_batch, -), (d_batch, -), ...]
    batch = Batch.merge(*batch).torch().to(device) # [(n_sample_task*d_batch, -), (n_sample_task*d_batch, -), ...]

    # NOTE: task sampling
    task_batch = get_multiple_sample_per_buffer(
        buffers=encoder_buffers, n_sample_per_buffer=n_task_embedding_transition, buffer_indices=task_indices,
        name=None
    )
    task_batch = Batch.stack(*task_batch).torch().to(device).cat() 
    # [(n_sample_task, d_batch, -), (n_sample_task, d_batch, -), ...] -> (n_sample_task, d_batch, -)

    task_dists = task_encoder.dist(task_batch) # (n_sample_task, d_task)
    e = task_dists.rsample(sample_shape=(d_batch_per_buffer, )).transpose(0, 1) # (n_sample_task, d_batch_per_buffer, d_task)
    e = e.reshape(-1, e.shape[-1]) # (n_sample_task*d_batch_per_buffer, d_task)
    batch.add_item(e, name="task")

    # [(n_sample_task*d_batch_per_buffer, -), (n_sample_task*d_batch_per_buffer, -), ...]
    return batch, task_dists


def update_high_model(
    hyper_conf:AttrDict,
    high_buffers:List[EpisodeBuffer], encoder_buffers:List[EpisodeBuffer],
    high_model:MainModule, task_encoder:Module,
):
    assert high_model.device == task_encoder.device
    n_high_sample_task          = hyper_conf.n_high_sample_task
    d_high_batch                = hyper_conf.d_high_batch
    n_task_embedding_transition = hyper_conf.n_task_embedding_transition
    n_task = len(encoder_buffers)
    # device = high_model.device

    # NOTE: prepare sample: (state, high_action, high_reward, next_state, done, task)
    task_indices = np.random.randint(low=0, high=n_task, size=n_high_sample_task)
    batch, task_dists = get_multiple_sample_with_task(
        task_indices=task_indices, d_batch_per_buffer=d_high_batch,
        n_task_embedding_transition=n_task_embedding_transition,
        buffers=high_buffers, encoder_buffers=encoder_buffers,
        task_encoder=task_encoder
    )

    result = high_model.update(task_indices, task_dists, batch)
    # loss_q, loss_qx, qx, loss_q_taskencoder, loss_taskencoder, kl_taskencoder_std
    # loss_pi, kl_pi_prior
    # loss_beta, alpha, loss_alpha, beta
    return result


def update_exp_model(
    hyper_conf:AttrDict,
    exploration_buffers:List[EpisodeBuffer], online_skill_buffers:List[EpisodeBuffer],
    exp_model:MainModule,
    state_rms:list, rnd:Module, target_rnd:Module, optimizer_rnd,
    use_bc:bool,
):
    """ Q_exp, Pi_exp, reward_model, RND update """
    n_exp_sample_task           = hyper_conf.n_exp_sample_task
    n_task_embedding_transition = hyper_conf.n_task_embedding_transition
    d_exp_batch                 = hyper_conf.d_exp_batch

    use_rnd                 = hyper_conf.use_rnd
    use_exploration_score   = hyper_conf.use_exploration_score
    use_rnd_state_dropout   = hyper_conf.use_rnd_state_dropout
    rnd_state_dropout       = hyper_conf.rnd_state_dropout
    rnd_ext_ratio           = hyper_conf.rnd_ext_ratio
    rnd_int_ratio           = hyper_conf.rnd_int_ratio
    rnd_train_freq          = hyper_conf.rnd_train_freq
    d_score_onehot          = hyper_conf.d_score_onehot
    
    exp_bc_ratio      = hyper_conf.exp_bc_ratio
    n_task = len(exploration_buffers)
    device = exp_model.device
    
    if exp_bc_ratio == 0:
        use_bc = False
    result = AttrDict()
    

    # NOTE: batch sampling
    task_indices = np.random.randint(low=0, high=n_task, size=n_exp_sample_task)
    with torch.no_grad():
        batches = []
        for task_idx in task_indices:
            batch = get_multiple_sample(
                buffers=[exploration_buffers[task_idx], online_skill_buffers[task_idx]], n_sample=d_exp_batch,
            )
            batches.append(batch)
        batch = Batch.merge(*batches).torch().to(device)
        
        states, actions, reward_exts, next_states, dones, scores = batch
        score_onehots = to_onehot(scores, num_classes=d_score_onehot)
        task_onehots = torch.tensor(task_indices).to(device)
        task_onehots = to_onehot(task_onehots, num_classes=n_task).repeat(1, d_exp_batch).reshape(-1, n_task)
    
    if use_rnd:
        with torch.no_grad():
            next_state_splits = next_states.split(d_exp_batch, dim=0)
            # NOTE: update state_rms
            for batch_idx in range(n_exp_sample_task):
                state_rms[task_indices[batch_idx]].update(next_state_splits[batch_idx])
            # NOTE: normalize next_state
            next_state_norm_splits = [state_rms[task_indices[batch_idx]](next_state_splits[batch_idx]) for batch_idx in range(n_exp_sample_task)]
            next_state_norm = torch.cat(next_state_norm_splits, dim=0)
            next_state_norm_target = next_state_norm.clone()
            next_state_norm_pred = next_state_norm.clone()
            if use_rnd_state_dropout:
                next_state_norm_pred = F.dropout(next_state_norm_pred, p=rnd_state_dropout)
            rnd_conditions = []
            if use_exploration_score:
                rnd_conditions.append(score_onehots)
            rnd_conditions.append(task_onehots)

        targets = target_rnd(next_state_norm_target, *rnd_conditions)
        preds = rnd(next_state_norm_pred, *rnd_conditions)
        reward_norms = ((preds - targets)**2).sum(dim=-1, keepdim=True)
        if is_freq(exp_model._step, rnd_train_freq):
            loss_rnd = (reward_norms / preds.shape[-1]).mean() # mse loss
            optimizer_rnd.zero_grad()
            loss_rnd.backward()
            optimizer_rnd.step()
            loss_rnd = loss_rnd.detach().cpu().item()
        else:
            loss_rnd = None
        reward_norms = reward_norms.detach()
        reward_norms_split = reward_norms.split(d_exp_batch)
        reward_ints = []
        for reward_norm in reward_norms_split:
            reward_ints.append(
                (reward_norm - reward_norm.min()) / (reward_norm.max() - reward_norm.min() + 1e-12)
            )
        reward_ints = torch.cat(reward_ints, dim=0)
        rewards = reward_exts*rnd_ext_ratio + reward_ints*rnd_int_ratio
    else:
        rewards = reward_exts
    result.update(loss_rnd=loss_rnd)
    
    # NOTE: update exploration_model
    curr_conditions, next_conditions = [], []
    if use_exploration_score:
        prev_score_onehots = to_onehot(scores - reward_exts, num_classes=d_score_onehot)
        curr_conditions.append(prev_score_onehots)
        next_conditions.append(score_onehots)
    curr_conditions.append(task_onehots)
    next_conditions.append(task_onehots)
    batch = Batch(states, actions, rewards, next_states, dones)
    
    # NOTE: bc sample
    bc_batch, bc_conditions = None, None
    if use_bc:
        bc_batch = get_multiple_sample_per_buffer(
            buffers=online_skill_buffers, n_sample_per_buffer=d_exp_batch, buffer_indices=task_indices,
        )
        bc_batch = Batch.merge(*bc_batch).torch().to(device)
        bc_conditions = []
        if use_exploration_score:
            bc_score_onehots = to_onehot((bc_batch.score - bc_batch.low_reward), num_classes=d_score_onehot)
            bc_conditions.append(bc_score_onehots)
        bc_conditions.append(task_onehots)
    
    result_rl = exp_model.update(
        rl_batch=batch, bc_batch=bc_batch, use_bc=use_bc,
        curr_conditions=curr_conditions, next_conditions=next_conditions, bc_conditions=bc_conditions,
    )
    result.update(result_rl)
    return result


def update_from_batch(
    model, optimizer, loss_fn:Callable, target:torch.Tensor, inputs:Tuple[torch.Tensor],
):
    pred = model(*inputs)
    loss = loss_fn(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss = loss.detach().cpu().item()
    pred = pred.detach().cpu()
    pred = pred.item() if pred.numel() == 1 else pred.mean().item()
    return AttrDict(loss=loss, pred=pred)

def update_reward_model(
    hyper_conf:AttrDict, task_indices:np.ndarray, n_task:int,
    exploration_buffers:List[EpisodeBuffer], online_skill_buffers:List[EpisodeBuffer],
    reward_model:Module, optimizer_reward,
):
    d_exp_batch             = hyper_conf.d_exp_batch
    device                  = reward_model.device
    
    # NOTE: batch sampling
    with torch.no_grad():
        batches = []
        for task_idx in task_indices:
            batch = get_multiple_sample(
                buffers=[exploration_buffers[task_idx], online_skill_buffers[task_idx]], n_sample=d_exp_batch,
                name=["state", "low_action", "low_reward", "score"],
            )
            batches.append(batch)
        batch = Batch.merge(*batches).torch().to(device)
        state, action, reward, score = batch
        
        conditions = []
        task_onehot = torch.tensor(task_indices).to(device)
        task_onehot = to_onehot(task_onehot, num_classes=n_task).repeat(1, d_exp_batch).reshape(-1, n_task)
        conditions.append(task_onehot)
        
    # NOTE: train
    result = update_from_batch(
        model=reward_model, optimizer=optimizer_reward, loss_fn=F.mse_loss, target=reward,
        inputs=[state, action, *conditions]
    )
    return result

@torch.no_grad()
def calc_episode_priority(
    hyper_conf:AttrDict, episode, n_task:int, reward_model:Module,
    task_index:Optional[int]=None,
):
    reward_distance_threshold   = hyper_conf.reward_distance_threshold
    reward_bound                = hyper_conf.reward_bound
    gamma                       = hyper_conf.skill_gamma
    return_min                  = hyper_conf.return_min
    return_max                  = hyper_conf.return_max
    device                      = reward_model.device
    horizon                     = episode.n_transition

    # NOTE: sample
    items = episode.as_batch().torch().to(device)
    if items.n_item > 2: # online sample
        state, action, reward_true = items[:3]
    else: # offline sample
        state, action = items[:2]
        reward_true = None

    states = state.unsqueeze(dim=0).expand(n_task, *state.shape) # (n_task, horizon, d_state)
    actions = action.unsqueeze(dim=0).expand(n_task, *action.shape) # (n_task, horizon, d_action)
    conditions = []
    es = torch.eye(n_task).to(device).repeat(1, horizon).reshape(n_task, horizon, -1) # (n_task, horizon, n_task)
    conditions.append(es)
    reward_pred = reward_model(states, actions, *conditions).detach().cpu().numpy() # (n_task, horizon, 1)
    reward_pred = reward_pred > reward_bound
    states = states.cpu().numpy()
    reward_fakes = []
    for task_idx in range(n_task):
        reward_pred_split = reward_pred[task_idx]
        reward_fake = np.zeros_like(reward_pred_split)
        high_reward_indices = np.where(reward_pred_split > 0)[0]
        while True:
            if len(high_reward_indices) == 0 or reward_fake.sum() >= return_max:
                break
            start_idx = high_reward_indices[0]
            reward_fake[start_idx] = reward_pred_split[start_idx]
            base_state = states[task_idx][start_idx]
            compare_states = states[task_idx, high_reward_indices]

            distance = np.linalg.norm(base_state - compare_states, axis=-1)
            far_distance_indices = np.where(distance > reward_distance_threshold)[0]
            if len(far_distance_indices) == 0:
                break
            end_idx = far_distance_indices[0]
            high_reward_indices = high_reward_indices[end_idx:]
        reward_fakes.append(reward_fake)
    reward_fakes = np.stack(reward_fakes, axis=0).squeeze(axis=-1) # (n_task, horizon)
    
    full_returns = reward_fakes.sum(axis=-1) # (n_task)
    if reward_true is not None:
        return_true = reward_true.cpu().numpy().sum()
        full_returns[task_index] = max(full_returns[task_index], return_true)
    
    gammas = np.power(gamma, np.arange(horizon)) # (horizon)
    gamma_returns = (reward_fakes * gammas).sum(axis=-1) # (n_task, horizon)
    if reward_true is not None:
        return_true = (reward_true.cpu().numpy().squeeze() * gammas).sum()
        gamma_returns[task_index] = max(gamma_returns[task_index], return_true)
    return full_returns, gamma_returns


def update_skill_model(
    hyper_conf:AttrDict, n_task:int,
    skill_model:MainModule, reward_model:Module,
    offline_skill_buffer:EpisodeBuffer, online_skill_buffers:List[EpisodeBuffer],
    online_mean:float,
):
    assert skill_model.device == reward_model.device
    d_seq                       = hyper_conf.d_seq
    n_skill_episode             = hyper_conf.n_skill_episode
    n_skill_sample_per_episode  = hyper_conf.n_skill_sample_per_episode
    temperature                 = hyper_conf.skill_priority_temperature
    
    device = skill_model.device
    buffers = [offline_skill_buffer, *online_skill_buffers]
    
    n_episodes = np.array([buf.n_episode for buf in buffers])
    offline_priorities = offline_skill_buffer.get_output_episode_priority().max(axis=-1)
    online_priorities = np.array(online_mean).repeat(n_episodes[1:].sum())
    priorities = np.concatenate([offline_priorities, online_priorities])
    buffer_indices, episode_indices = prioritized_sampling(
        n_sample=n_skill_episode, priorities=priorities, array_lengths=n_episodes,
        temperature=temperature
    )
    
    batches = []
    for buffer_idx, episode_idx in zip(buffer_indices, episode_indices):
        episode = buffers[buffer_idx][episode_idx]
        transition_indices = np.random.randint(0, episode.n_transition-d_seq+1, size=n_skill_sample_per_episode)
        batch = buffers[buffer_idx].get_sample_by_index(
            episode_indices=[episode_idx], transition_indices=transition_indices, sample_length=d_seq, 
            name=["state", "low_action"]
        )
        batches.append(batch)
    batch = Batch.merge(*batches).torch().to(device)
    
    n_off = (buffer_indices == 0).sum()
    n_on = (buffer_indices != 0).sum()
    
    states, actions = batch
    result = skill_model.update(states, actions)
    
    result.update(dict(
        n_offline=n_off, n_online=n_on,
    ))
    return result

def update_buffer_priority(
    hyper_conf:AttrDict, n_task:int,
    offline_skill_buffer:EpisodeBuffer,
    reward_model:Module, 
):
    episode, episode_idx = offline_skill_buffer.get_episode_sample(
        n_episode=1, mode_info=True,
    )
    episode, episode_idx = episode[0], episode_idx[0]
    priority = calc_episode_priority(
        hyper_conf=hyper_conf, episode=episode, n_task=n_task, 
        reward_model=reward_model, task_index=None,
    )
    offline_skill_buffer.set_output_episode_priority(
        index=episode_idx, priority=[priority[0]]
    )
    return AttrDict(
        full_return=priority[0],
        gamma_return=priority[1],
    )
    
    
    

def dataset2episode(dataset:dict, labels:list):
    terminal_ids = np.where(dataset["terminals"] == True)[0]
    episodes = []
    start_id = 0
    for terminal_id in terminal_ids:
        ep = Episode(*[dataset[label][start_id:terminal_id+1] for label in labels])
        episodes.append(ep)
        start_id = terminal_id + 1
    return episodes # episode list


def main(args):    
    torch.multiprocessing.set_start_method('spawn')
    
    conf = AttrDict(
        ENV                 = args.env,
        DEVICE              = int(args.device) if args.device.isnumeric() else args.device,
        DEVICE_SUB          = [int(d) if d.isnumeric() else d for d in args.device_sub],
        OFF_DATASET_PATH    = args.dataset_path,
        OFF_SKILL_PATH      = args.skill_path,
        MAX_ITERATION       = args.iteration,
        SAVE_FREQ           = args.save_freq,
        EPISODE_MAX_STEP    = None,
    )
    hyper_conf = AttrDict(
        use_warmup_exploration = None,
        n_warmup_exp = 5000,
        # --- buffer ---
        use_posterior_sample_to_exploration         = None,
        use_skill_buffer_input_priority             = True,
        use_skill_buffer_nonzero_reward_sample_only = False,
        use_skill_buffer_episode_clip               = True,
        high_buffer_size                            = None,
        exploration_buffer_size                     = None,
        skill_buffer_size                           = 10000,
        # ------------------
        # --- high-level ---
        # ------------------
        # --- high-agent ---
        high_gamma                  = 0.99,
        d_high_batch                = None,
        n_high_sample_task          = 30,
        n_task_embedding_transition = None,
        high_train_ratio            = 1,
        # --- high-level Q_h(s, z, e) ---
        high_q_n                    = 2,
        high_q_d_hiddens            = None,
        high_q_lr                   = 0.0003,
        high_q_target_update_freq   = 1,
        high_q_target_update_ratio  = 0.005, # target network update ratio
        # --- high-level Pi_h(Z | s, e) ---
        high_pi_d_hiddens           = None,
        high_pi_lr                  = 0.0003,
        high_pi_reg_train           = True,
        high_pi_reg_init            = None,
        high_pi_reg_target          = None,
        high_pi_reg_lr              = 0.0003,
        high_pi_reg_max             = None,
        high_pi_kl_clip             = None,
        use_high_pi_reg_increase_only   = True,
        # --- task encoder q(E | C=(s,z,r,s',d)) ---
        task_encoder_d_hidden       = 128, # task encoder
        task_encoder_n_hidden       = 2,
        task_encoder_lr             = 0.0003,
        task_encoder_reg_train      = True,
        task_encoder_reg_init       = None,
        task_encoder_reg_target     = None,
        task_encoder_reg_lr         = 0.0003,
        task_encoder_reg_max        = None,
        # -----------------
        # --- exploration ---
        # -----------------
        # --- exploration-agent ---
        exp_bc_ratio                = None,
        exp_gamma                   = None,
        exp_train_ratio             = 5,
        n_exp_sample_task           = 30,
        d_exp_batch                 = 256,
        d_score_onehot              = None,
        use_exploration_score       = True,
        # --- exploration Q_exp(s, a, i) ---
        exp_q_n                     = 2,
        exp_q_d_hiddens             = [256, 256, 256, 256],
        exp_q_lr                    = 0.0003,
        target_exp_q_update_freq    = 1,
        target_exp_q_update_ratio   = 0.005,
        # --- exploration Pi_exp(A | s, i) ---
        exp_pi_d_hiddens            = [256, 256, 256, 256],
        exp_pi_lr                   = 0.0003,
        exp_pi_reg_train            = None,
        exp_pi_reg_init             = None,
        target_exp_pi_reg_ratio     = None, # -d_action * value
        # --- RND ---
        use_rnd                     = True,
        use_rnd_state_dropout       = True,
        rnd_state_dropout           = 0.7,
        rnd_d_hiddens               = [128, 128, 128, 128],
        target_rnd_d_hiddens        = [128, 128, 128, 128],
        rnd_d_out                   = 10,
        rnd_ext_ratio               = None,
        rnd_int_ratio               = None,
        rnd_train_freq              = None,
        rnd_lr                      = 0.0003,
        # -------------
        # --- low-level(skill) ---
        # -------------
        # --- reward model ---
        reward_d_hiddens            = [128, 128, 128],
        reward_lr                   = 0.0003,
        n_reward_pretrain_step      = 3000,
        reward_distance_threshold   = None,
        reward_bound                = None,
        return_min                  = None,
        return_max                  = None,
        # --- skill ---
        n_skill_sample_task         = 30,
        skill_gamma                 = 0.99,
        skill_beta                  = 5e-4,
        skill_sync_ratio            = 1,
        skill_sync_freq             = None,
        n_update_priority_episode   = 200,
        n_skill_episode             = 128,
        n_skill_sample_per_episode  = 1, 
        skill_lr                    = 0.001,
        skill_train_ratio           = 5,
        skill_priority_temperature  = None,
        # --- etc. ---
        d_high_action               = 10, # z(skill) dimension
        d_task                      = None,  # e task dimension
        d_seq                       = 10, # skill length
    )
    if conf.ENV == "kitchen":
        conf.update(
            EPISODE_MAX_STEP    = 280,
        )
        hyper_conf.update(
            # high-level
            high_buffer_size            = 3000,
            high_q_d_hiddens            = [128, 128, 128, 128, 128, 128],
            high_pi_d_hiddens           = [128, 128, 128, 128, 128, 128],
            high_pi_reg_init            = 0.03,
            high_pi_reg_target          = 4,
            high_pi_kl_clip             = 6,
            task_encoder_reg_init       = 1e-4,
            task_encoder_reg_target     = 10,
            d_task                      = 6,
            d_high_batch                = 256,
            n_task_embedding_transition = 1024,
            # exploration
            exp_pi_reg_init             = 0.2,
            exp_gamma                   = 0.95,
            exp_bc_ratio                = 0.001,
            target_exp_pi_reg_ratio     = 1.5,
            # skill
            skill_sync_freq             = 1000,
            skill_priority_temperature  = 1.0,
            rnd_train_freq              = 280,
            rnd_ext_ratio               = 5,
            rnd_int_ratio               = 0.1,
            d_score_onehot              = 5,
            reward_distance_threshold   = 1,
            reward_bound                = 0.5,
            return_min                  = 0,
            return_max                  = 4,
            use_warmup_exploration              = False,
            exp_pi_reg_train                    = True,
            exploration_buffer_size             = 100000,
            use_posterior_sample_to_exploration = True,
            high_pi_reg_max                     = None,
            task_encoder_reg_max                = None,
        )

        from utils_kitchen import tasks
        train_tasks = tasks.train_tasks
        env_id = "simpl-kitchen-v0"
    elif conf.ENV == "office":
        conf.update(
            EPISODE_MAX_STEP    = 300,
        )
        hyper_conf.update(
            # high-level
            high_buffer_size            = 3000,
            high_q_d_hiddens            = [128, 128, 128, 128, 128, 128],
            high_pi_d_hiddens           = [128, 128, 128, 128, 128, 128],
            high_pi_reg_init            = 0.03,
            high_pi_reg_target          = 4,
            high_pi_kl_clip             = 6,
            task_encoder_reg_init       = 1e-6,
            task_encoder_reg_target     = 10,
            d_task                      = 6,
            d_high_batch                = 256,
            n_task_embedding_transition = 1024,
            # exploration
            exp_pi_reg_init             = 0.2,
            exp_gamma                   = 0.95,
            exp_bc_ratio                = 0.001,
            target_exp_pi_reg_ratio     = 1.5,
            # skill
            skill_sync_freq             = 1000,
            skill_priority_temperature  = 1.0,
            d_score_onehot              = 9,
            rnd_train_freq              = 300,
            rnd_ext_ratio               = 2,
            rnd_int_ratio               = 0.1,
            reward_distance_threshold   = 0.2,
            reward_bound                = 0.5,
            return_min                  = 0,
            return_max                  = 8,
            use_warmup_exploration              = False,
            exp_pi_reg_train                    = True,
            exploration_buffer_size             = 200000,
            use_posterior_sample_to_exploration = True,
            high_pi_reg_max             = None,
            task_encoder_reg_max        = None,
        )

        from utils_office import tasks
        train_tasks = tasks.train
        env_id = "office-v0"
    elif conf.ENV == "maze":
        conf.update(
            EPISODE_MAX_STEP    = 2000,
        )
        hyper_conf.update(
            # high-level
            high_buffer_size            = 20000,
            high_q_d_hiddens            = [256, 256, 256, 256],
            high_pi_d_hiddens           = [256, 256, 256, 256],
            high_pi_reg_init            = 0.001,
            high_pi_reg_target          = 1,
            high_pi_kl_clip             = 5,
            task_encoder_reg_init       = 3e-6,
            task_encoder_reg_target     = 10,
            d_task                      = 5,
            d_high_batch                = 1024,
            n_task_embedding_transition = 8192,
            # exploration
            exp_pi_reg_init             = 0.1,
            exp_gamma                   = 0.99,
            exp_bc_ratio                = 0.001,
            target_exp_pi_reg_ratio     = 1.5,
            # skill
            skill_sync_freq             = 500,
            skill_priority_temperature  = 0.5,
            rnd_train_freq              = 2000,
            rnd_ext_ratio               = 10,
            rnd_int_ratio               = 0.01,
            d_score_onehot              = 2,
            reward_distance_threshold   = 0,
            reward_bound                = 0.8,
            return_min                  = 0,
            return_max                  = 1,
            # etc.
            use_warmup_exploration              = False,
            exp_pi_reg_train                    = True,
            exploration_buffer_size             = 100000,
            use_posterior_sample_to_exploration = True,
            high_pi_reg_max                     = None,
            task_encoder_reg_max                = None,
        )

        from utils_maze import tasks
        train_tasks = tasks.train40
        env_id = "simpl-maze-size20-seed0-v0"
    elif conf.ENV == "antmaze":
        conf.update(
            EPISODE_MAX_STEP    = 1000,
        )
        hyper_conf.update(
            # high-level
            high_buffer_size            = 20000,
            high_q_d_hiddens            = [128, 128, 128, 128, 128, 128],
            high_pi_d_hiddens           = [128, 128, 128, 128, 128, 128],
            high_pi_reg_init            = 3e-4,
            high_pi_reg_target          = 3,
            high_pi_kl_clip             = 5,
            task_encoder_reg_init       = 3e-7,
            task_encoder_reg_target     = 10,
            d_task                      = 5,
            d_high_batch                = 512,
            n_task_embedding_transition = 4096,
            # exploration
            exp_pi_reg_init             = 0.1,
            exp_gamma                   = 0.99,
            exp_bc_ratio                = 0.001,
            target_exp_pi_reg_ratio     = 1.5,
            # skill
            skill_sync_freq             = 1000,
            skill_priority_temperature  = 0.5,
            d_score_onehot              = 2,
            rnd_train_freq              = 1000,
            rnd_ext_ratio               = 10,
            rnd_int_ratio               = 0.01,
            reward_distance_threshold   = 0,
            reward_bound                = 0.8,
            return_min                  = 0,
            return_max                  = 1,
            use_warmup_exploration              = False,
            exp_pi_reg_train                    = True,
            exploration_buffer_size             = 300000,
            use_posterior_sample_to_exploration = True,
            high_pi_reg_max             = 0.01,
            task_encoder_reg_max        = 5e-6,
            high_pi_lr                  = 1e-4,
            task_encoder_lr             = 1e-4,
        )
        
        from utils_antmaze import size10_tasks as tasks # NOTE: size 10
        train_tasks = tasks.train
        env_id = "antmaze-size10-v0"
    else: raise NotImplementedError
    
    if args.h_buffer_size is not None:
        hyper_conf.high_buffer_size = args.h_buffer_size
    if args.h_kld is not None:
        hyper_conf.high_pi_reg_init = args.h_kld
    if args.task_dim is not None:
        hyper_conf.d_task = args.task_dim
    if args.h_batch_size_rl is not None:
        hyper_conf.d_high_batch = args.h_batch_size_rl
    if args.h_batch_size_context is not None:
        hyper_conf.n_task_embedding_transition = args.h_batch_size_context
    if args.h_lr is not None:
        hyper_conf.high_q_lr = args.h_lr
        hyper_conf.high_pi_lr = args.h_lr
        hyper_conf.task_encoder_lr = args.h_lr
    if args.h_discount is not None:
        hyper_conf.high_gamma = args.h_discount
    
    if args.skill_buffer_size is not None:
        hyper_conf.skill_buffer_size = args.skill_buffer_size
    if args.skill_kld is not None:
        hyper_conf.skill_beta = args.skill_kld
    if args.skill_length is not None:
        hyper_conf.d_seq = args.skill_length
    if args.skill_dim is not None:
        hyper_conf.d_high_action = args.skill_dim
    if args.skill_lr is not None:
        hyper_conf.skill_lr = args.skill_lr
    if args.reward_lr is not None:
        hyper_conf.reward_lr = args.reward_lr
    if args.skill_n_priority is not None:
        hyper_conf.n_update_priority_episode = args.skill_n_priority
    if args.skill_k_iter is not None:
        hyper_conf.skill_sync_freq = args.skill_k_iter
    if args.skill_temp is not None:
        hyper_conf.skill_priority_temperature = args.skill_temp
    
    if args.exp_buffer_size is not None:
        hyper_conf.exploration_buffer_size = args.exp_buffer_size
    if args.exp_discount is not None:
        hyper_conf.exp_gamma = args.exp_discount
    if args.exp_lr is not None:
        hyper_conf.exp_q_lr = args.exp_lr
        hyper_conf.exp_pi_lr = args.exp_lr
    if args.exp_ent is not None:
        hyper_conf.exp_pi_reg_init = args.exp_ent
    if args.exp_kld is not None:
        hyper_conf.exp_bc_ratio = args.exp_kld
        
    if args.rnd_ext is not None:
        hyper_conf.rnd_ext_ratio = args.rnd_ext
    if args.rnd_int is not None:
        hyper_conf.rnd_int_ratio = args.rnd_int
    if args.rnd_dropout is not None:
        hyper_conf.rnd_state_dropout = args.rnd_dropout
    if args.rnd_dim is not None:
        hyper_conf.rnd_d_out = args.rnd_dim
    
    
    save_dir = f'./environments/{conf.ENV}/SISL'
    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------------------------
    # ----------- initialize environment -----------
    # ----------------------------------------------
    env_kwargs = {
        "env": env_id, "episode_max_step": conf.EPISODE_MAX_STEP, "truncated_done": False,
    }
    env = Environment(**env_kwargs)
    env_conf = env.get_conf()
    
    # ----------------------------------------------
    # ------------- initialize buffer --------------
    # ----------------------------------------------
    n_task = len(train_tasks)
    high_buffers = [
        FlatMaxTransitionEpisodeBuffer(
            max_transition=hyper_conf.high_buffer_size, 
            name=["state", "high_action", "high_reward", "next_state", "done"],
        ) for _ in range(n_task)
    ]
    encoder_buffers = [
        FlatMaxTransitionEpisodeBuffer(
            max_transition=hyper_conf.high_buffer_size, 
            name=["state", "high_action", "high_reward", "next_state", "done"],
        ) for _ in range(n_task)
    ]
    exploration_buffers = [
        FlatMaxTransitionEpisodeBuffer(
            max_transition=hyper_conf.exploration_buffer_size, 
            name=["state", "low_action", "low_reward", "next_state", "done", "score"],
        ) for _ in range(n_task)
    ]
    online_skill_buffers = [
        FlatMaxTransitionEpisodeBuffer(
            max_transition=hyper_conf.skill_buffer_size,  
            name=["state", "low_action", "low_reward", "next_state", "done", "score"],
            mode_input_priority=True,
        ) for _ in range(n_task)
    ]
    offline_skill_buffer      = FlatEpisodeBuffer(max_episode=999999, name=["state", "low_action"], mode_output_episode_priority=True)
    temp_offline_skill_buffer = FlatEpisodeBuffer(max_episode=999999, name=["state", "low_action"])

    # ----------------------------------------------
    # ---------------- define model ----------------
    # ----------------------------------------------
    
    # ----------- low-level(skill) model -----------
    # NOTE: skill encoder: q(Z | s_{i:i+H}, a_{i:i+H}), skill prior: p(Z | s_i), skill decoder(low policy): Pi(a | s, z)
    skill_model = Base.loads(
        path=conf.OFF_SKILL_PATH, 
        model={"encoder": SkillEncoder, "decoder": MLP, "prior": NormalMLP}
    )
    online_skill_encoder = skill_model["encoder"]
    online_skill_prior   = skill_model["prior"].eval()
    online_low_policy    = skill_model["decoder"].eval()
    target_skill_encoder = copy.deepcopy(online_skill_encoder).eval().requires_grad_(False)
    target_skill_prior = copy.deepcopy(online_skill_prior).eval().requires_grad_(False)
    target_low_policy = copy.deepcopy(online_low_policy).eval().requires_grad_(False)
    
    # ---------------- high-level -----------------
    # NOTE: high-level critic: Q_h(s, z, e)
    high_qs = [
        MLP(
            d_inputs=[env_conf.d_state, hyper_conf.d_high_action, hyper_conf.d_task], d_outputs=[1],
            d_hiddens=hyper_conf.high_q_d_hiddens
        ) for _ in range(hyper_conf.high_q_n)
    ]
    # NOTE: high-level policy: Pi_h(Z | s, e)
    high_policy = ResidualNormalMLP(
        d_state=env_conf.d_state, d_high_action=hyper_conf.d_high_action, d_task=hyper_conf.d_task,
        d_hiddens=hyper_conf.high_pi_d_hiddens, fixed_model=target_skill_prior,
        mode_limit_std="tanh"
    )
    # NOTE: task encoder q(E | C=(s,z,r,s',d))
    task_encoder = TaskEncoder(
        d_state=env_conf.d_state, d_action=hyper_conf.d_high_action, d_task=hyper_conf.d_task,
        d_hidden=hyper_conf.task_encoder_d_hidden, n_hidden=hyper_conf.task_encoder_n_hidden,
    )
    
    # ---------------- exploration ----------------
    d_conditions = [hyper_conf.d_score_onehot, n_task] if hyper_conf.use_exploration_score else [n_task]
    # NOTE: exploration critic: Q_exp(s, a, i)
    low_qs = [
        MLP(
            d_inputs=[env_conf.d_state, env_conf.d_action, *d_conditions], d_outputs=[1],
            d_hiddens=hyper_conf.exp_q_d_hiddens,
        ) for _ in range(hyper_conf.exp_q_n)
    ]
    # NOTE: exploration policy: Pi_exp(A | s, i)
    exploration_policy = TanhNormalMLP(
        d_inputs=[env_conf.d_state, *d_conditions], d_outputs=[env_conf.d_action],
        d_hiddens=hyper_conf.exp_pi_d_hiddens, sample_range=env_conf.action_range
    )
    # NOTE: reward model: r(s, a, i)
    reward_model = MLP(
        d_inputs=[env_conf.d_state, env_conf.d_action, n_task], d_outputs=[1], d_hiddens=hyper_conf.reward_d_hiddens,
    )
    if hyper_conf.use_rnd:
        # NOTE: RND: f(s', i)
        rnd = MLP(
            d_inputs=[env_conf.d_state, *d_conditions], d_outputs=[hyper_conf.rnd_d_out], d_hiddens=hyper_conf.rnd_d_hiddens,
        )
        # NOTE: target RND: f_hat(s', i)
        target_rnd = MLP(
            d_inputs=[env_conf.d_state, *d_conditions], d_outputs=[hyper_conf.rnd_d_out], d_hiddens=hyper_conf.target_rnd_d_hiddens,
        )
    else:
        rnd, target_rnd = None, None
        
    
    # ----------------------------------------------
    # --------------- define trainer ---------------
    # ----------------------------------------------
    high_model = SISL(
        high_qs=high_qs, high_q_lr=hyper_conf.high_q_lr, high_q_target_update_freq=hyper_conf.high_q_target_update_freq, high_q_target_update_ratio=hyper_conf.high_q_target_update_ratio,
        high_policy=high_policy, high_pi_lr=hyper_conf.high_pi_lr, task_encoder=task_encoder, task_encoder_lr=hyper_conf.task_encoder_lr,
        high_pi_reg_train=hyper_conf.high_pi_reg_train, high_pi_reg_init=hyper_conf.high_pi_reg_init,
        high_pi_reg_target=hyper_conf.high_pi_reg_target, high_pi_reg_lr=hyper_conf.high_pi_reg_lr,
        task_encoder_reg_train=hyper_conf.task_encoder_reg_train, task_encoder_reg_init=hyper_conf.task_encoder_reg_init,
        task_encoder_reg_target=hyper_conf.task_encoder_reg_target, task_encoder_reg_lr=hyper_conf.task_encoder_reg_lr,
        kl_clip=hyper_conf.high_pi_kl_clip, gamma=hyper_conf.high_gamma, n_train_task=n_task, skill_prior=target_skill_prior,
        d_high_action=hyper_conf.d_high_action,
        use_high_pi_reg_increase_only=hyper_conf.use_high_pi_reg_increase_only,
        high_pi_reg_max=hyper_conf.high_pi_reg_max, task_encoder_reg_max=hyper_conf.task_encoder_reg_max,
    )
    exp_model = AugmentedSAC(
        gamma=hyper_conf.exp_gamma,
        qs=low_qs, q_train_freq=1, q_lr=hyper_conf.exp_q_lr, target_q_update_ratio=hyper_conf.target_exp_q_update_ratio, target_q_update_freq=hyper_conf.target_exp_q_update_freq,
        policy=exploration_policy, pi_train_freq=1, pi_lr=hyper_conf.exp_pi_lr,
        dynamic_alpha=hyper_conf.exp_pi_reg_train, init_alpha=hyper_conf.exp_pi_reg_init, target_entropy=-env_conf.d_action * hyper_conf.target_exp_pi_reg_ratio,
        bc_ratio=hyper_conf.exp_bc_ratio,
    )
    optimizer_reward = optim.Adam(reward_model.parameters(), lr=hyper_conf.reward_lr)
    if hyper_conf.use_rnd:
        optimizer_rnd = optim.Adam(rnd.parameters(), lr=hyper_conf.rnd_lr)
        state_rms = [RunningMeanStd(shape=(env_conf.d_state, )) for _ in range(n_task)]

    skill_model = SpirlCL(
        encoder=online_skill_encoder, decoder=online_low_policy, prior=online_skill_prior, lr=hyper_conf.skill_lr, beta=hyper_conf.skill_beta
    )
    
    # --------- initilize collector ---------
    collector = HierarchicalTimeLimitCollector(
        d_high_action=hyper_conf.d_high_action, d_low_action=env_conf.d_action, d_seq=hyper_conf.d_seq,
        use_exploration_score=hyper_conf.use_exploration_score, d_score_onehot=hyper_conf.d_score_onehot,
    )
    conc_collector = ConcurrentCollector([GPUWorker(collector, gpu, **env_kwargs) for gpu in conf.DEVICE_SUB])
    
    # -------------------------------------------
    # -------------------------------------------
    # ------------------ main -------------------
    # -------------------------------------------
    # -------------------------------------------
    
    timer = TimeChecker()
    low2id = {"state": 0, "high_action": 1, "low_action": 2, "low_reward": 3, "next_state": 4, "done": 5}
    high2id = {"state": 0, "high_action": 1, "high_reward": 2, "next_state": 3, "done": 4}
    exp2id = {"state": 0, "low_action": 1, "low_reward": 2, "next_state": 3, "done": 4, "score": 5}
    temp_skill_buffer = [[] for _ in range(n_task)]
    gamma_sample = np.power(hyper_conf.skill_gamma, np.arange(conf.EPISODE_MAX_STEP))
    
    task_encoder.to(conf.DEVICE)
    high_model.to(conf.DEVICE)  # task_encoder, high_pi, high_qs, alpha, betas dependent
    exp_model.to(conf.DEVICE)   # exploration_pi, low_qs dependent, low_alpha dependent
    skill_model.to(conf.DEVICE) # online skill network dependent
    reward_model.to(conf.DEVICE)
    if hyper_conf.use_rnd:
        rnd.to(conf.DEVICE)
        target_rnd.to(conf.DEVICE)
        [rms_model.to(conf.DEVICE) for rms_model in state_rms]
    
        
    # -----------------------------------------------------------
    # --------------------- offline warmup ----------------------
    # -----------------------------------------------------------
    assert conf.get("OFF_DATASET_PATH", False), "OFF_DATASET_PATH is not defined"
    print(f'--- load offline dataset: {conf.OFF_DATASET_PATH} ---')
    offline_dataset = torch.load(conf.OFF_DATASET_PATH)
    episodes = dataset2episode(offline_dataset, labels=["observations", "actions"]) # use only state and action
    for episode in episodes:
        temp_offline_skill_buffer.set_sample(episode)
    
    # -----------------------------------------------------------
    # ---------------------- online warmup ----------------------
    # -----------------------------------------------------------
    high_model.to("cpu")
    high_model.high_policy._fixed_model[0].to("cpu")
    exp_model.to("cpu")
    task_encoder.to(conf.DEVICE)
    
    print("--- prior e~q(E) rollout ---")
    step = 0
    task_indices = list(range(n_task))
    while len(task_indices) > 0:
        step += 1
        prior_episodes = rollout_skill(
            hyper_conf=hyper_conf, mode_prior=True,
            task_indices=task_indices, tasks=train_tasks, conc_collector=conc_collector,
            encoder_buffers=encoder_buffers, 
            task_encoder=task_encoder, high_policy=high_policy, low_policy=target_low_policy, 
        )
        for task_idx, low_episode, high_episode in zip(task_indices, *prior_episodes):
            # NOTE: encoder_buffers
            encoder_buffers[task_idx].set_sample(high_episode)
            
            items = low_episode.export()
            score = items[low2id["low_reward"]].cumsum(axis=0) # (n_transition, 1)
            # NOTE: exploration_buffers
            new_items = [items[low2id["state"]], items[low2id["low_action"]], items[low2id["low_reward"]], items[low2id["next_state"]], items[low2id["done"]], score]
            new_exp_episode = Episode(*new_items)
            
            # NOTE: online_skill_buffers
            use_skill_buffer = True
            if hyper_conf.use_skill_buffer_nonzero_reward_sample_only:
                if score[-1].item() == 0:
                    use_skill_buffer = False
            if use_skill_buffer:
                if hyper_conf.use_skill_buffer_episode_clip:
                    reward_indices = np.where(items[low2id["low_reward"]] > 0)[0]
                    if len(reward_indices) > 0:
                        last_idx = reward_indices[-1] + 1
                        new_items = [item[:last_idx] for item in new_items]
                new_exp_episode = Episode(*new_items)
                if new_exp_episode.n_transition < hyper_conf.d_seq:
                    continue
                if hyper_conf.use_skill_buffer_input_priority:
                    full_return = score[-1].item()
                    gamma_return = (items[low2id["low_reward"]].squeeze() * gamma_sample[:len(items[0])]).sum()
                    priority = (full_return, gamma_return)
                else:
                    priority = None
                temp_skill_buffer[task_idx].append([new_exp_episode, priority])
        task_indices = [task_idx for task_idx, encoder_buffer in enumerate(encoder_buffers) if encoder_buffer.n_transition < hyper_conf.n_task_embedding_transition]
        print(f'encoder_buffers: {[buf.n_transition for buf in encoder_buffers]}')
        print(f'exploration_buffers: {[buf.n_transition for buf in exploration_buffers]}')
        # print(f'temp_skill_buffers: {[0 if len(buf)==0 else sum([item[0].n_transition for item in buf]) for buf in temp_skill_buffer]}')
    print(f'--- step: {step} done')
    
    print("--- posterior e~q(E|C) rollout ---")
    step = 0
    task_indices = list(range(n_task))
    while len(task_indices) > 0:
        step += 1
        posterior_episodes = rollout_skill(
            hyper_conf=hyper_conf, mode_prior=False,
            task_indices=task_indices, tasks=train_tasks, conc_collector=conc_collector,
            encoder_buffers=encoder_buffers, 
            task_encoder=task_encoder, high_policy=high_policy, low_policy=target_low_policy, 
        )
        for task_idx, low_episode, high_episode in zip(task_indices, *posterior_episodes):
            # NOTE: high_buffers
            high_buffers[task_idx].set_sample(high_episode)
            
            items = low_episode.export()
            score = items[low2id["low_reward"]].cumsum(axis=0) # (n_transition, 1)
            # NOTE: exploration_buffers
            new_items = [items[low2id["state"]], items[low2id["low_action"]], items[low2id["low_reward"]], items[low2id["next_state"]], items[low2id["done"]], score]
            new_exp_episode = Episode(*new_items)
            if not hyper_conf.use_warmup_exploration:
                exploration_buffers[task_idx].set_sample(new_exp_episode)
            
            # NOTE: online_skill_buffers
            use_skill_buffer = True
            if hyper_conf.use_skill_buffer_nonzero_reward_sample_only:
                if score[-1].item() == 0:
                    use_skill_buffer = False
            if use_skill_buffer:
                if hyper_conf.use_skill_buffer_episode_clip:
                    reward_indices = np.where(items[low2id["low_reward"]] > 0)[0]
                    if len(reward_indices) > 0:
                        last_idx = reward_indices[-1] + 1
                        new_items = [item[:last_idx] for item in new_items]
                new_exp_episode = Episode(*new_items)
                if new_exp_episode.n_transition < hyper_conf.d_seq:
                    continue
                if hyper_conf.use_skill_buffer_input_priority:
                    full_return = score[-1].item()
                    gamma_return = (items[low2id["low_reward"]].squeeze() * gamma_sample[:len(items[0])]).sum()
                    priority = (full_return, gamma_return)
                else:
                    priority = None
                temp_skill_buffer[task_idx].append([new_exp_episode, priority])
        task_indices = [task_idx for task_idx, high_buffer in enumerate(high_buffers) if high_buffer.n_transition < hyper_conf.d_high_batch]
        print(f'high_buffers: {[buf.n_transition for buf in high_buffers]}')
        print(f'exploration_buffers: {[buf.n_transition for buf in exploration_buffers]}')
        # print(f'temp_skill_buffers: {[0 if len(buf)==0 else sum([item[0].n_transition for item in buf]) for buf in temp_skill_buffer]}')
    print(f'--- step: {step} done')

    if hyper_conf.use_warmup_exploration:
        print("--- exploration Pi_exp(a | s, i) rollout ---")
        step = 0
        task_indices = list(range(n_task))
        while len(task_indices) > 0:
            step += 1
            exploration_episodes = rollout_exploration(
                hyper_conf=hyper_conf,
                task_indices=task_indices, tasks=train_tasks, conc_collector=conc_collector,
                exploration_policy=exploration_policy,
            )
            for task_idx, exp_episode in zip(task_indices, exploration_episodes):
                # NOTE: exploration_buffers
                exploration_buffers[task_idx].set_sample(exp_episode)
                
                # NOTE: online_skill_buffers
                items = exp_episode.export()
                use_skill_buffer = True
                if hyper_conf.use_skill_buffer_nonzero_reward_sample_only:
                    if items[exp2id["score"]][-1].item() == 0:
                        use_skill_buffer = False
                if use_skill_buffer:
                    new_items = items
                    if hyper_conf.use_skill_buffer_episode_clip:
                        reward_indices = np.where(items[exp2id["low_reward"]] > 0)[0]
                        if len(reward_indices) > 0:
                            last_idx = reward_indices[-1] + 1
                            new_items = [item[:last_idx] for item in new_items]
                    new_exp_episode = Episode(*new_items)
                    if new_exp_episode.n_transition < hyper_conf.d_seq:
                        continue
                    if hyper_conf.use_skill_buffer_input_priority:
                        full_return = items[exp2id["score"]][-1].item()
                        gamma_return = (items[exp2id["low_reward"]].squeeze() * gamma_sample[:len(items[0])]).sum()
                        priority = (full_return, gamma_return)
                    else:
                        priority = None
                    temp_skill_buffer[task_idx].append([new_exp_episode, priority])
            task_indices = [task_idx for task_idx, exploration_buffer in enumerate(exploration_buffers) if exploration_buffer.n_transition < hyper_conf.n_warmup_exp]
            print(f'exploration_buffers: {[buf.n_transition for buf in exploration_buffers]}')
            # print(f'temp_skill_buffers: {[0 if len(buf)==0 else sum([item[0].n_transition for item in buf]) for buf in temp_skill_buffer]}')
        print(f'--- step: {step} done')

    high_model.to(conf.DEVICE)
    high_model.high_policy._fixed_model[0].to(conf.DEVICE)
    exp_model.to(conf.DEVICE)

    # -----------------------------------------------------------
    # ------------------------ pre-train ------------------------
    # -----------------------------------------------------------
    # NOTE: reward_model pretrain
    if hyper_conf.n_reward_pretrain_step > 0:
        print(f'--- [{hyper_conf.n_reward_pretrain_step}] pretrain reward_model R(s, a, i) ---')
        timer.start_timer()
        for i in range(hyper_conf.n_reward_pretrain_step):
            task_indices = np.random.randint(0, n_task, size=hyper_conf.n_exp_sample_task)
            result_reward_model = update_reward_model(
                hyper_conf=hyper_conf, task_indices=task_indices, n_task=n_task,
                exploration_buffers=exploration_buffers, online_skill_buffers=online_skill_buffers,
                reward_model=reward_model, optimizer_reward=optimizer_reward,
            )
        timer.end_timer()
        # logger.log("reward_model", step=0)
        print(f'--- pretrain done (time: {timer.get_time():.3f}) ---')
        
    print("--- fill offline skill buffer ---")
    timer.start_timer()
    for i, ep in enumerate(temp_offline_skill_buffer.memory):
        output_priority = np.array(hyper_conf.return_max).repeat(n_task)
        offline_skill_buffer.set_sample(ep, output_episode_priority=[output_priority])
    timer.end_timer()
    print(f'--- offline skill buffer done (time: {timer.get_time():.3f})')
    
    print("--- fill online skill buffer ---")
    timer.start_timer()
    for task_idx, temp_buffer in enumerate(temp_skill_buffer):
        for i in range(len(temp_buffer)):
            ep, input_priority = temp_buffer[i]
            online_skill_buffers[task_idx].set_sample(ep, priority=[input_priority])
    timer.end_timer()
    print(f'--- online skill buffer done (time: {timer.get_time():.3f})')
    
    
    # -----------------------------------------------------------
    # -------------------------- train --------------------------
    # -----------------------------------------------------------
    use_bc = False
    for step in tqdm(range(1, conf.MAX_ITERATION+1)):
        # ----------------------- rollout -----------------------
        high_model.to("cpu")
        high_model.high_policy._fixed_model[0].to("cpu")
        exp_model.to("cpu")
        task_encoder.to(conf.DEVICE)
        
        temp_skill_buffer = [[] for _ in range(n_task)]
        task_indices = list(range(n_task))
        n_prior = 0
        n_post = 0
        n_exp = 0
        n_skill = 0
        
        # NOTE: prior e~q(E) rollout
        prior_episodes = rollout_skill(
            hyper_conf=hyper_conf, mode_prior=True,
            task_indices=task_indices, tasks=train_tasks, conc_collector=conc_collector,
            encoder_buffers=encoder_buffers, 
            task_encoder=task_encoder, high_policy=high_policy, low_policy=target_low_policy, 
        )
        for task_idx, low_episode, high_episode in zip(task_indices, *prior_episodes):
            n_prior += low_episode.n_transition
            # NOTE: encoder_buffers
            encoder_buffers[task_idx].set_sample(high_episode)
            # NOTE: high_buffers
            high_buffers[task_idx].set_sample(high_episode)
        
        # NOTE: posterior e~q(E|C) rollout
        posterior_episodes = rollout_skill(
            hyper_conf=hyper_conf, mode_prior=False,
            task_indices=task_indices, tasks=train_tasks, conc_collector=conc_collector,
            encoder_buffers=encoder_buffers, 
            task_encoder=task_encoder, high_policy=high_policy, low_policy=target_low_policy, 
        )
        for task_idx, low_episode, high_episode in zip(task_indices, *posterior_episodes):
            n_post += low_episode.n_transition
            # NOTE: high_buffers
            high_buffers[task_idx].set_sample(high_episode)
            
            items = low_episode.export()
            score = items[low2id["low_reward"]].cumsum(axis=0) # (n_transition, 1)
            # NOTE: exploration_buffers
            new_items = [items[low2id["state"]], items[low2id["low_action"]], items[low2id["low_reward"]], items[low2id["next_state"]], items[low2id["done"]], score]
            new_exp_episode = Episode(*new_items)
            if hyper_conf.use_posterior_sample_to_exploration:
                exploration_buffers[task_idx].set_sample(new_exp_episode)
            
            # NOTE: online_skill_buffers
            use_skill_buffer = True
            if hyper_conf.use_skill_buffer_nonzero_reward_sample_only:
                if score[-1].item() == 0:
                    use_skill_buffer = False
            if use_skill_buffer:
                if hyper_conf.use_skill_buffer_episode_clip:
                    reward_indices = np.where(items[low2id["low_reward"]] > 0)[0]
                    if len(reward_indices) > 0:
                        last_idx = reward_indices[-1] + 1
                        new_items = [item[:last_idx] for item in new_items]
                new_exp_episode = Episode(*new_items)
                if new_exp_episode.n_transition < hyper_conf.d_seq:
                    continue
                if hyper_conf.use_skill_buffer_input_priority:
                    full_return = score[-1].item()
                    gamma_return = (items[low2id["low_reward"]].squeeze() * gamma_sample[:len(items[0])]).sum()
                    priority = (full_return, gamma_return)
                else:
                    priority = None
                temp_skill_buffer[task_idx].append([new_exp_episode, priority])
        
        # NOTE: exploration rollout
        exploration_episodes = rollout_exploration(
            hyper_conf=hyper_conf,
            task_indices=task_indices, tasks=train_tasks, conc_collector=conc_collector,
            exploration_policy=exploration_policy,
        )
        for task_idx, exp_episode in zip(task_indices, exploration_episodes):
            n_exp += exp_episode.n_transition
            # NOTE: exploration_buffers
            exploration_buffers[task_idx].set_sample(exp_episode)

            # NOTE: online_skill_buffers
            items = exp_episode.export()
            use_skill_buffer = True
            if hyper_conf.use_skill_buffer_nonzero_reward_sample_only:
                if items[exp2id["score"]][-1].item() == 0:
                    use_skill_buffer = False
            if use_skill_buffer:
                new_items = items
                if hyper_conf.use_skill_buffer_episode_clip:
                    reward_indices = np.where(items[exp2id["low_reward"]] > 0)[0]
                    if len(reward_indices) > 0:
                        last_idx = reward_indices[-1] + 1
                        new_items = [item[:last_idx] for item in new_items]
                new_exp_episode = Episode(*new_items)
                if new_exp_episode.n_transition < hyper_conf.d_seq:
                    continue
                if hyper_conf.use_skill_buffer_input_priority:
                    full_return = items[exp2id["score"]][-1].item()
                    gamma_return = (items[exp2id["low_reward"]].squeeze() * gamma_sample[:len(items[0])]).sum()
                    priority = (full_return, gamma_return)
                else:
                    priority = None
                temp_skill_buffer[task_idx].append([new_exp_episode, priority])
        
        high_model.to(conf.DEVICE)
        high_model.high_policy._fixed_model[0].to(conf.DEVICE)
        exp_model.to(conf.DEVICE)
        # logger.log("episode", step=step)
        # logger.log("episode_exp", step=step)
        
        log_scores = [ep.export()[high2id["high_reward"]].sum() for ep in posterior_episodes[1]]
        # print()
        # print(f'[iteration={step}] Score={[ep.export()[high2id["high_reward"]].sum() for ep in posterior_episodes[1]]}')
        
        # -------------------- train --------------------
        n_high_step = max( int( ((n_prior + n_post)/hyper_conf.d_seq)//hyper_conf.n_high_sample_task )*hyper_conf.high_train_ratio, 1*hyper_conf.high_train_ratio )
        
        if hyper_conf.use_posterior_sample_to_exploration:
            n_exp_step  = max( int( ((n_exp + n_post)/hyper_conf.d_seq)//hyper_conf.n_exp_sample_task )*hyper_conf.exp_train_ratio, 1*hyper_conf.exp_train_ratio )
        else:
            n_exp_step  = max( int( ((n_exp)/hyper_conf.d_seq)//hyper_conf.n_exp_sample_task )*hyper_conf.exp_train_ratio, 1*hyper_conf.exp_train_ratio )

        # NOTE: train reward_model
        for i in range(1, n_exp_step+1):
            task_indices = np.random.randint(0, n_task, size=hyper_conf.n_exp_sample_task)
            result_reward_model = update_reward_model(
                hyper_conf=hyper_conf, task_indices=task_indices, n_task=n_task,
                exploration_buffers=exploration_buffers, online_skill_buffers=online_skill_buffers,
                reward_model=reward_model, optimizer_reward=optimizer_reward
            )
        # logger.log("reward_model", step=step)
        
        # NOTE: train exploration model
        if not use_bc:
            if all(buf.n_transition > hyper_conf.d_exp_batch for buf in online_skill_buffers):
                use_bc = True
        for i in range(1, n_exp_step+1):
            result_exp_model = update_exp_model(
                hyper_conf=hyper_conf,
                exploration_buffers=exploration_buffers, online_skill_buffers=online_skill_buffers,
                exp_model=exp_model,
                state_rms=state_rms, rnd=rnd, target_rnd=target_rnd, optimizer_rnd=optimizer_rnd,
                use_bc=use_bc,
            )
        # logger.log("exp_model", step=step)
        
        # NOTE: fill online_skill_buffers
        for task_idx, temp_buffer in enumerate(temp_skill_buffer):
            for i in range(len(temp_buffer)):
                ep, input_priority = temp_buffer[i]
                store_result = online_skill_buffers[task_idx].set_sample(ep, priority=[input_priority])
                if store_result:
                    n_skill += ep.n_transition
        
        # NOTE: skill_buffers logging
        full_returns = []
        gamma_returns = []
        is_empty = True
        for buf in online_skill_buffers:
            if buf.n_transition == 0:
                full_returns.append(None)
                gamma_returns.append(None)
                continue
            is_empty = False
            buf_full_returns = []
            buf_gamma_returns = []
            for ep in buf:
                items = ep.export()
                rewards, scores = items[exp2id["low_reward"]], items[exp2id["score"]]
                full_return = scores[-1].item()
                gamma_return = (rewards.squeeze() * gamma_sample[:len(rewards)]).sum()
                buf_full_returns.append(full_return)
                buf_gamma_returns.append(gamma_return)
            full_returns.append(sum(buf_full_returns) / len(buf_full_returns))
            gamma_returns.append(sum(buf_gamma_returns) / len(buf_gamma_returns))
        # lookup.full_returns = full_returns
        # lookup.gamma_returns = gamma_returns
        # lookup.full_returns_mean = np.array([r for r in full_returns if r is not None]).mean() if not is_empty else None
        # lookup.gamma_returns_mean = np.array([r for r in gamma_returns if r is not None]).mean() if not is_empty else None
        # logger.log("skill_buffers", step=step)

        
        limit = n_exp_step*2 if hyper_conf.use_posterior_sample_to_exploration else n_exp_step*4
        n_skill_step = min( int( ((n_skill)/hyper_conf.d_seq)*hyper_conf.skill_train_ratio ), limit )
        n_skill_step = max(n_skill_step, 1*hyper_conf.skill_train_ratio)

        # NOTE: --- train skill_model ---
        online_returns = []
        for buf in online_skill_buffers:
            for ep in buf:
                online_returns.append(ep[-1][-1].item())
        if len(online_returns) == 0:
            online_return_mean = 0
        else:
            online_return_mean = sum(online_returns) / len(online_returns)
        n_offline, n_online = 0, 0
        for i in range(1, n_skill_step+1):
            result_skill_model = update_skill_model(
                hyper_conf=hyper_conf, n_task=n_task,
                offline_skill_buffer=offline_skill_buffer, online_skill_buffers=online_skill_buffers,
                reward_model=reward_model, skill_model=skill_model,
                online_mean=online_return_mean, 
            )
            n_offline += result_skill_model.n_offline
            n_online += result_skill_model.n_online
        n_offline_ratio = n_offline / (n_offline + n_online)
        n_online_ratio = n_online / (n_offline + n_online)
        result_skill_model.update(dict(
            n_offline_ratio=n_offline_ratio, n_online_ratio=n_online_ratio,
        ))
        # logger.log("skill", step=step)
        log_beta = result_skill_model.get("n_online_ratio")
        # print()
        # print(f'[iteration={step}] beta={result_skill_model.get("n_online_ratio"):.3f}')
        
        # NOTE: update offline_skill_buffer priority
        for i in range(1, hyper_conf.n_update_priority_episode+1):
            result_offline_priority = update_buffer_priority(
                hyper_conf=hyper_conf, n_task=n_task,
                offline_skill_buffer=offline_skill_buffer, reward_model=reward_model,
            )
        
        # NOTE: train high_model
        for i in range(1, n_high_step+1):
            result_high_model = update_high_model(
                hyper_conf=hyper_conf,
                high_buffers=high_buffers, encoder_buffers=encoder_buffers,
                high_model=high_model, task_encoder=task_encoder,
            )
        # logger.log("high_model", step=step)
        
        if is_freq(step, conf.SAVE_FREQ) or step == conf.MAX_ITERATION:
            save_model = {
                # --- high-elvel ---
                "task_encoder": task_encoder,
                "high_policy" : high_policy,      "high_qs": high_qs,     
                "alpha"       : high_model.alpha, "betas"  : [b for b in high_model.betas],
                # --- low-level ---
                "exploration_policy": exploration_policy, "low_qs": low_qs,
                # --- skill ---
                "online_skill_encoder": online_skill_encoder, "online_low_policy": online_low_policy, "online_skill_prior": online_skill_prior,
                "reward_model": reward_model,
                "target_skill_encoder": target_skill_encoder, "target_low_policy": target_low_policy, "target_skill_prior": target_skill_prior,
            }
            if hyper_conf.use_rnd:
                save_model.update({
                    "rnd": rnd, "target_rnd": target_rnd,
                })
            save_path = f'{save_dir}/SISL-Env={conf.ENV}_Iter={step}.pt'
            Base.saves(
                path=save_path,
                model=save_model,
            )
            print(f'--- [iteration={step}] Save meta-train model to {save_path} ---')
            
        if is_freq(step, hyper_conf.skill_sync_freq):
            high_model.to("cpu")
            high_model.high_policy._fixed_model[0].to("cpu")
            exp_model.to("cpu")
            task_encoder.to(conf.DEVICE)
            
            # NOTE: skill sync
            skill_model.to("cpu")
            soft_update_param(target_low_policy, skill_model.decoder, tau=hyper_conf.skill_sync_ratio)
            soft_update_param(target_skill_prior, skill_model.prior, tau=hyper_conf.skill_sync_ratio)
            soft_update_param(target_skill_encoder, skill_model.encoder, tau=hyper_conf.skill_sync_ratio)
            high_model.skill_prior = target_skill_prior
            skill_model.to(conf.DEVICE)

            high_model.alpha = KLReg(high_model.high_pi_reg_init)
            high_model.optimizer_alpha = optim.Adam(high_model.alpha.parameters(), lr=high_model.high_pi_reg_lr)
            high_qs = [
                MLP(
                    d_inputs=[env_conf.d_state, hyper_conf.d_high_action, hyper_conf.d_task], d_outputs=[1],
                    d_hiddens=hyper_conf.high_q_d_hiddens
                ) for _ in range(hyper_conf.high_q_n)
            ]
            high_model.high_qs = nn.ModuleList(high_qs)
            high_model.target_high_qs = copy.deepcopy(high_model.high_qs).eval().requires_grad_(False)
            high_model.optimizer_q_high = optim.Adam(high_model.high_qs.parameters(), lr=high_model.high_q_lr)
            high_policy = ResidualNormalMLP(
                d_state=env_conf.d_state, d_high_action=hyper_conf.d_high_action, d_task=hyper_conf.d_task,
                d_hiddens=hyper_conf.high_pi_d_hiddens, fixed_model=target_skill_prior,
                mode_limit_std="tanh"
            )
            high_model.high_policy = high_policy
            high_model.optimizer_pi = optim.Adam(high_model.high_policy.parameters(), lr=high_model.high_pi_lr)

            # NOTE: posterior rollout
            pretrain_step = 0
            task_indices = list(range(n_task))
            while len(task_indices) > 0:
                posterior_episodes = rollout_skill(
                    hyper_conf=hyper_conf, mode_prior=False,
                    task_indices=task_indices, tasks=train_tasks, conc_collector=conc_collector,
                    encoder_buffers=encoder_buffers, 
                    task_encoder=task_encoder, high_policy=high_policy, low_policy=target_low_policy, 
                )
                for task_idx, low_episode, high_episode in zip(task_indices, *posterior_episodes):
                    pretrain_step += low_episode.n_transition
                    # NOTE: high_buffers
                    high_buffers[task_idx].set_sample(high_episode)
                task_indices = [task_idx for task_idx, high_buffer in enumerate(high_buffers) if high_buffer.n_transition < hyper_conf.d_high_batch]
            
            # NOTE: pretrain high_model
            high_model.to(conf.DEVICE)
            high_model.high_policy._fixed_model[0].to(conf.DEVICE)
            exp_model.to(conf.DEVICE)
            pretrain_step = max( int( ((pretrain_step)/hyper_conf.d_seq)//hyper_conf.n_high_sample_task )*hyper_conf.high_train_ratio, 1*hyper_conf.high_train_ratio )
            for i in range(1, pretrain_step+1):
                result_high_model = update_high_model(
                    hyper_conf=hyper_conf,
                    high_buffers=high_buffers, encoder_buffers=encoder_buffers,
                    high_model=high_model, task_encoder=task_encoder,
                )
            high_model.to("cpu")
            high_model.high_policy._fixed_model[0].to("cpu")
            exp_model.to("cpu")
            task_encoder.to(conf.DEVICE)
            
        print(f'\n[Iteration={step}] beta={log_beta:.3f}, scores={log_scores}')
            
    conc_collector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["kitchen", "office", "maze", "antmaze"], required=True)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--skill_path", required=True, type=str)
    
    parser.add_argument("--h_buffer_size", type=int)
    parser.add_argument("--h_kld", type=float)
    parser.add_argument("--task_dim", type=int)
    parser.add_argument("--h_batch_size_rl", type=int)
    parser.add_argument("--h_batch_size_context", type=int)
    parser.add_argument("--h_lr", type=float)
    parser.add_argument("--h_discount", type=float)
    
    parser.add_argument("--skill_buffer_size", type=int)
    parser.add_argument("--skill_kld", type=float)
    parser.add_argument("--skill_length", type=int)
    parser.add_argument("--skill_dim", type=int)
    parser.add_argument("--skill_lr", type=float)
    parser.add_argument("--reward_lr", type=float)
    parser.add_argument("--skill_n_priority", type=int)
    parser.add_argument("--skill_k_iter", type=int)
    parser.add_argument("--skill_temp", type=float)
    
    parser.add_argument("--exp_buffer_size", type=int)
    parser.add_argument("--exp_discount", type=float)
    parser.add_argument("--exp_lr", type=float)
    parser.add_argument("--exp_ent", type=float)
    parser.add_argument("--exp_kld", type=float)
    parser.add_argument("--rnd_ext", type=float)
    parser.add_argument("--rnd_int", type=float)
    parser.add_argument("--rnd_dropout", type=float)
    parser.add_argument("--rnd_dim", type=int)
    
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--device_sub", nargs="+", default="cpu", type=str)
    args = parser.parse_args()
    main(args)
    print("--- Done!!! ---")