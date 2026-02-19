import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import copy
from typing import List, Optional
from tqdm import tqdm

from utils.base import Environment, MainModule, Module, Base
from utils.utils import soft_update_param, is_freq, AttrDict
from utils.buffer import Episode, Batch, FlatMaxTransitionEpisodeBuffer
from utils.distribution import TanhNormal, kl_divergence
from utils.net import MLP, NormalMLP

from meta_skill_utils import HierarchicalTimeLimitCollector
from Prism import TaskEncoder, ResidualNormalMLP, KLReg


class ConstrainedSAC(MainModule):
    @Base.save_input(exclude=["qs", "policy"])
    def __init__(self, gamma:float, kl_clip:int,
                 qs:List[Module], q_lr:float, target_q_update_ratio:float, target_q_update_freq:int,
                 policy:Module, pi_lr:float,
                 dynamic_alpha:bool, init_alpha:float=1.0, target_divergence:Optional[float]=None,
                 ):
        super().__init__()
        # --- Q ---
        self.q_lr = q_lr
        self.target_q_update_ratio = target_q_update_ratio
        self.target_q_update_freq = target_q_update_freq
        self.qs = nn.ModuleList(qs)
        self.target_qs = copy.deepcopy(self.qs).eval().requires_grad_(False)
        self.optimizer_q = optim.Adam(self.qs.parameters(), lr=self.q_lr)
        
        # --- Pi ---
        self.pi_lr = pi_lr
        self.Pi = policy
        self.optimizer_pi = optim.Adam(self.Pi.parameters(), lr=self.pi_lr)
        
        # --- alpha(temperature) ---
        self.dynamic_alpha = dynamic_alpha
        if self.dynamic_alpha:
            self.target_divergence = target_divergence
            self.log_alpha = nn.Parameter(torch.tensor(init_alpha).float().log())
            self.alpha = self.log_alpha.detach().exp().item()
            self.optimizer_alpha = optim.Adam([self.log_alpha], lr=self.q_lr)
        else:
            self.alpha = init_alpha
        
        # --- etc. ---
        self.gamma = gamma
        self.kl_clip = kl_clip
        self._step = 0
    
    def loss_critic(self, sample:List[torch.Tensor], e, skill_prior) -> dict:
        state, action, reward, next_state, done = sample
        # --- estimate ---
        estimate_qs = [
            q(state, action, e) for q in self.qs
        ]
        
        # --- target ---
        with torch.no_grad():
            next_pi_action, next_pi_dist = self.Pi.sample(next_state, e, deterministic=False, with_dist=True)
            next_prior_dist = skill_prior.dist(next_state).detach()
            if isinstance(next_pi_dist, TanhNormal):
                raise NotImplementedError
            else:
                next_kl_pi_prior = kl_divergence(next_pi_dist, next_prior_dist, mode_multivariate=True)
            scales = next_kl_pi_prior.detach().clamp(0, self.kl_clip) / next_kl_pi_prior.detach()
            next_kl_pi_prior = next_kl_pi_prior*scales

            next_target_qs = [
                target_q(next_state, next_pi_action, e) for target_q in self.target_qs
            ]
            next_target_min_q = torch.cat(next_target_qs, dim=-1).min(dim=-1, keepdim=True).values
            next_target_v = next_target_min_q - self.alpha * next_kl_pi_prior
            target_q = reward + done.logical_not() * self.gamma * next_target_v
        
        loss_qs = [
            F.mse_loss(estimate_q, target_q) for estimate_q in estimate_qs
        ]
        loss_q = torch.stack(loss_qs).mean()
        return AttrDict(
            loss_q=loss_q,
            **{f'loss_q{i+1}': loss_qs[i].detach().cpu().item() for i in range(len(loss_qs))},
            **{f'q{i+1}': estimate_qs[i].detach().mean().cpu().item() for i in range(len(estimate_qs))},
        )
        
    def loss_actor(self, sample:List[torch.Tensor], e, skill_prior:Module):
        state, action, reward, next_state, done = sample
        
        pi_action, pi_dist = self.Pi.sample(state, e, deterministic=False, with_dist=True)
        prior_dist = skill_prior.dist(state).detach()
        kl_pi_prior = kl_divergence(pi_dist, prior_dist, mode_multivariate=True)
        
        with Module.freeze_modes(*self.qs):
            estimate_qs = [
                q(state, pi_action, e) for q in self.qs
            ]
        estimate_min_q = torch.cat(estimate_qs, dim=-1).min(dim=-1, keepdim=True).values
        loss_pi = (self.alpha * kl_pi_prior) - estimate_min_q
        loss_pi = loss_pi.mean()
        return AttrDict(
            loss_pi=loss_pi,
            kl_pi_prior=kl_pi_prior,
        )
        
    def update_target(self) -> None:
        if is_freq(self._step, self.target_q_update_freq):
            soft_update_param(self.target_qs, self.qs, tau=self.target_q_update_ratio)
        
    def update(self, sample:List[torch.Tensor], e, skill_prior:Module):
        self._step += 1
        self.update_target()
        result = AttrDict()
        
        result_critic = self.loss_critic(sample, e, skill_prior)
        self.optimizer_q.zero_grad()
        result_critic.loss_q.backward()
        self.optimizer_q.step()
        result_critic.loss_q = result_critic.loss_q.detach().cpu().item()
        result.update(result_critic)
        
        result_actor = self.loss_actor(sample, e, skill_prior)
        self.optimizer_pi.zero_grad()
        result_actor.loss_pi.backward()
        self.optimizer_pi.step()
        
        if self.dynamic_alpha:
            kl_pi_prior = result_actor.kl_pi_prior
            alpha = self.log_alpha.exp()
            loss_alpha = (alpha * (self.target_divergence - kl_pi_prior.detach())).mean()
            
            self.optimizer_alpha.zero_grad()
            loss_alpha = loss_alpha.clamp(min=-np.inf, max=0)
            loss_alpha.backward()
            self.optimizer_alpha.step()
            self.alpha = self.log_alpha.detach().exp().item()
            result.update(loss_alpha=loss_alpha.detach().cpu().item())
            
        result_actor.loss_pi = result_actor.loss_pi.detach().cpu().item()
        result_actor.kl_pi_prior = result_actor.kl_pi_prior.detach().mean().cpu().item()
        result.update(result_actor)
        
        return result

def rollout(hyper_conf:AttrDict, collector, env, e:torch.Tensor, high_policy:Module, low_policy:Module, mode_deterministic=False):
    d_seq = hyper_conf.d_seq
    device = e.device
    
    raw_episode = collector.collect_skill_episode(env=env, device=device, e=e, high_policy=high_policy, low_policy=low_policy, mode_deterministic=mode_deterministic)
    # low-level transition: (state, high_action, low_action, reward, next_state, done)
    
    # convert low/high-level episode
    low_episode = Episode(*raw_episode)
    horizon = low_episode.n_transition
    start_indices = np.arange(0, horizon, d_seq)
    end_indices = start_indices + d_seq -1
    end_indices[-1] = min(end_indices[-1], horizon - 1)
    
    high_episode = Episode(
        raw_episode[0][start_indices],
        raw_episode[1][start_indices],
        np.pad(raw_episode[3].flatten(), (0, (d_seq - (horizon%d_seq))%d_seq)).reshape(-1, d_seq).sum(axis=-1, keepdims=True),
        raw_episode[4][end_indices],
        raw_episode[5][end_indices],
    )
    return low_episode, high_episode


def main(args):
    conf = AttrDict(
        ENV                 = args.env,
        DEVICE              = int(args.device) if args.device.isnumeric() else args.device,
        HIGH_MODEL_PATH     = args.model_path,
        MAX_ITERATION       = args.iteration,
    )
    hyper_conf = AttrDict(
        buffer_size         = None,
        gamma               = 0.99,
        d_high_action       = 10,
        d_seq               = 10,
        n_prior_episode     = None, # prior warmup시 rollout episode
        d_batch             = 256,
        # --- critic ---
        kl_clip             = None,
        q_lr                = 0.0003,
        target_q_update_ratio = 0.005,
        target_q_update_freq  = 1,
        # --- actor ---
        pi_lr               = 0.0003,
        # --- alpha ---
        pi_reg_train   = True,
        pi_reg_target  = None, # target kl-divergence
    )
    if conf.ENV == "kitchen":
        conf.update(
            EPISODE_MAX_STEP    = 280,
        )
        hyper_conf.update(
            buffer_size         = 20000,
            n_prior_episode     = 20,
            pi_reg_target       = 5,
            kl_clip             = 20,
        )
        from utils_kitchen import tasks
        tasks = tasks.test_tasks
        env_id = "simpl-kitchen-v0"
    elif conf.ENV == "office":
        conf.update(
            EPISODE_MAX_STEP    = 300,
        )
        hyper_conf.update(
            buffer_size         = 20000,
            n_prior_episode     = 20,
            pi_reg_target       = 5,
            kl_clip             = 20,
        )
        from utils_office import tasks
        tasks = tasks.test
        env_id = "office-v0"
    elif conf.ENV == "maze":
        conf.update(
            EPISODE_MAX_STEP= 2000,
        )
        hyper_conf.update(
            buffer_size     = 20000,
            n_prior_episode = 20,
            pi_reg_target   = 4,
            kl_clip         = 5,
        )
        from utils_maze import tasks
        tasks = tasks.test
        env_id = "simpl-maze-size20-seed0-v0"
    elif conf.ENV == "antmaze":
        conf.update(
            EPISODE_MAX_STEP= 1000,
        )
        hyper_conf.update(
            buffer_size     = 20000,
            n_prior_episode = 20,
            pi_reg_target   = 3,
            kl_clip         = 5,
        )
        from utils_antmaze import size10_tasks as tasks
        tasks = tasks.test
        env_id = "antmaze-size10-v0"
    else: raise NotImplementedError

    # ----------------------------------------------
    # ----------------------------------------------
    # main
    # ----------------------------------------------
    # ----------------------------------------------
        
    # --- initialize environment ---
    env_kwargs = {
        "env": env_id, "episode_max_step": conf.EPISODE_MAX_STEP, "truncated_done": False,
    }
    env = Environment(**env_kwargs)
    env_test = Environment(**env_kwargs)
    env_conf = env.get_conf()
    
    name2id = {"state":0, "high_action":1, "high_reward":2, "next_state":3, "done":4}
    
    # --- initialize model  ---
    dummy_skill_prior = NormalMLP([1], [1], [1])
    model = Base.loads(
        path=conf.HIGH_MODEL_PATH,
        model={
            # --- low
            "target_low_policy": MLP, "target_skill_prior": NormalMLP,
            # "online_low_policy": MLP, "online_skill_prior": NormalMLP,
            # --- high
            "task_encoder": TaskEncoder, "high_qs": MLP,
            "high_policy": {"model": ResidualNormalMLP, "fixed_model": dummy_skill_prior},
            "alpha": KLReg,
        }
    )
    low_policy  = model["target_low_policy"].eval().requires_grad_(False)
    skill_prior = model["target_skill_prior"].eval().requires_grad_(False)
    fixed_model = copy.deepcopy(skill_prior).eval().requires_grad_(False)

    task_encoder = model["task_encoder"] # q(e | c)
    base_high_policy, base_high_qs = model["high_policy"], model["high_qs"] # Pi(z | s, e), Q(s, z, e)
    base_high_policy._fixed_model = (fixed_model, )
    base_alpha = model["alpha"] # alpha
    
    # --- initialize collector ---
    collector = HierarchicalTimeLimitCollector(
        d_high_action=hyper_conf.d_high_action, d_low_action=env_conf.d_action, d_seq=hyper_conf.d_seq
    )
    
    # -------------------------------------------
    # --- main
    # -------------------------------------------
    
    task_encoder.to(conf.DEVICE)
    skill_prior.to(conf.DEVICE)
    scores = {}
    
    for task_idx in range(len(tasks)):
        print(f'--- task: {task_idx} ---')
        # --- config setup ---
        new_conf, new_hyper_conf = conf.copy(), hyper_conf.copy()
        new_conf.update(TASK=task_idx)
        
        # --- network ready ---
        high_policy = copy.deepcopy(base_high_policy)
        high_qs = copy.deepcopy(base_high_qs)
        model = ConstrainedSAC(
            gamma=new_hyper_conf.gamma, kl_clip=new_hyper_conf.kl_clip,
            qs=high_qs, q_lr=new_hyper_conf.q_lr, 
            target_q_update_ratio=new_hyper_conf.target_q_update_ratio, target_q_update_freq=new_hyper_conf.target_q_update_freq,
            policy=high_policy, pi_lr=new_hyper_conf.pi_lr,
            dynamic_alpha=new_hyper_conf.pi_reg_train, 
            init_alpha=base_alpha().item(), 
            # init_alpha=base_alpha,
            target_divergence=new_hyper_conf.pi_reg_target,
        ).to(new_conf.DEVICE)
        buffer = FlatMaxTransitionEpisodeBuffer(
            max_transition=new_hyper_conf.buffer_size, name=["state", "high_action", "high_reward", "next_state", "done"]
        )
        
        episodes = []
        with env._env.set_task(tasks[task_idx]):
            for step in range(new_hyper_conf.n_prior_episode):
                e = task_encoder.prior.sample()
                low_episode, high_episode = rollout(
                    hyper_conf=new_hyper_conf, collector=collector, env=env, e=e, high_policy=high_policy, low_policy=low_policy,
                    mode_deterministic=False,
                )
                episodes.append(high_episode)
                # logger.log("episode", step=step)
        batches = [ep.as_batch() for ep in episodes]
        batches = Batch.merge(*batches).torch().to(new_conf.DEVICE).cat()
        with task_encoder.inference_mode():
            e = task_encoder(batches, deterministic=True)
        es = e.unsqueeze(dim=0).repeat(new_hyper_conf.d_batch, 1) # (d_batch, d_task)
        
        # --- train loop ---
        for step in tqdm(range(step + 1, new_conf.MAX_ITERATION + 1)):
            # --- rollout
            with env._env.set_task(tasks[task_idx]):
                low_episode, high_episode = rollout(
                    hyper_conf=new_hyper_conf, collector=collector, env=env, e=e, high_policy=high_policy, low_policy=low_policy,
                    mode_deterministic=False,
                )
            buffer.set_sample(high_episode)
            
            # --- train
            n_step = max( high_episode.n_transition, 1 )
            for _ in range(n_step):
                sample = buffer.get_sample(new_hyper_conf.d_batch).torch().to(new_conf.DEVICE)
                result = model.update(sample, es, skill_prior)
        
            # --- test ---
            with env_test._env.set_task(tasks[task_idx]):
                low_episode, high_episode = rollout(
                    hyper_conf=new_hyper_conf, collector=collector, env=env_test, e=e, high_policy=high_policy, low_policy=low_policy,
                    mode_deterministic=True,
                )
            # logger.log("test", step=step)
            score = high_episode.export()[name2id["high_reward"]].sum()
            print(f'\n[Task={task_idx}, Iteration={step}] score={score}')
        scores[f'task{str(task_idx).zfill(2)}'] = score
    
    mean_socre = np.array(list(scores.values())).mean()
    print(f'Total scores: {scores}')
    print(f'Mean scores: {mean_socre:.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["kitchen", "office", "maze", "antmaze"], required=True)
    parser.add_argument("--iteration", default=500, required=True, type=int)
    parser.add_argument("--model_path", required=True, type=str)
    
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()
    main(args)
    print("--- Done!!! ---")