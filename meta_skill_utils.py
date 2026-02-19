import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from utils.utils import to_onehot
from utils.base import Module, Environment

class HierarchicalTimeLimitCollector:
    def __init__(self, d_high_action, d_low_action, d_seq,
                 use_exploration_score:bool=False, d_score_onehot:int=None):
        self.d_high_action, self.d_low_action = d_high_action, d_low_action
        self.d_seq = d_seq

        self.use_exploration_score = use_exploration_score,
        self.d_score_onehot = d_score_onehot

    def collect_skill_episode(self, env, device, e, high_policy, low_policy, mode_deterministic=False):
        state, _info = env.reset()
        episode = []
        with Module.inference_modes(high_policy, low_policy):
            while True:
                low_step = env.n_step % self.d_seq # 0 ~ d_seq-1
                s = torch.from_numpy(state).float().to(device)
                if low_step == 0: # skill step
                    z = high_policy(s, e, deterministic=mode_deterministic) # (d_high_action)
                    z_numpy = z.cpu().numpy()
                a = low_policy(s, z).cpu().numpy() # (d_low_action)
                next_state, reward, done, truncated, _info = env.step(a)
                
                # NOTE: (s, z, a, r, s', d)
                episode.append([
                    state.astype(np.float32), 
                    z_numpy, 
                    a, 
                    np.array(reward).astype(np.float32),
                    next_state.astype(np.float32), 
                    done,
                ])
                state = next_state
                if done or truncated:
                    break
                
        return [np.vstack(item) for item in zip(*episode)]

    def collect_exploration_episode(self, env, device, e, exploration_policy):
        state, _info = env.reset()
        episode = []
        score = 0
        with Module.inference_mode(exploration_policy):
            while True:
                s = torch.from_numpy(state).float().to(device)
                conditions = []
                if self.use_exploration_score:
                    score_onehot = torch.from_numpy(to_onehot(score, self.d_score_onehot)).float().to(device)
                    conditions.append(score_onehot)
                conditions.append(e)
                action = exploration_policy(s, *conditions, deterministic=False).cpu().numpy()
                next_state, reward, done, truncated, _info = env.step(action)
                score += reward
                
                # NOTE: (s, a, r, s', d, score)
                transition  = [
                    state.astype(np.float32),
                    action,
                    np.array(reward).astype(np.float32),
                    next_state.astype(np.float32),
                    done,
                    np.array(score).astype(np.float32),
                ]
                episode.append(transition)
                state = next_state
                if done or truncated:
                    break
        
        return [np.vstack(item) for item in zip(*episode)]


class BaseWorker:
    def collect_episode(self):
        raise NotImplementedError
    
    def __call__(self, work_queue, received_queue, result_queue):
        while True:
            msg = work_queue.get()
            if msg is False:
                break
            received_queue.put(True)
            work_i, args, kwargs = msg
            mode = kwargs.pop("mode", False)
            if mode == "skill":
                # NOTE: high/low policy rollout
                result = self.collect_skill_episode(*args, **kwargs)
            elif mode == "exploration":
                # NOTE: exploration policy rollout
                result = self.collect_exploration_episode(*args, **kwargs)
            else: raise NotImplementedError
            msg = (work_i, result)
            result_queue.put(msg)

class GPUWorker(BaseWorker):
    def __init__(self, collector, gpu, **env_kwargs):
        self.collector = collector
        self.gpu = gpu
        self.env = None
        self.env_kwargs = env_kwargs
        
    def collect_skill_episode(self, task, e, high_policy, low_policy):
        if self.env is None:
            self.env = Environment(**self.env_kwargs)
        e = e.to(self.gpu) # (dTask)
        high_policy.to(self.gpu)
        low_policy.to(self.gpu)
        
        with self.env._env.set_task(task):
            episode = self.collector.collect_skill_episode(self.env, self.gpu, e, high_policy, low_policy)
        return episode
    
    def collect_exploration_episode(self, task, e, exploration_policy):
        if self.env is None:
            self.env = Environment(**self.env_kwargs)
        e = e.to(self.gpu)
        exploration_policy.to(self.gpu)
        
        with self.env._env.set_task(task):
            episode = self.collector.collect_exploration_episode(self.env, self.gpu, e, exploration_policy)
        return episode

class ConcurrentCollector:
    def __init__(self, workers):
        self.work_queue = mp.Queue(1) 
        self.received_queue = mp.Queue(1)
        self.result_queue = mp.Queue()
        
        self.processes = []
        for worker in workers:
            queues = (self.work_queue, self.received_queue, self.result_queue)
            p = mp.Process(target=worker, args=queues)
            p.daemon = True
            p.start()
            self.processes.append(p)
        self.work_i = 0
    
    def submit(self, *args, **kwargs):
        msg = (self.work_i, args, kwargs)
        self.work_queue.put(msg)
        self.received_queue.get()
        
        self.work_i += 1
    
    def wait(self):
        episodes = [None]*self.work_i
        while self.work_i > 0:
            msg = self.result_queue.get()
            work_i, episode = msg
            episode_clone = episode
            
            episodes[work_i] = episode_clone
            self.work_i -= 1
        return episodes
    
    def close(self):
        for _ in range(len(self.processes)):
            self.work_queue.put(False)
        for process in self.processes:
            process.join()
        

# settransformer
import math
import torch.distributions as torch_dist
import torch.nn as nn

def inverse_softplus(x):
    return float(np.log(np.exp(x) - 1))

def inverse_sigmoid(x):
    return float(-np.log(1/x - 1))


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class MLP(nn.Module):
    activation_classes = {
        'relu': nn.ReLU,
    }
    def __init__(self, dims, activation='relu'):
        super().__init__()
        layers = []
        prev_dim = dims[0]
        for dim in dims[1:-1]:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(self.activation_classes[activation]())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SetTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, n_attention, n_mlp_layer,
                 n_ind=32, n_head=4, ln=False, activation='relu'):
        super().__init__()

        attention_layers =  [ISAB(in_dim, hidden_dim, n_head, n_ind, ln=ln)]
        attention_layers += [
            ISAB(hidden_dim, hidden_dim, n_head, n_ind, ln=ln)
            for _ in range(n_attention-1)
        ]
        self.attention = nn.Sequential(*attention_layers)
        self.pool = PMA(hidden_dim, n_head, 1, ln=ln)
        self.mlp = MLP([hidden_dim]*n_mlp_layer + [out_dim], activation=activation)

    def forward(self, batch_set_x):
        return self.mlp(self.pool(self.attention(batch_set_x)).squeeze(1))
    


class StochasticEncoder(torch.nn.Module):
    def __init__(self, z_dim, prior_scale):
        super().__init__()
        self.register_buffer('prior_loc', torch.zeros(z_dim))
        self.register_buffer('prior_scale', prior_scale*torch.ones(z_dim))

        self.device = None

    def to(self, device):
        self.device = device
        return super().to(device)
     
    @property
    def prior_dist(self):
        return torch_dist.Independent(torch_dist.Normal(self.prior_loc, self.prior_scale), 1)
    
    def dist(self, batch_transitions):
        raise NotImplementedError

    def encode(self, list_batch, sample):
        if len(list_batch) == 0:
            dist = self.prior_dist

            if sample is True:
                z = dist.sample()
            else:
                z = dist.mean
        else:
            transitions = torch.cat([batch.as_transitions() for batch in list_batch], dim=0)
            with torch.no_grad():
                dist = self.dist(transitions.to(self.device).unsqueeze(0))
        
            if sample is True:
                z = dist.sample().squeeze(0)
            else:
                z = dist.mean.squeeze(0)
        return z.cpu().numpy()

class SetTransformerEncoder(StochasticEncoder):
    def __init__(self, state_dim, action_dim, z_dim,
                 hidden_dim, n_hidden, activation='relu',
                 min_scale=0.001, max_scale=None, init_scale=1, prior_scale=1):
        super().__init__(z_dim, prior_scale)

        self.net = SetTransformer(
            2*state_dim + action_dim + 2, 2*z_dim,
            hidden_dim, n_hidden, n_hidden, activation=activation
        )
        self.min_scale = min_scale
        self.max_scale = max_scale

        if max_scale is None:
            self.pre_init_scale = inverse_softplus(init_scale)
        else:
            self.pre_init_scale = inverse_sigmoid(max_scale/init_scale - 1)

    def dist(self, batch_transitions):
        loc, pre_scale = self.net(batch_transitions).chunk(2, dim=-1)
        if self.max_scale is None:
            scale = self.min_scale + F.softplus(self.pre_init_scale + pre_scale)
        else:
            scale = self.min_scale + self.max_scale*torch.sigmoid(self.pre_init_scale + pre_scale)
        return torch_dist.Independent(torch_dist.Normal(loc, scale), 1)
