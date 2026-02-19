import warnings # ignore warning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import torch
import gym
from typing import List, Tuple, Optional, Union

NUMPY = np.ndarray
TORCH = torch.Tensor

CLASSIC = "CLASSIC"
BOX2D = "BOX2D"
D4RL = "D4RL"
MUJOCO = "MUJOCO"
ATARI = "ATARI"
UNKNOWN_ENV = "UNKNOWN_ENV"


class AttrDict(dict):
    """To store hyperparameters"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
    
    def update(self, *args, **kwargs):
        if args:
            super().update(*args)
        else:
            super().update(dict(**kwargs))
    
    def copy(self):
        return type(self)(**super().copy())

def is_set(condition):
    """ 
    condition == None, False, -1 -> False
    """
    if condition is None:
        return False
    elif isinstance(condition, bool):
        return condition
    else:
        return condition != -1
    
def is_low(step, condition):     return is_set(condition) and (step < condition)
def is_low_eq(step, condition):  return is_set(condition) and (step <= condition)
def is_high(step, condition):    return is_set(condition) and (step > condition)
def is_high_eq(step, condition): return is_set(condition) and (step >= condition)
def is_freq(step, condition):    return is_set(condition) and (step % condition == 0)

def to_torch(x:Union[int, float, List, NUMPY, TORCH], dtype:torch.dtype=torch.float32):
    if not isinstance(x, (TORCH, NUMPY)):
        x = np.array(x)
    if isinstance(x, NUMPY):
        x = torch.from_numpy(x)
    if x.dtype not in (torch.bool, torch.uint8, dtype):
        x = x.type(dtype)
    if x.ndim == 0:
        x = x.unsqueeze(dim=0) # e.g. 3 -> [3]
    return x

def to_numpy(x:Union[int, float, List, NUMPY, TORCH], dtype=np.float32):
    if not isinstance(x, (TORCH, NUMPY)):
        x = np.array(x)
    if isinstance(x, TORCH):
        x = x.cpu().detach().numpy()
    if x.dtype not in (bool, np.uint8, dtype):
        x = x.astype(dtype)
    if x.ndim == 0:
        x = np.expand_dims(x, axis=-1)
    return x

def to_list(x:Union[int, float, List, NUMPY, TORCH, Tuple]):
    if isinstance(x, (NUMPY, TORCH)) and x.ndim == 1:
        x = x.squeeze()
    if isinstance(x, TORCH):
        x = x.detach().cpu().tolist()
    elif isinstance(x, NUMPY):
        x = x.tolist()
    if isinstance(x, Tuple):
        x = list(x)
    if not isinstance(x, List):
        x = [x]
    return x

def numpy_split(x:NUMPY, split_shapes:List[int], dim:int=0):
    if not isinstance(split_shapes, NUMPY):
        split_shapes = np.array(split_shapes)
    assert x.shape[dim] == split_shapes.sum()
    split_shapes = split_shapes.cumsum()[:-1]
    return np.split(x, split_shapes, axis=dim)


def get_env_type(env):
    name = env if isinstance(env, str) else env.spec.id
    
    if any([env_name in name for env_name in ["CartPole", "Acrobot", "MountainCar", "Pendulum"]]):
        return CLASSIC
    elif any([env_name in name for env_name in ["LunarLander", "BipedalWalker", "CarRacing"]]):
        return BOX2D
    elif any([env_name in name for env_name in ["maze2d", "antmaze",  "minigrid",     "pen",       "hammer", 
                                              "door",   "relocate", "halfcheetah-", "walker2d-", "hopper-",
                                              "ant-",   "flow-",    "carla-",
                                              "kitchen-complete-v0", "kitchen-partial-v0", "kitchen-mixed-v0"]]):
        return D4RL
    elif any([env_name in name for env_name in ["Ant",            "HalfCheetah", "Hopper",  "Humanoid", "InvertedDoublePendulum", 
                                              "InvertedPendulum", "Reacher",     "Swimmer", "Walker2d"]]):
        return MUJOCO
    elif any([env_name in name for env_name in ["Adventure",    "AirRaid",      "Alien",        "Amidar",           "Assault", 
                                              "Asterix",      "Asteriods",    "Atlantis",     "BankHeist",        "Bowling",
                                              "Boxing",       "Breakout",     "Carnival",     "Centipede",        "ChopperCommand",
                                              "CrazyClimber", "Defender",     "DemonAttack",  "DoubleDunk",       "ElevatorAction",
                                              "Enduro",       "FishingDerby", "Freeway",      "Frostbite",        "Gopher",
                                              "Gravitar",     "Hero",         "IceHockey",    "Jamesbond",        "JourneyEscape",
                                              "Kangaroo",     "Krull",        "KungFuMaster", "MontezumaRevenge", "MsPacman",
                                              "NameThisGame", "Phoenix",      "Pitfall",      "Pong",             "Pooyan",
                                              "PrivateEye",   "Qbert",        "Riverraid",    "RoadRunner",       "Robotank",
                                              "Seaquest",     "Skiing",       "Solaris",      "SpaceInvaders",    "StarGunner",
                                              "Tennis",       "TimePilot",    "Tutankham",    "UpNDown",          "Venture",
                                              "VideoPinball", "WizardOfWar",  "YarsRevenge",  "Zaxxon"]]):
        return ATARI
    else:
        return UNKNOWN_ENV

def get_env_conf(env) -> AttrDict:
    if isinstance(env.observation_space, gym.spaces.Box): # continuous state space
        observation_space = dict(high=env.observation_space.high, low=env.observation_space.low)
        d_state = env.observation_space.shape[0]
    else: # discrete state space
        observation_space = None
        d_state = env.observation_space.n
        
    if isinstance(env.action_space, gym.spaces.Box): # continuous action space
        action_space = dict(high=env.action_space.high, low=env.action_space.low)
        d_action = env.action_space.shape[0] if env.action_space.shape else env.action_space.n
        action_range = (env.action_space.low[0], env.action_space.high[0])
        assert (env.action_space.low == action_range[0]).all() and (env.action_space.high == action_range[1]).all()
    else: # discrete action space
        action_space = None
        d_action = env.action_space.n
        action_range = None
    return AttrDict(d_state=d_state, d_action=d_action, observation_space=observation_space, action_space=action_space, action_range=action_range)

def make_env(name:str, mode_render:bool=False, mode_test:bool=False, time_limit:bool=True, **kwargs) -> gym.Env:
    """create environment"""
    env_type = get_env_type(name)
    if mode_render:
        kwargs.update({"render_mode":"rgb_array"})
        
    if env_type in [CLASSIC, BOX2D, MUJOCO]:
        env = gym.make(name, **kwargs)
    elif env_type == D4RL:
        # NOTE: remove import error print
        import sys, os
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = open(os.devnull, "w"), open(os.devnull, "w")
        try:
            import d4rl
            env = gym.make(name, **kwargs)
        except:
            sys.stdout, sys.stderr = stdout, stderr
            print("D4RL import error")
        finally:
            sys.stdout, sys.stderr = stdout, stderr
    elif env_type == ATARI:
        # from .utils.atari_warpper import AtariWrapper
        from .atari_wrapper import AtariWrapper
        env = gym.make(name, **kwargs)
        if isinstance(env, gym.wrappers.TimeLimit) and not time_limit:
            env = env.env # remove TimeLimit wrapper
            
        if mode_test:
            env = AtariWrapper(env, clip_reward=False)
        else:
            env = AtariWrapper(env)
    else:
        # raise NotImplementedError("Unknown Environment")
        env = gym.make(name, **kwargs)
    
    if isinstance(env, gym.wrappers.TimeLimit) and not time_limit:
        env = env.env
    
    return env

def make_input(x, dim:int, device=None) -> torch.Tensor: # ([dBatch], -)
    assert x.ndim >= dim - 1
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.ndim == dim-1:
        x = x.unsqueeze(dim=0) # (1, -)
    if (device is not None) and (x.device != torch.device(device)):
        x = x.to(device)
    return x # (dBatch, -)

def make_output(x:torch.Tensor, device=None) -> torch.Tensor: # (dBatch, -)
    if x.shape[0] == 1:
        x = x.squeeze(dim=0) # (-)
    if (device is not None) and (x.device != torch.device(device)):
        x = x.to(device)
    return x # ([dBatch], -)


def soft_update_param(net_a:torch.nn.Module, net_b:torch.nn.Module, tau:float):
    for paramA, paramB in zip(net_a.parameters(), net_b.parameters()):
        paramA.data = tau*paramB.detach().clone() + (1-tau)*paramA.data
        
def hard_update_param(net_a:torch.nn.Module, net_b:torch.nn.Module, strict:bool=True):
    net_a.load_state_dict(net_b.state_dict(), strict=strict)

def create_mlp(d_in:int, d_out:int, d_hiddens:List[int], leakyrelu:bool=False, leakyrelu_slope:Optional[float]=None, batchnorm:bool=False):
    """ Create MLP sequential model 
        e.g.) d_in=10, d_out=20, d_hiddens=[256, 128]
            >> 10 -> 256 -> 128 -> 20
    """
    if leakyrelu_slope is None:
        leakyrelu_slope = 0.01 # NOTE: torch default leakyrelu slope
    
    activation = torch.nn.LeakyReLU(leakyrelu_slope) if leakyrelu else torch.nn.ReLU()
    layers = [torch.nn.Linear(d_in, d_hiddens[0])]
    layers.append(activation)
    for i, _ in enumerate(d_hiddens[:-1]):
        if batchnorm:
            layers.append(torch.nn.Linear(d_hiddens[i], d_hiddens[i+1], bias=False))
            layers.append(torch.nn.BatchNorm1d(d_hiddens[i+1]))
            layers.append(activation)
        else:
            layers.append(torch.nn.Linear(d_hiddens[i], d_hiddens[i+1]))
            layers.append(activation)
    layers.append(torch.nn.Linear(d_hiddens[-1], d_out))
    return torch.nn.Sequential(*layers)

def to_onehot(x, num_classes:int): # (..., [1])
    if isinstance(x, torch.Tensor):
        x = x.type(torch.int64)
        while x.ndim < 1:
            x = x.unsqueeze(-1)
        base = torch.eye(num_classes, device=x.device)
    else:
        x = to_numpy(x)
        x = x.astype(np.int64)
        while x.ndim < 1:
            x = np.expand_dims(x, axis=-1)
        base = np.eye(num_classes)
    
    if x.shape[-1] == 1:
        x = x.squeeze(-1)
    return base[x]


def logsumexp(x):
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

def softmax_logsumexp(x, temperature:float=1.0):
    x = x / temperature
    logsumexp_x = logsumexp(x)
    return np.exp(x - logsumexp_x)


class RunningMeanStd:
    def __init__(self, eps:float=1e-4, device:str="cpu", shape:tuple=()):
        self.mean = torch.zeros(shape).float()
        self.var = torch.ones(shape).float()
        self.count = eps
        self.to(device)
        
    def forward(self, x:torch.Tensor):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)
        
    def update(self, x:torch.Tensor):
        assert self.mean.device == self.var.device == x.device
        assert self.mean.shape == self.var.shape == x.shape[1:]
        batch_count = x.shape[0]
        assert batch_count > 1, "batch dimension must > 1 due to the variance calculation"
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        ma = self.var * self.count
        mb = batch_var * batch_count
        M2 = ma + mb + torch.pow(delta, 2) * self.count * batch_count / total_count
        new_var = M2 / total_count
        new_count = total_count
        
        self.mean, self.var, self.count = new_mean, new_var, new_count
        
    def to(self, device:str):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)




# --- debug ---
import time
# from typing import Union

class TimeChecker:
    def __init__(self):
        self.timers = {}
        
    def start_timer(self, flag_name:Union[int, str]=0):
        self.timers[flag_name] = [time.time(), None]
        
    def end_timer(self, flag_name:int=0):
        self.timers[flag_name][-1] = time.time()
        
    def get_time(self, flag_name:int=0):
        return self.timers[flag_name][1] - self.timers[flag_name][0]

