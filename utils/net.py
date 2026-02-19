import abc
from typing import Optional, Union, List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import create_mlp, make_input, make_output
from .base import Base, Module, ToDeviceMixin
from .distribution import Distribution, Normal, TanhNormal

LOG_STD_MIN = -10
LOG_STD_MAX = 2
mode_clamp = ["none", "hard", "sigmoid", "tanh", "atan"]
mode_dist = ["normal", "tanh"]

def clamp(fn:str, value:torch.Tensor, value_min:float, value_max:float):
    """ (-inf, inf) -> [value_min, value_max] """
    if fn == "hard":
        return value.clamp(min=value_min, max=value_max)
    elif fn == "sigmoid":
        return value_min + value.sigmoid() * (value_max - value_min)
    elif fn == "tanh":
        return value_min + 0.5 * (value.tanh() + 1) * (value_max - value_min)
    elif fn == "atan":
        return value_min + (1.0 / torch.pi) * (value.atan() + (torch.pi / 2.0)) * (value_max - value_min)
    else:
        raise NotImplementedError

def reverse_clamp(fn:str, value:torch.Tensor, value_min:float, value_max:float):
    eps = 0.000001
    if fn == "hard":
        return value
    elif fn == "sigmoid":
        scaled_value = (value - value_min) / (value_max - value_min)
        scaled_value = scaled_value.clamp(min=eps, max=1-eps)
        return torch.log(scaled_value / (1 - scaled_value))
    elif fn == "tanh":
        scaled_value = ( 2 * (value - value_min) / (value_max - value_min) ) - 1
        scaled_value = scaled_value.clamp(min=-1+eps, max=1-eps)
        return scaled_value.atanh()
    elif fn == "atan":
        scaled_value = ( (value - value_min) / (1.0 / torch.pi) / (value_max - value_min) ) - (torch.pi / 2.0)
        scaled_value = scaled_value.clamp(min=-torch.pi/2 + eps, max=torch.pi/2 - eps)
        return scaled_value.tan()
    else:
        raise NotImplementedError


# p(y | x) ~ Deterministic
class Deterministic:
    @torch.no_grad()
    @Module.transform_data(2, use_args=True)
    def _inference(self, *args): # ([dBatch], ...)
        return self.forward(*args)
        
# p(y | x) ~ Distribution
class Stochastic:
    def __init__(self, dist:str, 
                 mode_limit_mean:str="none", mean_range:Optional[Tuple]=None,
                 mode_limit_std:str="none", log_std_range:Optional[Tuple]=None, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mode_limit_mean in mode_clamp and mode_limit_std in mode_clamp
        assert dist in mode_dist
        self.mode_limit_mean, self.mean_range = mode_limit_mean, mean_range
        self.mode_limit_std, self.log_std_range = mode_limit_std, log_std_range
        if dist == "normal":
            self._dist = Normal
        elif dist == "tanh":
            self._dist = TanhNormal
        else: raise NotImplementedError
    
    # p(- | x)
    def dist(self, *args): # (dBatch, ...)
        mean, log_std = self.forward(*args) # (dBatch, ...)
        if self.mode_limit_mean != "none":
            mean = clamp(self.mode_limit_mean, mean, *self.mean_range)
        if self.mode_limit_std != "none":
            log_std = clamp(self.mode_limit_std, log_std, *self.log_std_range)
        return self._dist(mean, log_std.exp()) # (dBatch, ...) ~ Distribution
    
    # y ~ p(- | x)
    def sample(self, *args, deterministic:bool=False, 
               with_dist:bool=False, with_log_prob:bool=False): # (dBatch, ...)
        dist = self.dist(*args) # (dBatch, ...) ~ Distribution
        sample = dist.mean if deterministic else dist.rsample() # (dBatch, ...)
        if with_log_prob:
            assert isinstance(dist, Distribution)
            log_prob = dist.log_prob(sample).sum(dim=-1, keepdim=True) # (dBatch, ..., 1)
        
        if with_dist and with_log_prob:
            return sample, dist, log_prob # (dBatch, ...), (dBatch, ...)~Distribution, (dBatch, ..., 1)
        elif with_dist:
            return sample, dist
        elif with_log_prob:
            return sample, log_prob
        else:
            return sample

    @torch.no_grad()
    @Module.transform_data(2, use_args=True) # NOTE: input ndim=2
    def _inference(self, *args, deterministic:bool=True):
        return self.sample(*args, deterministic=deterministic)


class MLP(Deterministic, Module):
    @Base.save_input()
    def __init__(self, d_inputs:List[int], d_outputs:List[int], d_hiddens:List[int],
                 leakyrelu:bool=False, leakyrelu_slope:Optional[float]=None, batchnorm:bool=False,
                 mode_limit_output:str="none", output_range:Optional[Tuple]=None):
        super().__init__()
        assert mode_limit_output in mode_clamp
        self.d_inputs, self.d_outputs, self.d_hiddens = d_inputs, d_outputs, d_hiddens
        self.d_input, self.d_output = sum(d_inputs), sum(d_outputs)
        self.mode_limit_output, self.output_range = mode_limit_output, output_range
        self.net = create_mlp(
            d_in=self.d_input, d_out=self.d_output, d_hiddens=self.d_hiddens,
            leakyrelu=leakyrelu, leakyrelu_slope=leakyrelu_slope, batchnorm=batchnorm
        )
    
    def _check_input(self, *args) -> None:
        assert len(self.d_inputs) == len(args), "different number of inputs"
        assert all([d_input == arg.shape[-1] for d_input, arg in zip(self.d_inputs, args)]), "feature dim error"
    
    def forward(self, *args): # (dBatch, d_inputs[0]), (dBatch, d_inputs[1]), ...
        self._check_input(*args)
        feature = torch.cat(args, dim=-1) # (dBatch, d_input)
        out = self.net(feature) # (dBatch, d_outputs)
        if self.mode_limit_output != "none": # NOTE: limit output range
            out = clamp(self.mode_limit_output, out, *self.output_range) # (dBatch, d_outputs)
        if len(self.d_outputs) > 1: # NOTE: multiple output
            out = out.split(self.d_outputs, dim=-1)
        return out # (dBatch, d_output)

# NOTE: p(y | x) ~ Normal(mu, sigma^2)
#       mu   : (-inf, inf)
#       sigma: [e^-10, e^2]=[0.00004, 7.38]
class NormalMLP(Stochastic, MLP):
    @Base.save_input()
    def __init__(self, d_inputs:List[int], d_outputs:List[int], d_hiddens:List[int],
                 leakyrelu:bool=False, leakyrelu_slope:Optional[float]=None, batchnorm:bool=False,
                 mode_limit_mean:str="none", mean_range:Optional[Tuple]=None,
                 mode_limit_std:str="hard", log_std_range:Optional[Tuple]=(LOG_STD_MIN, LOG_STD_MAX)):
        assert len(d_outputs) == 1, "support single output only"
        d_outputs = [d_output for d_output in d_outputs for _ in range(2)]
        super().__init__(
            d_inputs=d_inputs, d_outputs=d_outputs, d_hiddens=d_hiddens,
            leakyrelu=leakyrelu, leakyrelu_slope=leakyrelu_slope, batchnorm=batchnorm,
            mode_limit_mean=mode_limit_mean, mean_range=mean_range,
            mode_limit_std=mode_limit_std, log_std_range=log_std_range,
            dist="normal", mode_limit_output="none"
        )

# NOTE: p(y | x) ~ Tanh( Normal(mu, sigma^2) )
#       mu   : (-1, 1)
#       sigma: [e^-10, e^2]=[0.00004, 7.38]
class TanhNormalMLP(Stochastic, MLP):
    @Base.save_input()
    def __init__(self, d_inputs:List[int], d_outputs:List[int], d_hiddens:List[int], sample_range:Tuple=(-1, 1),
                 leakyrelu:bool=False, leakyrelu_slope:Optional[float]=None, batchnorm:bool=False,
                 mode_limit_mean:str="none", mean_range:Optional[Tuple]=None,
                 mode_limit_std:str="hard", log_std_range:Optional[Tuple]=(LOG_STD_MIN, LOG_STD_MAX)):
        assert len(d_outputs) == 1, "support single output only"
        d_outputs = [d_output for d_output in d_outputs for _ in range(2)]
        super().__init__(
            d_inputs=d_inputs, d_outputs=d_outputs, d_hiddens=d_hiddens,
            leakyrelu=leakyrelu, leakyrelu_slope=leakyrelu_slope, batchnorm=batchnorm,
            mode_limit_mean=mode_limit_mean, mean_range=mean_range,
            mode_limit_std=mode_limit_std, log_std_range=log_std_range,
            dist="tanh", mode_limit_output="none"
        )
        self.sample_range = sample_range
        
    
    def sample(self, *args, deterministic:bool=False, 
               with_dist:bool=False, with_log_prob:bool=False): # (dBatch, ...)
        dist = self.dist(*args) # (dBatch, ...) ~ Distribution
        pre_sample = dist.normal.mean if deterministic else dist.normal.rsample() # (dBatch, ...)
        if with_log_prob:
            assert isinstance(dist, Distribution)
            log_prob = dist.log_prob_from_pre_tanh(pre_sample).sum(dim=-1, keepdim=True) # (dBatch, ..., 1)
        sample = clamp("tanh", pre_sample, *self.sample_range)
        
        if with_dist and with_log_prob:
            return sample, dist, log_prob # (dBatch, ...), (dBatch, ...)~Distribution, (dBatch, ..., 1)
        elif with_dist:
            return sample, dist
        elif with_log_prob:
            return sample, log_prob
        else:
            return sample
        


class SinusoidalPosEncoder(ToDeviceMixin, nn.Module): 
    def __init__(self, dim:int, theta:int=10000, max_len:int=5000):
        super().__init__()
        embedding_map = torch.zeros(size=(max_len, dim))
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(dim=-1) # (max_len, 1)
        temp = torch.arange(0, dim, step=2, dtype=torch.float32)
        if dim % 2 == 0:
            embedding_map[:, 1::2] = torch.cos(pos / theta ** (temp / dim))
        else:
            embedding_map[:, 1::2] = torch.cos(pos / theta ** (temp / dim))[:, :-1]
        embedding_map[:, 0::2] = torch.sin(pos / theta ** (temp / dim))
        self.register_buffer("embedding_map", embedding_map, persistent=False)
        
    def forward(self, pos:torch.Tensor): # ([b], 1) -> ([b], dim)
        # NOTE: pos: position(or timestep)
        ndim, shape = pos.ndim, pos.shape
        assert ndim == 1 or shape[-1] == 1, f'input ndim({ndim}) or shape({shape}) error'
        pos = pos.view(-1) # flatten
        embedding = self.embedding_map[pos]
        if ndim > 1:
            embedding = embedding.view(*shape[:-1], embedding.shape[-1])
        return embedding