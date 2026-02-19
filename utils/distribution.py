from typing import Tuple
import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence as kl_torch
import abc

class Distribution(abc.ABC):
    @abc.abstractmethod
    def to(self, *args, **kwargs): 
        pass
    
    @abc.abstractmethod
    def detach(self, *args, **kwargs):
        pass


class Normal(Distribution, torch.distributions.Normal):
    @property
    def mu(self): return self.mean
    @property
    def sigma(self): return self.stddev
    @property
    def std(self): return self.stddev
    @property
    def shape(self): return self.mu.shape
    @property
    def device(self): return self.mean.device
    
    def detach(self):
        return type(self)(self.mean.detach(), self.std.detach())
    
    def to(self, *args, **kwargs):
        return type(self)(self.mean.to(*args, **kwargs), self.std.to(*args, **kwargs))
    
    def __repr__(self):
        return f'{self.__class__.__name__}' + \
                f'[mu{tuple(self.mean.shape)}, sigma{tuple(self.std.shape)}, {str(self.mean.device)}]'


class TanhNormal(Distribution, torch.distributions.Distribution):
    def __init__(self, loc, scale):
        self.normal = Normal(loc, scale)
    def cdf(self, value):
        raise NotImplementedError("dist error")
    def entropy(self):
        raise NotImplementedError("dist error")
    def icdf(self, value):
        raise NotImplementedError("dist error")
    def perplexity(self):
        raise NotImplementedError("dist error")
    
    @property
    def mean(self):
        return self.normal.mean.tanh()
    @property
    def mu(self):
        return self.normal.mean.tanh()
    @property
    def stddev(self):
        return self.normal.stddev
    @property
    def std(self):
        return self.normal.stddev
    @property
    def device(self):
        return self.normal.mean.device
    
    def log_prob_from_pre_tanh(self, pre_tanh_value):
        log_prob = self.normal.log_prob(pre_tanh_value)
        correction = - 2.0 * (torch.tensor(2.0).log() - pre_tanh_value
            - F.softplus(-2.0 * pre_tanh_value)
        )
        return (log_prob + correction)
    
    
    def log_prob(self, value, value_range:Tuple[float]=(-1, 1)):
        value = value.clamp(min=-0.999999, max=0.999999)
        return self.log_prob_from_pre_tanh(value.atanh())
    
    def sample(self, sample_shape=torch.Size()):
        z = self.normal.sample(sample_shape=sample_shape)
        return z.tanh()
    
    def rsample(self, sample_shape=torch.Size()):
        z = self.normal.rsample(sample_shape=sample_shape)
        return z.tanh()
    
    def to(self, *args, **kwargs):
        return type(self)(self.normal.mean.to(*args, **kwargs), self.normal.stddev.to(*args, **kwargs))
    
    def detach(self):
        return type(self)(self.normal.mean.detach(), self.normal.stddev.detach())
    
    def __repr__(self):
        return f'{self.__class__.__name__}' + \
                f'[mu{tuple(self.normal.mean.shape)}, sigma{tuple(self.normal.stddev.shape)}, {str(self.normal.mean.device)}]'


def kl_divergence(p_dist:Distribution, q_dist:Distribution, mode_multivariate=False, mode_torch=True):
    """ analytic KL( N(p_mean, p_std^2) || N(q_mean, q_std^2) ) """
    # NOTE: p_dist, q_dist = ([dBatch], dLatent) ~ Normal
    if mode_torch:
        kld = kl_torch(p_dist, q_dist)
    else:
        p_mean, p_logStd = p_dist.mean, p_dist.stddev.log()
        q_mean, q_logStd = q_dist.mean, q_dist.stddev.log()
        # NOTE: log(q_std) - log(p_std) + (p_std^2 + (p_mean - q_mean)^2) / (2*q_std^2) - 0.5
        kld =  (q_logStd - p_logStd) + ( (2*p_logStd).exp() + (p_mean-q_mean).pow(2) ) / ( 2*(2*q_logStd).exp() ) - 0.5
    
    if mode_multivariate:
        kld = kld.sum(dim=-1, keepdim=True) # ([dBatch], 1)
    return kld

def kl_divergence_mc(p_dist:Distribution, q_dist:Distribution, n_sample=50, mode_multivariate=False):
    """ sample-based(MC) KL( N(p_mean, p_std^2) || N(q_mean, q_std^2) ) """
    p = p_dist.rsample(sample_shape=(n_sample, )) # (n_sample, [dBatch], dLatent)
    kld = (p_dist.log_prob(p) - q_dist.log_prob(p)).mean(dim=0) # ([dBatch], dLatent)
    if mode_multivariate:
        kld = kld.sum(dim=-1, keepdim=True) # ([dBatch], 1)
    return kld



