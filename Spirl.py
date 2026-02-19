import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple

from utils.utils import AttrDict, make_input
from utils.distribution import Normal, kl_divergence
from utils.base import Base, Module, MainModule
from utils.net import Stochastic, Deterministic


# q(z | a_i)
class SkillEncoder(Stochastic, Module):
    @Base.save_input()
    def __init__(self, d_low_action:int, d_high_action:int, d_hidden:int, d_seq:int,
                 mode_limit_std:str="hard", log_std_range:Tuple=(-10, 2)):
        super().__init__(dist="normal", mode_limit_std=mode_limit_std, log_std_range=log_std_range)
        self.d_low_action, self.d_high_action, self.d_hidden, self.d_seq = d_low_action, d_high_action, d_hidden, d_seq
        self.input_layer = nn.Linear(in_features=d_low_action, out_features=d_hidden)
        self.lstm_layer = nn.LSTM(input_size=d_hidden, hidden_size=d_hidden, batch_first=True)
        self.output_layer = nn.Linear(in_features=d_hidden, out_features=d_high_action*2)
    
    def _check_input(self, *args) -> None:
        assert len(args) == 1, "different number of inputs"
        assert args[0].ndim == 3, "input ndim error"
        assert args[0].shape[-1] == self.d_low_action and args[0].shape[-2] == self.d_seq, "input dim error"
    
    def forward(self, a): # (d_batch, d_seq, d_low_action)
        self._check_input(a)
        in_feature = self.input_layer(a)            # (d_batch, d_seq, d_hidden)
        out, (_h, _c) = self.lstm_layer(in_feature) # (d_batch, d_seq, d_hidden)
        out = out[:, -1]                            # (d_batch, d_hidden)
        out_feature = self.output_layer(out)        # (d_batch, d_high_action*2)
        mean, log_std = out_feature.chunk(2, dim=-1)# (d_batch, d_high_action), (d_batch, d_high_action)
        return mean, log_std

    @torch.no_grad()
    @Module.transform_data(3)
    def _inference(self, a, deterministic:bool=True):      # ([d_batch], d_seq, d_low_action)
        return self.sample(a, deterministic=deterministic) # ([d_batch], d_high_action)

# p(a_i | z)
class SkillDecoder(Deterministic, Module):
    @Base.save_input()
    def __init__(self, d_low_action:int, d_high_action:int, d_hidden:int, d_seq:int):
        super().__init__()
        self.d_low_action, self.d_high_action, self.d_hidden, self.d_seq = d_low_action, d_high_action, d_hidden, d_seq
        self.input_layer = nn.Linear(in_features=d_high_action, out_features=d_hidden)
        self.lstm_layer = nn.LSTM(input_size=d_hidden, hidden_size=d_hidden, batch_first=True)
        self.output_layer = nn.Linear(in_features=d_hidden, out_features=d_low_action)
    
    def forward(self, z): # (d_batch, d_high_action)
        assert (z.shape[-1] == self.d_high_action)
        
        in_feature = self.input_layer(z)                                    # (d_batch, d_hidden)
        in_feature = in_feature.unsqueeze(dim=1).expand(-1, self.d_seq, -1) # (d_batch, d_seq, d_hidden)
        out, (_h, _c) = self.lstm_layer(in_feature)                         # (d_batch, d_seq, d_hidden)
        actions = self.output_layer(out)                                    # (d_batch, d_seq, dActinoL)
        return actions # (d_batch, d_seq, d_low_action)


class Spirl(MainModule):
    @Base.save_input(exclude=["encoder", "decoder", "prior"])
    def __init__(self, encoder:Module, decoder:Module, prior:Module,
                 lr:float, beta:float):
        super().__init__()
        self.lr = lr
        self.beta = beta
        # NOTE: encoder: q(z | a_i) : a(d_batch, d_seq, d_low_action) -> (d_batch, d_high_action)~Normal
        self.encoder = encoder
        # NOTE: decoder: p(a_i | z) : z(d_batch, d_high_action)       -> a(d_batch, d_seq, d_low_action)
        self.decoder = decoder
        # NOTE: prior  : p(z | s_0) : s_0(d_batch, dState)            -> (d_batch, d_high_action)~Normal
        self.prior = prior
        # NOTE: std prior : p(z) : Normal(0, 1)
        self.prior_std = Normal(0, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    # q(z | a_i)
    def encode(self, low_actions): # ([d_batch], d_seq, d_low_action)
        low_actions = make_input(low_actions, dim=3, device=self.device) # (d_batch, d_seq, d_low_action)
        return self.encoder.dist(low_actions) # (d_batch, d_high_action)~Normal
    
    # a_i ~ p(a_i | z)
    def decode(self, actionH): # ([d_batch], d_high_action)
        actionHs = make_input(actionH, dim=2, device=self.device) # (d_batch, d_high_action)
        return self.decoder(actionHs) # (d_batch, d_seq, d_low_action)
    
    def forward(self, states, low_actions): # (d_batch, d_seq, dState), (d_batch, d_seq, d_low_action)
        # q(z | a_i)~Normal
        q = self.encode(low_actions)        # (d_batch, d_high_action)~Normal
        # z ~ q(z | a_i)
        z = q.rsample()                     # (d_batch, d_high_action)
        # p(a'_i | z)
        low_actions_recon = self.decode(z)  # (d_batch, d_seq, d_low_action)
        
        # p(z | s_0)
        state = states[:, 0]           # (d_batch, dState)
        prior = self.prior.dist(state) # (d_batch, d_high_action)~Normal
        
        return AttrDict(q=q, z=z, low_actions_recon=low_actions_recon, prior=prior)
    
    def loss_recon(self, low_actions, output): # (d_batch, d_seq, d_low_action)
        low_actions_recon = output.low_actions_recon # (d_batch, d_seq, d_low_action)
        # --- loss reconstruction ---
        # maximize E_{z~q(z|a_i)} [ log p(a_i|z) ] -> minimize MSE(a, a')

        loss_recon = F.gaussian_nll_loss(
            input=low_actions_recon,
            target=low_actions,
            var=torch.ones_like(low_actions_recon),
            full=True, reduction="mean"
        )
        # -Normal(actions_recon, torch.ones_like(actions_recon)).log_prob(actions).mean()
        
        # loss_recon = F.mse_loss(actions, actions_recon, reduction="none")   # (d_batch, dSeqLen, dAction)
        # loss_recon = loss_recon.sum(dim=(1, 2)).mean()                      # scalar
        return loss_recon
        
    def loss_kld(self, output):
        q = output.q                                    # (d_batch, d_high_action)~Normal
        # --- loss regularization ---
        # minimize KL( q(z|a_i) || N(0, 1) )
        # loss_kld = kl_divergence(q, self.prior_std)     # (d_batch, d_high_action)
        
        loss_kld = kl_divergence(q, self.prior_std, mode_multivariate=False) # (d_batch, d_high_action)
        loss_kld = loss_kld.mean()
        return loss_kld
    
    def loss_prior(self, output):
        prior = output.prior  # (d_batch, d_high_action)~Normal
        z = output.z.detach() # (d_batch, d_high_action)
        
        # loss_prior = -prior.log_prob(z).mean()
        loss_prior = -prior.log_prob(z).sum(dim=-1).mean()
        return loss_prior
    
    def loss(self, states, low_actions): # (d_batch, d_seq, dState), (d_batch, d_seq, d_low_action)
        output = self.forward(states, low_actions)
        
        loss_recon  = self.loss_recon(low_actions, output)
        loss_kld    = self.loss_kld(output)
        loss_prior  = self.loss_prior(output)
        
        loss = loss_recon + self.beta*loss_kld + loss_prior
        
        return AttrDict(
            loss=loss, 
            loss_recon=loss_recon.detach().cpu().item(),
            loss_kld=loss_kld.detach().cpu().item(),
            loss_prior=loss_prior.detach().cpu().item() 
            )

    def update(self, states, low_actions):
        output = self.loss(states, low_actions)
        self.optimizer.zero_grad()
        output.loss.backward()
        self.optimizer.step()
        
        output.loss = output.loss.detach().cpu().item()
        return output