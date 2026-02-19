import argparse
import os
from typing import Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.utils import AttrDict, make_input
from utils.dataset import TrajectoryCropDataset
from utils.base import Base, Module
from utils.net import Stochastic, MLP, NormalMLP
from Spirl import Spirl


# q(z | s_i, a_i)
class SkillEncoder(Stochastic, Module):
    @Base.save_input()
    def __init__(self, d_state:int, d_low_action:int, d_high_action:int, d_hidden:int, d_seq:int, 
                 mode_limit_std:str="hard", log_std_range:Tuple=(-10, 2)):
        super().__init__(dist="normal", mode_limit_std=mode_limit_std, log_std_range=log_std_range)
        self.d_state, self.d_low_action, self.d_high_action, self.d_hidden, self.d_seq = d_state, d_low_action, d_high_action, d_hidden, d_seq
        self.input_layer = nn.Linear(in_features=d_state+d_low_action, out_features=d_hidden)
        self.lstm_layer = nn.LSTM(input_size=d_hidden, hidden_size=d_hidden, batch_first=True)
        self.output_layer = nn.Linear(in_features=d_hidden, out_features=d_high_action*2)
    
    def _check_input(self, s, a) -> None:
        assert s.ndim == a.ndim == 3, "input ndim error"
        assert s.shape[-1] == self.d_state and a.shape[-1] == self.d_low_action, "input dim error"
        assert s.shape[-2] == a.shape[-2] == self.d_seq, "sequence length error"
    
    def forward(self, s, a): # (d_batch, d_seq, d_state), (d_batch, d_seq, d_low_action)
        self._check_input(s, a)
        x = torch.concat([s, a], dim=-1)                # (d_batch, d_seq, d_state+d_low_action)
        in_feature = self.input_layer(x)                # (d_batch, d_seq, d_hidden)
        out, (_h, _c) = self.lstm_layer(in_feature)     # (d_batch, d_seq, d_hidden)
        out = out[:, -1]                                # (d_batch, d_hidden)
        out_feature = self.output_layer(out)            # (d_batch, d_high_action*2)
        mean, log_std = out_feature.chunk(2, dim=-1)    # (d_batch, d_high_action), (d_batch, d_high_action)
        return mean, log_std
    
    @torch.no_grad()
    @Module.transform_data(3, 3)
    def _inference(self, s, a, deterministic:bool=True):
        return self.sample(s, a, deterministic=deterministic)

class SpirlCL(Spirl):
    # q(z | s_i, a_i)
    def encode(self, states, low_actions): # ([d_batch], d_seq, d_state) ([d_batch], d_seq, d_low_action)
        states = make_input(states, dim=3, device=self.device)       # (d_batch, d_seq, d_state)
        low_actions = make_input(low_actions, dim=3, device=self.device)   # (d_batch, d_seq, d_low_action)
        return self.encoder.dist(states, low_actions) # (d_batch, d_high_action)~Normal
    
    # a ~ p(a | s, z)
    def decode(self, state, high_action): # ([d_batch], d_state), ([d_batch], d_high_action)
        state = make_input(state, dim=2, device=self.device)         # (d_batch, d_state)
        high_action = make_input(high_action, dim=2, device=self.device)     # (d_batch, d_high_action)
        return self.decoder(state, high_action) # (d_batch, d_low_action)
    
    # a_i ~ p(a_i | s_i, z)
    def decode_seq(self, states, high_action): # ([d_batch], d_seq, d_state), ([d_batch], d_high_action)
        d_state, d_high_action = states.shape[-1], high_action.shape[-1]
        d_seq = states.shape[-2]
        
        states = make_input(states, dim=3, device=self.device)           # (d_batch, d_seq, d_state)
        states = states.reshape(-1, d_state)                           # (d_batch*d_seq, d_state)
        high_action = make_input(high_action, dim=2, device=self.device)         # (d_batch, d_high_action)
        # (d_batch, d_high_action) -> (d_batch, d_high_action + d_high_action + ...) -> (d_batch*d_seq, d_high_action)
        high_actions = high_action.repeat(1, d_seq).reshape(-1, d_high_action) # (d_batch*d_seq, d_high_action)
        low_actions = self.decode(states, high_actions)                        # (d_batch*d_seq, d_low_action)
        return low_actions.reshape(-1, d_seq, low_actions.shape[-1]) # (d_batch, d_seq, d_low_action)
        
    def forward(self, states, low_actions): # (d_batch, d_seq, d_state), (d_batch, d_seq, d_low_action)
        # encode: q(z | s_i, a_i)~Normal
        q = self.encode(states, low_actions)           # (d_batch, d_high_action)~Normal
        # sample: z ~ q(z | s_i, a_i)
        z = q.rsample()                             # (d_batch, d_high_action)
        # decode: a'_i ~ p(a'_i | s_i, z_i)
        low_actions_recon = self.decode_seq(states, z)  # (d_batch, d_seq, d_low_action)
        
        # p(z | s_0)~Normal
        state = states[:, 0]             # (d_batch, d_state)
        prior = self.prior.dist(state)        # (d_batch, d_high_action)~Normal
        return AttrDict(q=q, z=z, low_actions_recon=low_actions_recon, prior=prior)

def main(args):
    conf = AttrDict(
        # --- experiment ---
        ENV             = args.env,
        DEVICE          = int(args.device) if args.device.isnumeric() else args.device,
        DATASET_PATH    = args.dataset_path,
        EPOCH           = args.epoch,
    )
    hyper_conf = AttrDict(
        d_batch             = 128,
        skill_length        = 10,
        skill_dim           = 10,
        lr                  = 0.001,
        l_kld               = 0.0005,
        encoder_d_hidden    = 128,
        decoder_d_hiddens   = [128, 128, 128, 128, 128, 128],
        prior_d_hiddens     = [128, 128, 128, 128, 128, 128, 128],
    )
    save_dir = f'./environments/{conf.ENV}/skill'
    os.makedirs(save_dir, exist_ok=True)
    # ----------------------------------------------
    # ----------------------------------------------
    # main
    # ----------------------------------------------
    # ----------------------------------------------

    # --- initialize offline dataset ---
    original_dataset = torch.load(conf.DATASET_PATH, map_location="cpu")
    env_conf = AttrDict(
        d_state=original_dataset["observations"].shape[-1], d_action=original_dataset["actions"].shape[-1],
    )
    dataset = TrajectoryCropDataset(original_dataset, hyper_conf.skill_length)
    train_loader = DataLoader(dataset, batch_size=hyper_conf.d_batch, shuffle=True, drop_last=True)

    # --- initialize model ---
    # NOTE: encoder: q(z | s_i, a_i) : s(d_batch, skill_length, d_state), a(d_batch, skill_length, d_low_action) -> (d_batch, d_high_action)~Normal
    encoder = SkillEncoder(
        d_state=env_conf.d_state, d_low_action=env_conf.d_action, d_high_action=hyper_conf.skill_dim, 
        d_hidden=hyper_conf.encoder_d_hidden, d_seq=hyper_conf.skill_length, mode_limit_std="tanh",
    )
    # NOTE: decoder: p(a | s, z) : s(d_batch, d_state), z(d_batch, d_high_action) -> a(d_batch, d_low_action)
    decoder = MLP(
        d_inputs=[env_conf.d_state, hyper_conf.skill_dim], d_outputs=[env_conf.d_action], d_hiddens=hyper_conf.decoder_d_hiddens,
        leakyrelu=True, leakyrelu_slope=0.2, batchnorm=True
    )
    # NOTE: p(z | s_0) : s_0(d_batch, d_state)         -> (d_batch, d_high_action)~Normal
    prior = NormalMLP(
        d_inputs=[env_conf.d_state], d_outputs=[hyper_conf.skill_dim], d_hiddens=hyper_conf.prior_d_hiddens,
        leakyrelu=True, leakyrelu_slope=0.2, batchnorm=True, mode_limit_std="tanh"
    )
    
    model = SpirlCL(encoder, decoder, prior, lr=hyper_conf.lr, beta=hyper_conf.l_kld).to(conf.DEVICE)

    # --- main ---
    step = 0
    pbar = tqdm(total=conf.EPOCH * len(train_loader))
    for epoch in range(1, conf.EPOCH + 1):
        model.train()
        for data in train_loader:
            step += 1
            states = data.observations.to(model.device) # (d_batch, skill_length, d_state)
            actions = data.actions.to(model.device) # (d_batch, skill_length, d_action)
            result = model.update(states, actions)
            pbar.update(1)
            
    pbar.close()
    save_path = f'{save_dir}/Skill-Env={conf.ENV}_Epoch={conf.EPOCH}_Step={step}.pt'
    Base.saves(
        path=save_path,
        model={"encoder": encoder, "decoder": decoder, "prior": prior},
    )
    print(f'--- Save skill model to {save_path} ---')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["kitchen", "office", "maze", "antmaze"], required=True)
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--epoch", default=200, type=int)
    args = parser.parse_args()
    main(args)
    print("--- Done!!! ---")