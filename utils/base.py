import abc
import gym
import numpy as np
import torch
from contextlib import contextmanager, ExitStack
from typing import Tuple, Union

from .utils import AttrDict, make_env, get_env_conf, make_input, make_output
from .distribution import Distribution

class ToDeviceMixin:
    _device = torch.device("cpu")
    
    @property
    def device(self):
        return self._device
    
    def _apply(self, fn):
        if "t" in fn.__code__.co_varnames:
            empty = torch.empty(0)
            device = fn(empty).device
            self._device = device
            
        for name, var in vars(self).items():
            if isinstance(var, Distribution):
                setattr(self, name, var.to(self._device))
        
        return super()._apply(fn)


class FreezeMixin:
    _freeze_mode = False
    
    def requires_grad_(self, requires_grad:bool=True):
        for name, m in dict(self.named_modules()).items(): 
            if isinstance(m, Base):
                m._freeze_mode = not requires_grad
        return super().requires_grad_(requires_grad)
    
    @contextmanager
    def freeze_mode(self, mode_keep_submodule:bool=False):
        modules = dict(self.named_modules())
        prev_modes = {name:m._freeze_mode for name, m in modules.items() if isinstance(m, Base)}
        self.requires_grad_(False)
        if mode_keep_submodule:
            for i, (name, mode) in enumerate(prev_modes.items()):
                if i == 0: continue # skip (main) module
                modules[name].requires_grad_(requires_grad= not mode)
        yield
        for name, mode in prev_modes.items():
            modules[name].requires_grad_(requires_grad=not mode)
    
    @contextmanager
    def unfreeze_mode(self, mode_keep_submodule:bool=False):
        modules = dict(self.named_modules())
        prev_modes = {name:m._freeze_mode for name, m in modules.items() if isinstance(m, Base)}
        self.requires_grad_(True)
        if mode_keep_submodule:
            for i, (name, mode) in enumerate(prev_modes.items()):
                if i == 0: continue # skip (main) module
                modules[name].requires_grad_(requires_grad= not mode)
        yield
        for name, mode in prev_modes.items():
            modules[name].requires_grad_(requires_grad=not mode)
    
    @staticmethod
    @contextmanager
    def freeze_modes(*models, mode_keep_submodule:bool=False):
        with ExitStack() as stack:
            for model in models:
                stack.enter_context(model.freeze_mode(mode_keep_submodule=mode_keep_submodule))
            yield
    
    @staticmethod
    @contextmanager
    def unfreeze_modes(*models, mode_keep_submodule:bool=False):
        with ExitStack() as stack:
            for model in models:
                stack.enter_context(model.unfreeze_mode(mode_keep_submodule=mode_keep_submodule))
            yield

class EvaluationMixin:
    _evaluation_mode = False
    
    def train(self, mode:bool=True):
        self._evaluation_mode = not mode
        return super().train(mode)
    
    @contextmanager
    def evaluation_mode(self, mode_keep_submodule:bool=False):
        modules = dict(self.named_modules())
        prev_modes = {name:m._evaluation_mode for name, m in modules.items() if isinstance(m, Base)}
        self.eval()
        if mode_keep_submodule:
            for i, (name, mode) in enumerate(prev_modes.items()):
                if i == 0: continue # main module
                modules[name].train(mode=not mode)
        yield
        for name, mode in prev_modes.items():
            modules[name].train(mode=not mode)
            
    @contextmanager
    def train_mode(self, mode_keep_submodule:bool=False):
        """evaluation_mode -> train_mode"""
        modules = dict(self.named_modules())
        prev_modes = {name:m._evaluation_mode for name, m in modules.items() if isinstance(m, Base)}
        self.train()
        if mode_keep_submodule:
            for i, (name, mode) in enumerate(prev_modes.items()):
                if i == 0: continue
                modules[name].train(mode= not mode)
        yield
        for name, mode in prev_modes.items():
            modules[name].train(mode=not mode)

    @staticmethod
    @contextmanager
    def evaluation_modes(*models, mode_keep_submodule:bool=False):
        """multiple evaluation_mode"""
        with ExitStack() as stack:
            for model in models:
                stack.enter_context(model.evaluation_mode(mode_keep_submodule=mode_keep_submodule))
            yield

    @staticmethod
    @contextmanager
    def train_modes(*models, mode_keep_submodule:bool=False):
        """multiple train_mode"""
        with ExitStack() as stack:
            for model in models:
                stack.enter_context(model.train_mode(mode_keep_submodule=mode_keep_submodule))
            yield

class SaveMixin:
    _input_data = None

    def export(self) -> dict:
        data = dict()
        data["parameter"] = self.state_dict()
        if self._input_data is not None:
            data["input"] = self._input_data
        return data
    
    def save(self, path:str) -> None:
        torch.save(self.export(), path)

    @staticmethod
    def save_input(include=[], exclude=[]):
        if include and exclude:
            raise Exception("Cannot use both options [include, exclude] togather.")
        
        def save_input_decorator(func):
            def wrapper(self, *args, **kwargs):
                if self._input_data is None: 
                    var_names = list(func.__code__.co_varnames[1:])
                    save_kwargs = kwargs.copy()
                    save_kwargs.update({var_name:arg for var_name, arg in zip(var_names, args)})
                    if include:
                        save_kwargs = {var_name:arg for var_name, arg in save_kwargs.items() if var_name in include}
                    elif exclude:
                        save_kwargs = {var_name:arg for var_name, arg in save_kwargs.items() if var_name not in exclude}
                    self._input_data = save_kwargs
                return func(self, *args, **kwargs)
            return wrapper
        return save_input_decorator
    
    @classmethod
    def _from(cls, exported_model:dict, **kwargs):
        """ exported_model e.g.
                {"parameter": ..., "input", ...}
        """
        model_input = {}
        if "input" in exported_model.keys():
            model_input.update(exported_model["input"])
        model_input.update(kwargs)
        
        model = cls(**model_input)
        model.load_state_dict(exported_model["parameter"])
        return model

    @classmethod
    def load(cls, path:str, **kwargs):
        raw = torch.load(path, map_location="cpu")
        return cls._from(raw, **kwargs)

    @staticmethod
    def saves(path:str, model:dict, **kwargs):
        data = dict()
        data["model"] = dict()
        for name, v in model.items():
            if isinstance(v, list):
                data["model"][name] = [item.export() for item in v]
            else:
                data["model"][name] = v.export()
        data.update(kwargs)
        torch.save(data, path)
    
    @staticmethod
    def loads(path:str, model:dict={}):
        raw = torch.load(path, map_location="cpu")
        raw_model = raw.pop("model")
        
        out = {}
        for name, model_info in model.items():
            if isinstance(model_info, dict):
                model_cls = model_info["model"]
                model_input = {k:v for k, v in model_info.items() if k != "model"}
            else:
                model_cls = model_info
                model_input = {}
                
            if isinstance(raw_model[name], list):
                out[name] = [model_cls._from(r, **model_input) for r in raw_model[name]]
            else:
                out[name] = model_cls._from(raw_model[name], **model_input)
        if raw:
            out.update(raw)
        return out

class Base(ToDeviceMixin, FreezeMixin, EvaluationMixin, SaveMixin, torch.nn.Module):
    def show_info(self, mode_base_only:bool=False):
        """print model mode"""
        name_max_len = max([len(name) for name, m in dict(self.named_modules()).items() if (mode_base_only and isinstance(m, Base)) or (not mode_base_only)])
        name_max_len = max(len("(main)"), name_max_len)
        
        title = f'{"name": ^{name_max_len}} | {"base": ^6s} | {"eval": ^6s} | {"freeze": ^6s} | {"infer": ^6s} | {"device": ^6s}'
        print(title)
        print("-"*len(title))
        for name, m in dict(self.named_modules()).items():
            name = name if not name == "" else "(main)"
            is_base = isinstance(m, Base)
            is_module = isinstance(m, Module)
            is_eval = not m.training
            is_freeze = m._freeze_mode if is_base else "-"
            is_infer = m._inference_mode if isinstance(m, Module) else "-"
            device = m.device if isinstance(m, Base) else "-"
            if mode_base_only and not is_base:
                continue
            
            print(f'{name:<{name_max_len}s} | {str(is_base):^6s} | {str(is_eval):^6s} | {str(is_freeze):^6s} | {str(is_infer):^6s} | {str(device):^6s}')
        print("-"*len(title))
    # for debug
    def show_parameter_size(self):
        """print model parameter size"""
        total_params = 0
        for p in self.parameters():
            total_params += p.numel()
        print(f'Total parameters: {total_params}')
        print(f'Total size: {(total_params * 4) / 1024**2:.2f} MB')
        


class Module(abc.ABC, Base):
    _inference_mode = False
    
    @abc.abstractmethod
    def forward():
        """
            for train
            - use batch input only
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def _inference(self, *args, **kwargs):
        """
            for inference
            - if single input, add batch dimension
            - if single input, remove output batch dimension
            - transfer input device to model device
            - if stochastic model, output will sample
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if self._inference_mode:
            return self._inference(*args, **kwargs)
        else:
            return self.forward(*args, **kwargs)
    
    @contextmanager
    def inference_mode(self, mode_keep_submodule:bool=False):
        """
            to inference mode
            - evaluation_mode = True
            - inference_mode = True
        """
        modules = dict(self.named_modules())
        prev_modes_eval = {name:m._evaluation_mode for name, m in modules.items() if isinstance(m, Base)}
        prev_modes_infer = {name:m._inference_mode for name, m in modules.items() if isinstance(m, Module)}
        self._inference_mode = True
        self.eval()
        for i, (name, mode_eval) in enumerate(prev_modes_eval.items()):
            if i == 0: continue
            if mode_keep_submodule:
                modules[name].train(mode=not mode_eval)
            else:
                if prev_modes_infer.get(name, False):
                    modules[name]._inference_mode = True
        
        yield
        
        self._inference_mode = False
        for name, mode_eval in prev_modes_eval.items():
            modules[name].train(mode=not mode_eval)
            if prev_modes_infer.get(name, False):
                modules[name]._inference_mode = prev_modes_infer[name]
    
    @staticmethod
    @contextmanager
    def inference_modes(*models, mode_keep_submodule:bool=False):
        with ExitStack() as stack:
            for model in models:
                stack.enter_context(model.inference_mode(mode_keep_submodule=mode_keep_submodule))
            yield
    
    @staticmethod
    def transform_data(
        *n_dims, use_args:bool=False,
        input_batch:bool=True, input_device:bool=True,
        output_batch:bool=False, output_device:bool=False,
    ):
        if use_args and len(n_dims) > 1:
            raise Exception("use_args option cannot be used with multiple n_dims")
        
        def transform_data_decorator(func):
            def wrapper(self, *args, **kwargs):
                if input_batch:
                    dims = n_dims
                    if use_args:
                        dims = [n_dims[0] for _ in range(len(args))]
                    args = [make_input(arg, dim=dim, device=self.device if input_device else None) for arg, dim in zip(args, dims)]
                
                result = func(self, *args, **kwargs) # (dBatch, ...)
                
                if not output_batch:
                    if isinstance(result, tuple) or isinstance(result, list): # multiple output
                        result = [make_output(r) for r in result]
                    elif isinstance(result, torch.Tensor): # single output
                        result = make_output(result) # ([dBatch], ...)
                    else: raise NotImplementedError
                return result
            return wrapper
        return transform_data_decorator
    

class MainModule(abc.ABC, Base):
    @abc.abstractmethod
    def update():
        raise NotImplementedError


# ------ environment class ------
class Environment:
    """
        support both new/old version Gym
    """
    def __init__(self, env:Union[str, gym.Env], episode_max_step:int=-1, truncated_done:bool=False, mode_test:bool=False, **kwargs):
        if isinstance(env, str):
            self._name = env
            self._env = make_env(name=env, mode_test=mode_test, time_limit=False, **kwargs)
        else:
            self._name = env.spec.id
            self._env = env
        self._episode_max_step = episode_max_step if episode_max_step != -1 else self._env.spec.max_episode_steps
        self._truncated_done = truncated_done
        self._step = None
    
    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, dict]:
        self._step = 0
        result = self._env.reset(*args, **kwargs)
        if isinstance(result, tuple): # new gym
            state, info = result
        else: # old gym
            state, info = result, {}
        return state, info
    
    def step(self, action, *args, **kwargs) -> Tuple:
        result = self._env.step(action, *args, **kwargs)
        self._step += 1
        
        if len(result) == 4: # old gym
            next_state, reward, done, info = result
            truncated = False
        elif len(result) == 5: # new gym
            next_state, reward, done, truncated, info = result
        else:
            raise NotImplementedError("[Environment] Unknown environment")
        
        if self._step >= self._episode_max_step:
            truncated = True
        if self._truncated_done:
            done = done or truncated
        
        return (next_state, reward, done, truncated, info)
    
    def get_conf(self) -> AttrDict:
        """environment dimension, space"""
        return get_env_conf(self._env)

    def get_action(self) -> Union[np.ndarray, int]:
        """random action sampling"""
        return self._env.action_space.sample()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self._name})'

    @property
    def action_space(self):
        return self._env.action_space
    @property
    def observation_space(self):
        return self._env.observation_space
    @property
    def n_step(self) -> int:
        return self._step
    

class Task:
    def __init__(self, env, test_env=None):
        self.env = env
        self._test_env = test_env
        self.is_test = False
        self.init()
        
    def init(self) -> None:
        self.data = dict(state=None, score=None, terminal=None, episode=0) # train env data
        self._test_data = dict(state=None, score=None, terminal=None, episode=0) # test env data

    # NOTE: env, data dependent
    def reset(self, *args, **kwargs):
        state, info = self.env.reset(*args, **kwargs)
        
        self.data["state"] = state
        self.data["score"] = 0.0
        self.data["terminal"] = False
        self.data["episode"] += 1
        return state, info
    
    # NOTE: env, data dependent
    def step(self, action, *args, **kwargs):
        # self._n_transition += 1
        next_state, reward, done, truncated, info = self.env.step(action, *args, **kwargs)
        
        self.data["state"] = next_state
        self.data["score"] += reward
        self.data["terminal"] = done or truncated
        return next_state, reward, done, truncated, info
    
    # NOTE: env dependent
    def get_random_action(self, *args, **kwargs):
        return self.env.action_space.sample()
    
    @contextmanager
    def test_mode(self):
        prev_env = self.env
        prev_data = self.data
        prev_mode = self.is_test
        
        self.env = self._test_env
        self.data = self._test_data
        self.is_test = True
        yield
        self.env = prev_env
        self.data = prev_data
        self.is_test = prev_mode
        
    # environment dependent
    @property
    def state(self):
        return self.data["state"]
    @property
    def score(self):
        return self.data["score"]
    @property
    def n_episode(self):
        return self.data["episode"]
    def is_terminal(self):
        return self.data["terminal"]
