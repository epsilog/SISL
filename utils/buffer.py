import abc
import numpy as np
import torch
import bisect

from typing import Optional, Union, Dict, List, Tuple
from utils.base import SaveMixin
from utils.utils import to_torch, to_numpy, to_list, numpy_split, softmax_logsumexp

NUMPY = np.ndarray
TORCH = torch.Tensor

def longlist2array(longlist):
    return np.fromiter(longlist, np.float32)

class Transition(list):
    """single transition"""
    def __init__(self, *args):
        """e.g. Transition(state, action, reward, nextState, done, ...)"""
        return super().__init__([to_numpy(item) for item in args])

    def cat(self) -> TORCH:
        return np.concatenate(self, axis=-1)

    def __repr__(self):
        string = f'{self.__class__.__name__}'
        if self:
            string += "(\n  " + ",\n  ".join([f'[{i}] '+repr(j) for i, j in enumerate(self)]) + "\n)"
        else:
            string += "()"
        return string
    
class Batch(list):
    def __init__(self, *args, name:Optional[List[str]]=None):
        """
            make batch (e.g. [states, actions, rewards, ...])
            e.g. 
            - Batch(Transition)
            - Batch(Transition, Transition, ...)
            - Batch(states, actions, ...)
        """
        if isinstance(args[0], Transition):
            args = [np.stack(item, axis=0) for item in zip(*args)]
        args = [to_torch(item) if isinstance(item, TORCH) else to_numpy(item) for item in args]
        args = [item.contiguous() if isinstance(item, TORCH) else np.ascontiguousarray(item) for item in args]
        super().__init__(args)
        if name is not None:
            assert len(self) == len(name), "different n_item"
        super().__setattr__("name", name)

    # --- torch ---
    def to(self, device:Union[str, torch.device]):
        return type(self)(*[item.to(device) for item in self], name=self.name)
    
    def cpu(self):
        return self.to("cpu")
    
    def detach(self):
        return type(self)(*[item.detach() for item in self], name=self.name)
    
    def float(self):
        return type(self)(*[item.float() for item in self], name=self.name)
    
    def torch(self):
        return type(self)(*[item if isinstance(item, TORCH) else to_torch(item) for item in self], name=self.name)
    # -------------
    def numpy(self):
        return type(self)(*[item if isinstance(item, NUMPY) else to_numpy(item) for item in self], name=self.name)
    
    def cat(self):
        assert all([self[0].ndim == item.ndim for item in self]), "different dim"
        if isinstance(self[0], TORCH):
            return torch.cat(self, dim=-1)
        else:
            return np.concatenate(self, axis=-1)
        
    def add_item(self, item:Union[NUMPY, TORCH], name:Optional[str]=None):
        assert self.n_size == item.shape[0]
        if isinstance(self[0], NUMPY):
            item = np.ascontiguousarray(to_numpy(item))
        else:
            item = to_torch(item).contiguous()
            item = item.to(self[0].device)
        self.append(item)

        if name is not None:
            assert self.name is not None
            self.name = [*self.name, name]
        else:
            assert self.name is None

    @property
    def n_item(self) -> int:
        """number of items"""
        return super().__len__()
    
    @property
    def n_size(self):
        """number of batch length"""
        return self[0].shape[0]
    
    @property
    def shape(self) -> List:
        return [tuple(item.shape) for item in self]
    
    def __repr__(self):
        return f'{self.__class__.__name__}(n_item={self.n_item}, shape={[tuple(item.shape) for item in self]})'

    def __getattr__(self, key):
        if (self.name is not None) and (key in self.name):
            idx = self.name.index(key)
            return self[idx]
        else:
            return super().__getattribute__(key)
        
    def __setattr__(self, key, value):
        if (self.name is not None) and (key in self.name):
            idx = self.name.index(key)
            self[idx] = value
        else:
            super().__setattr__(key, value)
    # ----------------------------------------------------
    @classmethod
    def merge(cls, *batches):
        assert all([batch.n_item == batches[0].n_item for batch in batches]), "not same n_item"
        assert all([all([batch.shape[i][1:] == batches[0].shape[i][1:] for i in range(batches[0].n_item)]) for batch in batches]), "not same shape(dim)"
        if all(batch.name is not None for batch in batches):
            name = batches[0].name
        else:
            name = None
        
        if all([isinstance(batch[0], TORCH) for batch in batches]):
            return cls(*[torch.cat(item, dim=0) for item in zip(*batches)], name=name)
        else:
            batches = [batch.numpy() for batch in batches]
            return cls(*[np.concatenate(item, axis=0) for item in zip(*batches)], name=name)
    
    @classmethod
    def stack(cls, *batches):
        assert all([batch.n_item == batches[0].n_item for batch in batches]), "not same n_item"
        assert all([all([batch.shape[i] == batches[0].shape[i] for i in range(batches[0].n_item)]) for batch in batches]), "not same shape(dim)"
        if all(batch.name is not None for batch in batches):
            name = batches[0].name
        else:
            name = None
        
        if all([isinstance(batch[0], TORCH) for batch in batches]):
            return cls(*[torch.stack(item, dim=0) for item in zip(*batches)], name=name)
        else:
            batches = [batch.numpy() for batch in batches]
            return cls(*[np.stack(item, axis=0) for item in zip(*batches)], name=name)

class AbstractEpisode(abc.ABC):
    @abc.abstractmethod
    def export(self) -> List[NUMPY]:
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def n_item(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_transition(self) -> int:
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def shape(self) -> List[Tuple]:
        raise NotImplementedError


class Episode(AbstractEpisode, list):
    def __init__(self, *args):
        """
            e.g.
            - Episode().add_transition(Transition)
            - Episode(Transition)
            - Episode(Transition, Transition, ...)
            - Episode(states, actions, ...)
        """
        if args:
            if isinstance(args[0], (NUMPY, TORCH)):
                # e.g. Episode(states, actions, ...)
                items = [to_numpy(item) for item in args]
                assert all(len(items[0]) == len(item) for item in items), "different item lengths"
                transitions = [Transition(*items) for items in zip(*items)]
            elif isinstance(args[0], Transition):
                # e.g. Episode(Transition, Transition, ...)
                transitions = args
            else: raise NotImplementedError
            super().__init__(transitions)
        else:
            # e.g. empty episode: Episode()
            super().__init__()
    
    def add_transition(self, *args) -> None:
        """
            e.g.
            - add_transition(Transition)
            - add_transition(state, action, ...)
        """
        if isinstance(args[0], Transition):
            transition = args[0]
        else:
            transition = Transition(*args)
        
        if self.n_transition > 0:
            assert len(self.shape) == len(transition), "different number of items"
            assert all(self.shape[i][1:] == item.shape for i, item in enumerate(transition)), "different shape of items"
        self.append(transition)

    def export(self) -> List[NUMPY]:
        if self:
            return [np.stack(item, axis=0) for item in zip(*self)]
        else: raise Exception("empty episode")

    def pack(self):
        """Transition list Episode -> tensor StaticEpisode"""
        return StaticEpisode(*self)
    
    def as_batch(self, name:Optional[List[str]]=None):
        return Batch(*self.export(), name=name)
 
    def __repr__(self):
        return f'{self.__class__.__name__}(n_item={len(self[0])}, n_transition={self.n_transition})'
    
    def __getitem__(self, key):
        return super().__getitem__(key)

    @property
    def n_item(self) -> int:
        return len(self[0])

    @property
    def n_transition(self) -> int:
        return len(self)
    
    @property
    def shape(self) -> List[Tuple]:
        n_transition = self.n_transition
        return [(n_transition, *item.shape) for item in self[0]]


class StaticEpisode(AbstractEpisode):
    def __init__(self, *args):
        """
            e.g.
            - StaticEpisode(Transition)
            - StaticEpisode(Transition, Transition, ...)
            - StaticEpisode(states, actions, ...)
        """
        if not args:
            raise Exception("empty episode")
        if isinstance(args[0], Transition): 
            # NOTE: e.g. StaticEpisode(Transitions, Transition, ...)
            memory = [np.stack(item, axis=0) for item in zip(*args)]
        else: 
            # NOTE: e.g. StaticEpisode(states, actions, ...)
            memory = [to_numpy(item) for item in args]
        memory = [np.ascontiguousarray(item) for item in memory]
        self.memory = memory
        
    def as_batch(self, name:Optional[List[str]]=None):
        return Batch(*self.export(), name=name)

    def unpack(self):
        return Episode(*self.export())
    
    def export(self) -> List[NUMPY]:
        return self.memory
    
    def __repr__(self):
        return f'{self.__class__.__name__}(n_item={self.n_item}, n_transition={self.n_transition})'
    
    def __getitem__(self, key) -> List[NUMPY]:
        return [item.__getitem__(key) for item in self.memory]
    
    def __len__(self): return self.n_transition
    
    @property
    def n_item(self) -> int:
        return len(self.shape)

    @property
    def n_transition(self) -> int:
        return self.shape[0][0]
    
    @property
    def shape(self) -> List[Tuple]:
        return [tuple(item.shape) for item in self.memory]

class FlatStaticEpisode(StaticEpisode):
    def __init__(self, *args):
        """
            e.g.
            - FlatStaticEpisode(Transition)
            - FlatStaticEpisode(Transition, Transition, ...)
            - FlatStaticEpisode(states, actions, ...)
        """
        if not args:
            raise Exception("empty episode")
        if isinstance(args[0], Transition): 
            # NOTE: e.g. StaticEpisode(Transitions, Transition, ...)
            memory = [np.stack(item, axis=0) for item in zip(*args)]
        else: 
            # NOTE: e.g. StaticEpisode(states, actions, ...)
            memory = [to_numpy(item) for item in args]
        
        self._shape = [tuple(item.shape) for item in memory]
        self.memory = np.concatenate([item.reshape(item.shape[0], -1) for item in memory], axis=-1)
    
    def export(self) -> List[NUMPY]:
        dims = [np.prod(s[1:]) for s in self.shape]
        return [item.reshape(*self.shape[i]) for i, item in enumerate(numpy_split(self.memory, split_shapes=dims, dim=-1))]
    
    def __getitem__(self, key) -> List[NUMPY]:
        transition_concat = self.memory.__getitem__(key)
        dims = [np.prod(s[1:]) for s in self.shape]
        transition = numpy_split(transition_concat, split_shapes=dims, dim=-1)
        transition = [item.reshape(s[1:]) for item, s in zip(transition, self.shape)]
        return transition

    @property
    def shape(self) -> List[Tuple]:
        return self._shape


class AbstractBuffer(abc.ABC, SaveMixin):
    @abc.abstractmethod
    def get_sample(self) -> Batch:
        raise NotImplementedError

    @abc.abstractmethod
    def set_sample(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def export(self) -> Dict:
        raise NotImplementedError
    
    @classmethod
    @abc.abstractmethod
    def _from(cls, exported_data:dict, **kwargs):
        raise NotImplementedError
        
    @property
    @abc.abstractmethod
    def n_transition(self) -> int:
        """number of transitions"""
        raise NotImplementedError


class TransitionBuffer(AbstractBuffer):
    @SaveMixin.save_input()
    def __init__(self, max_transition:int, name:Optional[List[str]]=None,
                 mode_input_priority:bool=False, mode_output_priority:bool=False,
                 mode_priority_share:bool=False, mode_priority_lifo:bool=False):
        self.max_transition = max_transition
        self.name = name
        self._mode_input_priority = mode_input_priority
        self._mode_output_priority = mode_output_priority
        self._mode_priority_share = mode_priority_share
        self._mode_priority_lifo = mode_priority_lifo
        if mode_priority_share:
            self._mode_input_priority = True
            self._model_output_priority = False
        
        self.memory = []
        self.status = {
            "current_id": 0,
            "id": [],
            "input_priority": [],
            "output_priority": [],
            "priority_rank": [],
        }
        self._cache = {
            "input_priority"    : {"flag": False, "data": None},
            "output_priority"   : {"flag": False, "data": None},
        }
        
    def clear(self) -> None:
        self.memory.clear()
        self.status["current_id"] = 0
        self.status["id"].clear()
        self.status["input_priority"].clear()
        self.status["output_priority"].clear()
        self.status["priority_rank"].clear()
        for k, v in self._cache.items():
            self._cache[k]["flag"] = False
            self._cache[k]["data"] = None
        
    def _create_memory(self, transition:Transition) -> None:
        assert self.memory == [], "memory must be empty"
        for item in transition:
            dtype = to_numpy(item).dtype if not isinstance(item, NUMPY) else item.dtype
            mem = np.empty(shape=(self.max_transition, *item.shape), dtype=dtype)
            self.memory.append(mem)
        
        if self.name is not None:
            assert len(self.name) == len(self.memory), "different n_item"
    
    def _create_info(self, transition:Transition, input_priority:Optional[float]=None, output_priority:Optional[float]=None) -> tuple:
        id = self.status["current_id"] + 1
        input_priority = input_priority if input_priority is not None else id
        return {"id": id, "input_priority": input_priority, "output_priority": output_priority}
    
    def _create_priority_tuple(self, info, index:Optional[int]=None):
        id = info["id"] if not self._mode_priority_lifo else -info["id"]
        return (info["input_priority"], id, index)
        
    def _can_store(self, transition:Transition, info:dict) -> bool:
        if self.n_transition < self.max_transition:
            return True
        else:
            sample_priority_tuple = self._create_priority_tuple(info, index=None)
            min_priority_tuple = self.status["priority_rank"][0]
            return sample_priority_tuple > min_priority_tuple
    
    def _set_sample(self, transition:Transition, info:dict) -> None:
        # NOTE: get index
        if self.n_transition < self.max_transition:
            index = self.n_transition
            new_index = True
        else:
            index = self.status["priority_rank"][0][-1]
            new_index = False
        
        # NOTE: save sample
        assert len(self.memory) == len(transition), "different number of items"
        for i, item in enumerate(transition):
            assert self.memory[i].shape[-1] == item.shape[-1], "different dim of items"
            self.memory[i][index] = item
        # NOTE: save info
        if new_index:
            self.status["id"].append(info["id"])
            self.status["input_priority"].append(info["input_priority"])
            self.status["output_priority"].append(info["output_priority"])
        else:
            self.status["id"][index] = info["id"]
            self.status["input_priority"][index] = info["input_priority"]
            self.status["output_priority"][index] = info["output_priority"]
            del self.status["priority_rank"][0]
        # NOTE: update buffer status
        priority_tuple = self._create_priority_tuple(info, index=index)
        bisect.insort_left(self.status["priority_rank"], priority_tuple)
            
        self.status["current_id"] += 1
        for k, v in self._cache.items():
            self._cache[k]["flag"] = False
    
    def set_sample(self, *args, 
                   priority:Union[float, List[float], None]=None, output_priority:Union[float, List[float], None]=None) -> None:
        """
            e.g. single property: set_sample(state, action, reward, ...)
            e.g. single Transition: set_sample(Transition(...))
            e.g. multiple Transition: set_sample(Transition, Transition, ...)
        """
        transitions = args if isinstance(args[0], Transition) else [Transition(*args)]
        
        if self._mode_input_priority:
            assert priority is not None
            if isinstance(priority, (NUMPY, TORCH)):
                priority = priority.squeeze()
                assert priority.ndim <= 1, "dim error"
            input_priorities = to_list(priority)
            assert len(input_priorities) == len(transitions)
        else:
            assert priority is None
            input_priorities = [None] * len(transitions)
        if self._mode_output_priority:
            assert output_priority is not None, "priority needed"
            if isinstance(output_priority, (NUMPY, TORCH)):
                output_priority = output_priority.squeeze()
                assert output_priority.ndim <= 1, "dim error"
            output_priorities = to_list(output_priority)
            assert len(output_priorities) == len(transitions)
        else:
            assert output_priority is None
            output_priorities = [None] * len(transitions)
        
        if not self.memory:
            self._create_memory(transitions[0])
        
        for transition, input_priority, output_priority in zip(transitions, input_priorities, output_priorities):
            info = self._create_info(transition, input_priority, output_priority)
            if self._can_store(transition, info):
                self._set_sample(transition, info)
    
    def sample_index(self, n_sample:int, method:str="uniform", **kwargs) -> NUMPY:
        assert self.n_transition > 0, "empty buffer"
        if method == "uniform":
            assert self.n_transition >= n_sample
            assert not kwargs, "kwargs does not support"
            if self.n_transition < n_sample:
                raise Exception("buffer n_transition < n_sample")
            else:
                sample_indices = np.random.randint(low=0, high=self.n_transition, size=n_sample)
            
        elif method == "prioritized":
            priorities = self.get_output_priority()
            _array_indices, sample_indices = prioritized_sampling(
                n_sample=n_sample, priorities=priorities, sample_length=1, **kwargs
            )

        else: raise NotImplementedError("unknown sampling method")

        return sample_indices

    def get_sample_by_index(self, indices:NUMPY, name:Optional[List[str]]=None) -> Batch:
        if (name is None) and (self.name is not None):
            name = self.name
        if name is not None:
            item_indices = [self.name.index(n) for n in name]
        else:
            item_indices = [item_idx for item_idx in range(len(self.shape))]

        items = [self.memory[item_idx][indices] for item_idx in item_indices]
        return Batch(*items, name=name) # NUMPY
    
    def get_sample(self, n_sample:int, name:Optional[List[str]]=None, method:str="uniform",
                   mode_info:bool=False, **kwargs) -> Batch:
        indices = self.sample_index(
            n_sample=n_sample, method=method, **kwargs,
        )
        batch = self.get_sample_by_index(indices=indices, name=name)
        
        if mode_info:
            return batch, indices
        else:
            return batch
    
    def get_input_priority(self) -> NUMPY:
        if not self._cache["input_priority"]["flag"]:
            assert self._mode_input_priority and self.n_transition > 0
            input_priority = np.array(self.status["input_priority"])
            self._cache["input_priority"]["data"] = input_priority
            self._cache["input_priority"]["flag"] = True
        else:
            input_priority = self._cache["input_priority"]["data"]
        return input_priority
    
    def get_output_priority(self) -> NUMPY:
        if not self._cache["output_priority"]["flag"]:
            assert self.n_transition > 0
            if self._mode_priority_share and self._mode_input_priority:
                output_priority = np.array(self.status["input_priority"])
            elif self._mode_output_priority:
                output_priority = np.array(self.status["output_priority"])
            else: raise Exception("buffer mode error")
            self._cache["output_priority"]["data"] = output_priority
            self._cache["output_priority"]["flag"] = True
        else:
            output_priority = self._cache["output_priority"]["data"]
        return output_priority
    
    def set_output_priority(self, index:Union[int, List[int]], priority:Union[float, List[float]]):
        assert self._mode_output_priority and self.n_transition > 0
        if isinstance(index, (NUMPY, TORCH)):
            index = index.squeeze()
            assert index.ndim <= 1, "dim error"
        if isinstance(priority, (NUMPY, TORCH)):
            priority = priority.squeeze()
            assert priority.ndim <= 1, "dim error"
        index, priority = to_list(index), to_list(priority)
        for idx, p in zip(index, priority):
            self.status["output_priority"][idx] = p
            
        self._cache["output_priority"]["flag"] = False
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n_item={len(self.memory)}, n_size={self.n_transition}/{self.max_transition})'
    
    @property
    def n_transition(self) -> int:
        return len(self.status["id"])
    
    @property
    def shape(self) -> list:
        n_transition = self.n_transition
        return [(n_transition, *item.shape[1:]) for item in self.memory] if self.memory else None
    
    def export(self) -> dict:
        data = dict()
        data["data"] = self.memory
        data["status"] = self.status
        if self._input_data is not None:
            data["input"] = self._input_data
        return data
    
    @classmethod
    def _from(cls, exported_data:dict, **kwargs):
        model_input = {}
        if "input" in exported_data.keys():
            model_input.update(exported_data["input"])
        model_input.update(kwargs)
        
        buf = cls(**model_input)
        buf.memory = exported_data["data"]
        buf.status = exported_data["status"]
        return buf
    
    def __getitem__(self, key):
        if isinstance(key, int):
            assert key < self.n_transition, "wrong index"
            return Transition(*[self.memory[i][key] for i in range(len(self.memory))])
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop
            step = key.step if key.step is not None else 1
            assert start < self.n_transition and stop < self.n_transition, "wrong index"
            return [Transition(*[self.memory[i][k] for i in range(len(self.memory))]) for k in range(start, stop, step)]

class EpisodeBuffer(AbstractBuffer):
    @SaveMixin.save_input()
    def __init__(self, max_episode:int, name:Optional[List[str]]=None,
                 mode_input_priority:bool=False,
                 mode_output_episode_priority:bool=False, mode_output_transition_priority:bool=False,
                 mode_priority_share:bool=False, mode_priority_lifo:bool=False):
        self.max_episode = max_episode
        self.name = name
        self._mode_input_priority               = mode_input_priority
        self._mode_output_episode_priority      = mode_output_episode_priority
        self._mode_output_transition_priority   = mode_output_transition_priority
        self._mode_priority_share               = mode_priority_share
        self._mode_priority_lifo                = mode_priority_lifo
        
        if mode_priority_share:
            self._mode_input_priority = True
            self._mode_output_episode_priority = False
        
        self.memory = []
        self.status = {
            "current_id": 0,
            # "shape": None,
            "id": [],
            "horizon": [],
            "input_priority": [],
            "output_episode_priority": [],
            "output_transition_priority": [],
            "priority_rank": [],
        }

        self._cache = {
            "horizon"                   : {"flag": False, "data": None},
            "input_priority"            : {"flag": False, "data": None},
            "output_episode_priority"   : {"flag": False, "data": None},
            "output_transition_priority": {"flag": False, "data": None},
        }
        
    def clear(self) -> None:
        self.memory.clear()
        self.status["current_id"] = 0
        self.status["id"].clear()
        self.status["horizon"].clear()
        self.status["input_priority"].clear()
        self.status["output_episode_priority"].clear()
        self.status["output_transition_priority"].clear()
        self.status["priority_rank"].clear()

        for k, v in self._cache.items():
            self._cache[k]["flag"] = False
            self._cache[k]["data"] = None
        
    def _create_memory(self, episode:StaticEpisode) -> None:
        pass
    
    def _create_info(self, episode:StaticEpisode, input_priority:Optional[float]=None,
                     output_episode_priority:Optional[float]=None, output_transition_priority:Optional[List]=None) -> tuple:
        id = self.status["current_id"] + 1
        input_priority = input_priority if input_priority is not None else id
        return {"id": id, "horizon": episode.n_transition, "input_priority": input_priority, 
                "output_episode_priority": output_episode_priority, "output_transition_priority": output_transition_priority}
    
    def _create_priority_tuple(self, info, index:Optional[int]=None):
        id = info["id"] if not self._mode_priority_lifo else -info["id"]
        return (info["input_priority"], id, index)
    
    def _can_store(self, episode:StaticEpisode, info:dict) -> bool:
        if self.n_episode < self.max_episode:
            return True
        else:
            sample_priority_tuple = self._create_priority_tuple(info, index=None)
            min_priority_tuple = self.status["priority_rank"][0]
            return sample_priority_tuple > min_priority_tuple
    
    def _set_sample(self, episode:StaticEpisode, info:dict) -> None:
        # NOTE: get index
        if self.n_episode < self.max_episode:
            index = self.n_episode
            new_index = True
        else:
            index = self.status["priority_rank"][0][-1] # NOTE: lowerst priority index
            new_index = False
        
        # NOTE: save sample & info
        if self.memory:
            base_episode = self.memory[0]
            assert base_episode.n_item == episode.n_item, "different number of items"
            assert all(base_shape[1:] == shape[1:] for base_shape, shape in zip(base_episode.shape, episode.shape)), "different shape of episode"
        if new_index:
            self.memory.append(episode)
            self.status["id"].append(info["id"])
            self.status["horizon"].append(episode.n_transition)
            self.status["input_priority"].append(info["input_priority"])
            self.status["output_episode_priority"].append(info["output_episode_priority"])
            self.status["output_transition_priority"].append(info["output_transition_priority"])
        else:
            self.memory[index] = episode
            self.status["id"][index] = info["id"]
            self.status["horizon"][index] = episode.n_transition
            self.status["input_priority"][index] = info["input_priority"]
            self.status["output_episode_priority"][index] = info["output_episode_priority"]
            self.status["output_transition_priority"][index] = info["output_transition_priority"]
            del self.status["priority_rank"][0]
        
        # NOTE: update buffer status
        priority_tuple = self._create_priority_tuple(info, index=index)
        bisect.insort_left(self.status["priority_rank"], priority_tuple)
        
        self.status["current_id"] += 1
        for k, v in self._cache.items():
            self._cache[k]["flag"] = False

    def _transform_store_format(self, episode):
        return StaticEpisode(*episode.export())
        
    def set_sample(self, *args:List[Union[Episode, StaticEpisode]],
                   priority:Union[float, List[float], None]=None,
                   output_episode_priority:Union[float, List[float], None]=None,
                   output_transition_priority:Union[List[TORCH], List[NUMPY], TORCH, NUMPY, None]=None):
        """
            e.g. single episode: set_sample(Episode(...))
            e.g. multiple episode: set_sample(Episode, Episode, ...)
        """
        assert all(isinstance(ep, (Episode, StaticEpisode)) for ep in args), "input type error"
        episodes = [self._transform_store_format(ep) for ep in args]
        
        # NOTE: priority 입력 확인
        if self._mode_input_priority:
            assert priority is not None
            if isinstance(priority, (NUMPY, TORCH)):
                priority = priority.squeeze()
                assert priority.ndim <= 1, "dim error"
            input_priorities = to_list(priority)
            assert len(input_priorities) == len(episodes)
        else:
            assert priority is None
            input_priorities = [None] * len(episodes)
        if self._mode_output_episode_priority:
            assert output_episode_priority is not None, "episoe_priority needed"
            if isinstance(output_episode_priority, (NUMPY, TORCH)):
                output_episode_priority = output_episode_priority.squeeze()
                assert output_episode_priority.ndim <= 1, "dim error"
            output_episode_priorities = to_list(output_episode_priority)
            assert len(output_episode_priorities) == len(episodes)
        else:
            assert output_episode_priority is None
            output_episode_priorities = [None] * len(episodes)
        if self._mode_output_transition_priority:
            assert output_transition_priority is not None, "transition_priority needed"
            if isinstance(output_transition_priority, (TORCH, NUMPY)):
                output_transition_priority = [output_transition_priority]
            output_transition_priorities = []
            for p in output_transition_priority:
                if isinstance(p, TORCH):
                    p = p.detach().cpu().numpy()
                assert isinstance(p, NUMPY)
                if p.ndim > 1:
                    p = p.squeeze()
                output_transition_priorities.append(p) # (len_episode)
            assert len(output_transition_priorities) == len(episodes)
            assert all(len(t) == ep.n_transition for t, ep in zip(output_transition_priorities, episodes))
        else:
            assert output_transition_priority is None
            output_transition_priorities = [None] * len(episodes)
        
        if not self.memory:
            self._create_memory(episodes[0])
        
        save_flag = True
        for episode, input_priority, output_episode_priority, output_transition_priority in \
            zip(episodes, input_priorities, output_episode_priorities, output_transition_priorities):
                info = self._create_info(episode, input_priority, output_episode_priority, output_transition_priority)
                if self._can_store(episode, info):
                    self._set_sample(episode, info)
                else:
                    save_flag = False
        return save_flag
    
    def sample_index(self, n_sample:int, sample_length:int=1, method:str="uniform",
                     mode_episode:bool=False, **kwargs) -> Union[NUMPY, List[NUMPY]]:
        assert self.n_episode > 0, "empty buffer"
        if method == "uniform":
            assert not kwargs, "kwargs does not support"
            if mode_episode:
                # NOTE: uniform episode sampling
                assert sample_length == 1, "sample_length != 1 does not support"
                episode_indices = np.random.randint(low=0, high=self.n_episode, size=n_sample)
            else:
                # NOTE: uniform transition sampling
                horizons = self.get_horizon()
                episode_indices, transition_indices = uniform_sampling(
                    n_sample=n_sample, array_lengths=horizons, sample_length=sample_length
                )

        elif method == "prioritized":
            if mode_episode:
                # NOTE: prioritized episode sampling
                assert sample_length == 1, "sample_length != 1 does not support"
                priorities = self.get_output_episode_priority() # (n_episode)
                _, episode_indices = prioritized_sampling(
                    n_sample=n_sample, priorities=priorities, **kwargs
                )
            else:
                # NOTE: prioritized transition sampling
                horizons = self.get_horizon()
                priorities = np.concatenate(self.get_output_transition_priority())
                # priorities = self.get_output_transition_priority() # [(n_transition), (n_transition), ...]
                episode_indices, transition_indices = prioritized_sampling(
                    n_sample=n_sample, priorities=priorities, array_lengths=horizons, sample_length=sample_length, **kwargs
                )

        else: raise NotImplementedError("unknown sampling method")
        
        if mode_episode:
            return episode_indices
        else:
            return episode_indices, transition_indices
    
    def get_sample_by_index(self, episode_indices:NUMPY, transition_indices:Optional[NUMPY]=None, 
                            sample_length:int=1, name:Optional[List[str]]=None) -> List:
        if transition_indices is None: 
            # NOTE: Episode sampling
            assert sample_length == 1, "sample_length != 1 does not support"
            assert name is None
            return [self.memory[eid] for eid in episode_indices]
        
        if (name is None) and (self.name is not None):
            name = self.name
        if name is not None:
            item_indices = [self.name.index(n) for n in name]
        else:
            item_indices = [item_idx for item_idx in range(len(self.shape))]

        if sample_length == 1:
            transitions = [[self.memory[eid].memory[item_idx][tid] for item_idx in item_indices] for eid, tid in zip(episode_indices, transition_indices)]
        else:
            transitions = [[self.memory[eid].memory[item_idx][tid:tid+sample_length] for item_idx in item_indices] for eid, tid in zip(episode_indices, transition_indices)]
        items = [np.stack(item, axis=0) for item in zip(*transitions)]
        return Batch(*items, name=name)
    
    def get_sample(self, n_sample:int, sample_length:int=1, name:Optional[List[str]]=None, 
                   method:str="uniform", mode_info:bool=False, **kwargs) -> Batch:
        episode_indices, transition_indices = self.sample_index(
            n_sample=n_sample, sample_length=sample_length, method=method, **kwargs
        )
        batch = self.get_sample_by_index(episode_indices, transition_indices, sample_length=sample_length, name=name)
        
        if mode_info:
            return batch, episode_indices, transition_indices
        else:
            return batch
    
    def get_episode_sample(self, n_episode:int, method:str="uniform",
                           mode_horizon_weight:bool=False, mode_info:bool=False, **kwargs) -> List[StaticEpisode]:
        episode_indices = self.sample_index(
            n_sample=n_episode, method=method, mode_episode=True, **kwargs
        )
        samples = self.get_sample_by_index(
            episode_indices=episode_indices, transition_indices=None
        )

        if mode_info:
            return samples, episode_indices
        else:
            return samples
    
    def get_horizon(self) -> NUMPY:
        if not self._cache["horizon"]["flag"]:
            horizon = np.array(self.status["horizon"])
            self._cache["horizon"]["data"] = horizon
            self._cache["horizon"]["flag"] = True
        else:
            horizon = self._cache["horizon"]["data"]
        return horizon
    
    def get_input_priority(self) -> NUMPY:
        if not self._cache["input_priority"]["flag"]:
            assert self._mode_input_priority and self.n_episode > 0
            input_priority = np.array(self.status["input_priority"])
            self._cache["input_priority"]["data"] = input_priority
            self._cache["input_priority"]["flag"] = True
        else:
            input_priority = self._cache["input_priority"]["data"]
        return input_priority
    
    def get_output_episode_priority(self) -> NUMPY:
        if not self._cache["output_episode_priority"]["flag"]:
            assert self.n_episode > 0
            if self._mode_priority_share and self._mode_input_priority:
                output_episode_priority = np.array(self.status["input_priority"])
            elif self._mode_output_episode_priority:
                output_episode_priority = np.array(self.status["output_episode_priority"])
            else: raise Exception("buffer mode error")
            self._cache["output_episode_priority"]["data"] = output_episode_priority
            self._cache["output_episode_priority"]["flag"] = True
        else:
            output_episode_priority = self._cache["output_episode_priority"]["data"]
        return output_episode_priority
    
    def get_output_transition_priority(self) -> List[NUMPY]:
        if not self._cache["output_transition_priority"]["flag"]:
            assert self._mode_output_transition_priority and self.n_episode > 0
            output_transition_priority = [self.status["output_transition_priority"][i] for i in range(self.n_episode)]
            self._cache["output_transition_priority"]["data"] = output_transition_priority
            self._cache["output_transition_priority"]["flag"] = True
        else:
            output_transition_priority = self._cache["output_transition_priority"]["data"]
        return output_transition_priority
        
    def set_output_episode_priority(self, index:Union[int, List[int]], priority:Union[float, List[float]]):
        assert self._mode_output_episode_priority and self.n_episode > 0
        if isinstance(index, (NUMPY, TORCH)):
            index = index.squeeze()
            assert index.ndim <= 1, "dim error"
        if isinstance(priority, (NUMPY, TORCH)):
            priority = priority.squeeze()
            assert priority.ndim <= 1, "dim error"
        index, priority = to_list(index), to_list(priority)
        for idx, p in zip(index, priority):
            self.status["output_episode_priority"][idx] = p

        self._cache["output_episode_priority"]["flag"] = False
    
    def set_output_transition_priority(
        self, episode_index:Union[int, List, NUMPY, TORCH], transition_index:Union[int, List, NUMPY, TORCH], 
        priority:Union[float, List, NUMPY, TORCH]
    ):
        assert self._mode_output_transition_priority and self.n_episode > 0
        if isinstance(episode_index, (NUMPY, TORCH)):
            episode_index = episode_index.squeeze()
            assert episode_index.ndim <= 1, "dim error"
        episode_index = to_list(episode_index)
        if isinstance(transition_index, (NUMPY, TORCH)):
            transition_index = transition_index.squeeze()
            assert transition_index.ndim <= 1, "dim error"
        transition_index = to_list(transition_index)
        if isinstance(priority, (NUMPY, TORCH)):
            priority = priority.squeeze()
            assert priority.ndim <= 1, "dim error"
        priority = to_list(priority)
        
        assert len(episode_index) == len(transition_index) == len(priority)
        for i in range(len(episode_index)):
            self.status["output_transition_priority"][episode_index[i]][transition_index[i]] = priority[i]

        self._cache["output_transition_priority"]["flag"] = False
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n_item={self.memory[0].n_item if self.memory else 0}, n_episode={self.n_episode}, n_transition={self.n_transition})'

    def _remove_episode_from_rank(self, rank:int) -> None:
        index = self.status["priority_rank"][rank][-1]
        del self.status["id"][index]
        del self.status["horizon"][index]
        del self.status["input_priority"][index]
        del self.status["output_episode_priority"][index]
        del self.status["output_transition_priority"][index]
        del self.status["priority_rank"][rank]
        del self.memory[index]

        for k, v in self._cache.items():
            self._cache[k]["flag"] = False
    
    @property
    def n_episode(self) -> int:
        return len(self.memory)
    
    @property
    def n_transition(self) -> int:
        return sum(self.status["horizon"])
        # return 
    
    @property
    def shape(self) -> list:
        if not self.memory:
            return None
        else:
            ep = self.memory[0]
            return [(None, *s[1:]) for s in ep.shape]
    
    def export(self) -> dict:
        data = dict()
        data["data"] = [ep.export() for ep in self.memory]
        data["status"] = self.status
        if self._input_data is not None:
            data["input"] = self._input_data
        return data
    
    @classmethod
    def _from(cls, exported_data:dict, **kwargs):
        model_input = {}
        if "input" in exported_data.keys():
            model_input.update(exported_data["input"])
        model_input.update(kwargs)
        
        buf = cls(**model_input)
        buf.memory = [StaticEpisode(*list_episode) for list_episode in exported_data["data"]]
        buf.status = exported_data["status"]
        return buf
    
    def __getitem__(self, key):
        return self.memory[key]


class MaxTransitionEpisodeBuffer(EpisodeBuffer):
    @SaveMixin.save_input()
    def __init__(self, max_transition:int, name:Optional[List[str]]=None,
                 mode_input_priority:bool=False, mode_output_episode_priority:bool=False, mode_output_transition_priority:bool=False,
                 mode_priority_share:bool=False, mode_priority_lifo:bool=False):
        self.max_transition = max_transition
        self.name = name
        self._mode_input_priority = mode_input_priority
        self._mode_output_episode_priority = mode_output_episode_priority
        self._mode_output_transition_priority = mode_output_transition_priority
        self._mode_priority_share = mode_priority_share
        self._mode_priority_lifo = mode_priority_lifo
        
        if mode_priority_share:
            self._mode_input_priority = True
            self._mode_output_episode_priority = False
        
        self._flag_new_sample = True
        self.memory = []
        self.status = {
            "current_id": 0,
            "id": [],
            "horizon": [],
            "input_priority": [],
            "output_episode_priority": [],
            "output_transition_priority": [],
            "priority_rank": [],
        }
        self._cache = {
            "horizon"                   : {"flag": False, "data": None},
            "input_priority"            : {"flag": False, "data": None},
            "output_episode_priority"   : {"flag": False, "data": None},
            "output_transition_priority": {"flag": False, "data": None},
        }
    
    def _can_store(self, episode:StaticEpisode, info:tuple) -> bool:
        current_n_transition = self.n_transition
        if current_n_transition + episode.n_transition < self.max_transition:
            return True
        else:
            sample_priority_tuple = self._create_priority_tuple(info, index=None)
            
            loop_cnt = 0
            drop_n_transition = 0
            while True:
                min_priority_tuple = self.status["priority_rank"][loop_cnt]
                if sample_priority_tuple > min_priority_tuple:
                    min_priority_index = min_priority_tuple[-1]
                    drop_n_transition += self.memory[min_priority_index].n_transition
                else:
                    return False
                
                if current_n_transition - drop_n_transition + episode.n_transition < self.max_transition:
                    return True
                loop_cnt += 1
                if loop_cnt > 100:
                    raise Exception("[buffer] infinite loop...")
    
    def _set_sample(self, episode:StaticEpisode, info:dict) -> None:
        while True:
            if self.n_transition + episode.n_transition <= self.max_transition:
                break
            self._remove_episode_from_rank(0)
        index = self.n_episode
        
        # NOTE: save sample & info
        if self.memory:
            base_episode = self.memory[0]
            assert base_episode.n_item == episode.n_item, "different number of items"
            assert all(base_shape[1:] == shape[1:] for base_shape, shape in zip(base_episode.shape, episode.shape)), "different shape of episode"
        self.memory.append(episode)
        self.status["id"].append(info["id"])
        self.status["horizon"].append(episode.n_transition)
        self.status["input_priority"].append(info["input_priority"])
        self.status["output_episode_priority"].append(info["output_episode_priority"])
        self.status["output_transition_priority"].append(info["output_transition_priority"])
        
        # NOTE: update buffer status
        priority_tuple = self._create_priority_tuple(info, index=index)
        bisect.insort_left(self.status["priority_rank"], priority_tuple)
        
        self.status["current_id"] += 1
        for k, v in self._cache.items():
            self._cache[k]["flag"] = False
    
    def _remove_episode_from_rank(self, rank:int) -> None:
        index = self.status["priority_rank"][rank][-1]
        del self.status["id"][index]
        del self.status["horizon"][index]
        del self.status["input_priority"][index]
        del self.status["output_episode_priority"][index]
        del self.status["output_transition_priority"][index]
        del self.memory[index]
        
        for i, p in enumerate(self.status["priority_rank"]):
            if p[-1] > index:
                self.status["priority_rank"][i] = (*p[:-1], p[-1]-1)
        
        del self.status["priority_rank"][rank]
        for k, v in self._cache.items():
            self._cache[k]["flag"] = False


class FlatEpisodeBuffer(EpisodeBuffer):
    def _transform_store_format(self, episode):
        return FlatStaticEpisode(*episode.export())
    
    def get_sample_by_index(self, episode_indices:NUMPY, transition_indices:Optional[NUMPY]=None, 
                            sample_length:int=1, name:Optional[List[str]]=None) -> List:
        if transition_indices is None: 
            # NOTE: Episode sampling
            assert sample_length == 1, "sample_length != 1 does not support"
            assert name is None
            return [self.memory[eid] for eid in episode_indices]
        
        if (name is None) and (self.name is not None):
            name = self.name
        if name is not None:
            item_indices = [self.name.index(n) for n in name]
        else:
            item_indices = [item_idx for item_idx in range(len(self.shape))]

        if sample_length == 1:
            transitions = np.concatenate([self.memory[eid].memory[tid] for eid, tid in zip(episode_indices, transition_indices)]).reshape(len(episode_indices), -1)
            item_shapes = [self.shape[item_idx][1:] for item_idx in item_indices]
        else:
            transitions = np.stack([self.memory[eid].memory[tid:tid+sample_length] for eid, tid in zip(episode_indices, transition_indices)], axis=0)
            item_shapes = [(sample_length, *self.shape[item_idx][1:]) for item_idx in item_indices]
            
        items = numpy_split(transitions, split_shapes=[np.prod(s[1:]) for s in self.shape], dim=-1)
        items = [np.ascontiguousarray(items[item_idx].reshape(-1, *item_shape)) for item_idx, item_shape in zip(item_indices, item_shapes)]
        return Batch(*items, name=name)
    
    @classmethod
    def _from(cls, exported_data:dict, **kwargs):
        model_input = {}
        if "input" in exported_data.keys():
            model_input.update(exported_data["input"])
        model_input.update(kwargs)
        
        buf = cls(**model_input)
        buf.memory = [FlatStaticEpisode(*list_episode) for list_episode in exported_data["data"]]
        buf.status = exported_data["status"]
        return buf

class FlatMaxTransitionEpisodeBuffer(FlatEpisodeBuffer, MaxTransitionEpisodeBuffer):
    pass



def uniform_sampling(n_sample:int, array_lengths:NUMPY, sample_length:int=1):
    valid_array_lengths = (array_lengths - sample_length + 1).clip(min=0)
    total_valid_sample = valid_array_lengths.sum()
    assert total_valid_sample > 0, "no samples available"
    samples = np.random.randint(low=0, high=total_valid_sample, size=n_sample)
    valid_array_lengths_cumsum = valid_array_lengths.cumsum()
    array_indices = np.searchsorted(valid_array_lengths_cumsum - 1, samples)
    offsets = np.concatenate(([0], valid_array_lengths_cumsum[:-1]))
    sample_indices = samples - offsets[array_indices]
    return array_indices, sample_indices

def prioritized_sampling(n_sample:int, priorities:NUMPY, array_lengths:Optional[NUMPY]=None, sample_length:int=1, temperature:float=1.0):
    if array_lengths is None:
        array_lengths = np.array(len(priorities))

    array_lengths_cumsum = array_lengths.cumsum()
    priorities = priorities.copy()
    if sample_length > 1:
        invalid_array_lengths = array_lengths - (array_lengths - sample_length + 1).clip(min=0)
        start_indices = array_lengths_cumsum - invalid_array_lengths
        end_indices = array_lengths_cumsum

        idx = np.arange(len(priorities))
        mask = np.logical_or.reduce(((idx[:, None] >= start_indices) & (idx[:, None] < end_indices)), axis=1)
        priorities[mask] = -np.inf
    probs = softmax_logsumexp(priorities.astype(np.float64), temperature=temperature)
    samples = np.random.multinomial(n_sample, probs)

    samples_cumsum = samples.cumsum()
    n_sample_per_array = np.diff(np.concatenate(([0], samples_cumsum[array_lengths_cumsum - 1])))
    array_indices = np.repeat(np.arange(len(n_sample_per_array)), n_sample_per_array)
    sample_indices_total = np.repeat(np.arange(len(samples)), samples)
    sample_indices = sample_indices_total - np.concatenate(([0], array_lengths_cumsum))[array_indices]
    return array_indices, sample_indices


def multiple_sample_index(
    buffers:List[Union[TransitionBuffer, EpisodeBuffer]], n_sample:int, 
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
            priorities = np.concatenate([buf.get_output_episode_priority() for buf in buffers if buf.n_episode > 0])
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


def get_multiple_sample_by_index(
    buffers:List[Union[TransitionBuffer, EpisodeBuffer]],
    episode_indices:List[NUMPY], transition_indices:Optional[List[NUMPY]]=None,
    sample_length:int=1, name:List[str]=None,
):
    mode_episode = transition_indices is None

    batches = []
    for buffer_idx in range(len(buffers)):
        if isinstance(buffers[buffer_idx], EpisodeBuffer):
            if episode_indices[buffer_idx] is not None:
                batch = buffers[buffer_idx].get_sample_by_index(
                    episode_indices=episode_indices[buffer_idx], 
                    transition_indices=transition_indices[buffer_idx] if not mode_episode else None,
                    sample_length=sample_length, name=name
                )
                if mode_episode:
                    batches.extend(batch)
                else:
                    batches.append(batch)
        else:
            if (not mode_episode) and transition_indices[buffer_idx] is not None:
                batch = buffers[buffer_idx].get_sample_by_index(
                    indices=transition_indices[buffer_idx], name=name
                )
                batches.append(batch)

    if not mode_episode:
        batches = Batch.merge(*batches)
    return batches

def get_multiple_sample(
    buffers:List[Union[TransitionBuffer, EpisodeBuffer]], n_sample:int, name:List[str]=None,
    sample_length:int=1, method:str="uniform", mode_episode:bool=False, 
    mode_info:bool=False, **kwargs
):
    n_buffer_samples, episode_indices, transition_indices = multiple_sample_index(
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


def get_multiple_sample_per_buffer(
    buffers:List[Union[TransitionBuffer, EpisodeBuffer]], 
    n_sample_per_buffer:Union[List[int], int], buffer_indices:Optional[List[int]]=None, 
    name:List[str]=None, sample_length:int=1, method:str="uniform", mode_episode:bool=False,
    mode_info:bool=False, **kwargs
):
    if buffer_indices is None:
        buffer_indices = list(range(len(buffers)))
    if isinstance(n_sample_per_buffer, int):
        n_sample_per_buffer = [n_sample_per_buffer]*len(buffer_indices)
    assert len(n_sample_per_buffer) == len(buffer_indices)

    batches = []
    episode_infos = []
    transition_infos = []
    for buffer_idx, n_sample in zip(buffer_indices, n_sample_per_buffer):
        if n_sample == 0: 
            episode_infos.append(None)
            transition_infos.append(None)
            continue
        if isinstance(buffers[buffer_idx], EpisodeBuffer):
            if mode_episode:
                assert (name is None) and (sample_length == 1)
                batch, episode_indices = buffers[buffer_idx].get_episode_sample(
                    n_episode=n_sample, mode_info=True, **kwargs
                )
                batches.extend(batch)
                episode_infos.append(episode_indices)
                transition_infos.append(None)
            else:
                batch, episode_indices, transition_indices = buffers[buffer_idx].get_sample(
                    n_sample=n_sample, sample_length=sample_length, name=name, 
                    method=method,mode_info=True, **kwargs
                )
                batches.append(batch)
                episode_infos.append(episode_indices)
                transition_infos.append(transition_indices)
            
        else:
            assert (not mode_episode) and (sample_length == 1)
            batch, transition_indices = buffers[buffer_idx].get_sample(
                n_sample=n_sample, name=name, method=method, 
                mode_info=True, **kwargs
            )
            batches.append(batch)
            episode_infos.append(None)
            transition_infos.append(transition_indices)
    
    if mode_info:
        return batches, episode_infos, transition_infos
    else:
        return batches



