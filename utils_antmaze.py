import sys, os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from contextlib import contextmanager
from utils.utils import AttrDict
from utils.base import Base, Module
from utils.net import Stochastic
from utils.buffer import Episode

import warnings # warning ignore
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
stdout, stderr = sys.stdout, sys.stderr
sys.stdout, sys.stderr = open(os.devnull, "w"), open(os.devnull, "w")
try:
    import d4rl
    from d4rl.locomotion.wrappers import NormalizedBoxEnv
    from d4rl.locomotion.ant import AntMazeEnv
except:
    sys.stdout, sys.stderr = stdout, stderr
    print("D4RL import error2")
finally:
    sys.stdout, sys.stderr = stdout, stderr


EMPTY       = 0
WALL        = 1
RESET = R   = "r" # dummy starting point
GOAL = G    = "g"

MAZE_SIZE_10 =[
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

gym.register(
    id = "antmaze-size10-v0",
    entry_point="utils_antmaze:make_ant_maze_env",
    kwargs={
        "maze_map": MAZE_SIZE_10,
        "mode_absolute_obs": False,
        "reward_type": "sparse",
        "non_zero_reset": False,
        # "eval": False,
        "eval": True,
        "maze_size_scaling": 4.0,
        "ref_min_score": 0.0,
        "ref_max_score": 1.0,
        "v2_resets": False,
    }
)

def make_ant_maze_env(*args, **kwargs):
    env = MTAntMazeEnv(*args, **kwargs)
    return NormalizedBoxEnv(env)

def map_to_numpy(map_raw):
    map_np = []
    for row in map_raw:
        row_new = []
        for c in row:
            if c == WALL:
                row_new.append(WALL)
            else:
                row_new.append(EMPTY)
        map_np.append(row_new)
    return np.array(map_np)

def random_empty_point(map_np):
    prob = (1.0 - map_np) / np.sum(1.0 - map_np)
    prob_row = prob.sum(axis=1)
    row = np.random.choice(np.arange(map_np.shape[0]), p=prob_row)
    col = np.random.choice(np.arange(map_np.shape[1]), p=(prob[row] / prob_row[row]))
    return (row, col)


def draw_map(ax, map_raw):
    img = map_to_numpy(map_raw)
    img = np.rot90(img)[::-1, :]
    # image plot
    x, y = np.arange(img.shape[0]), np.arange(img.shape[1])
    X, Y = np.meshgrid(x, y)
    ax.pcolormesh(X, Y, img, cmap="Reds", alpha=0.2)
    # tick & label
    ax.set_xticks(np.arange(0, img.shape[0], 1))
    ax.set_yticks(np.arange(0, img.shape[1], 1))
    ax.set_xticklabels(np.arange(0, img.shape[0], 1))
    ax.set_yticklabels(np.arange(0, img.shape[1], 1))
    # outline
    for i in range(img.shape[0]):
        ax.axhline(i - 0.5, color='black', linewidth=1, alpha=0.2)
    for j in range(img.shape[1]):
        ax.axvline(j - 0.5, color='black', linewidth=1, alpha=0.2)

def draw_trajectory(ax, episodes:Union[list, Episode], block_size:int=4, base_point:np.ndarray=np.array([1, 1]), color="royalblue"):
    if isinstance(episodes, Episode):
        episodes = [episodes]
    block_center = block_size / 2
    base_xy = base_point * block_size
    base_xy = base_xy[::-1]
    
    for episode in episodes:
        rel_xys = episode.export()[0][:, :2]
        rel_last_xy = episode.export()[3][-1:, :2]
        rel_xy = np.concatenate([rel_xys, rel_last_xy], axis=0)
        abs_xy = (base_xy + rel_xy) / block_size
        ax.plot(abs_xy[:, 1], abs_xy[:, 0], color=color, alpha=0.5)

def draw_point(ax, points:np.ndarray, marker="x", color="red", size=100, linewidths=2):
    if points.ndim < 2:
        points = points[None]
    x, y = points[:, 0], points[:, 1]
    ax.scatter(x, y, marker=marker, c=color, s=size, linewidths=linewidths, alpha=1, zorder=10)
    

class MazeTask:
    def __init__(self, start_point:tuple, end_point:tuple,
                 start_point_noise:float=0.1, end_point_noise:float=0.5):
        self.start_point = np.array(start_point, dtype=np.float32)
        self.end_point = np.array(end_point, dtype=np.float32)
        self.start_point_noise = start_point_noise

    def __repr__(self):
        return f'MTMazeTask(start:{self.start_point}+-{self.start_point_noise}, end: {self.end_point})'


class MTAntMazeEnv(AntMazeEnv):
    def __init__(self, maze_map:list, **kwargs):
        self.mode_absolute_obs = False
        self.task = MazeTask([0, 0], [0, 0])
        super().__init__(maze_map=maze_map, **kwargs)
        self.task = None
        
    def rel_to_abs(self, pos:np.ndarray):
        x, y = pos.copy()
        center_pos = self._maze_size_scaling / 2
        offset_x, offset_y = (self._init_torso_x - center_pos), (self._init_torso_y - center_pos)
        new_x, new_y = (x + offset_x, y + offset_y)
        return np.array([new_x, new_y], dtype=np.float32)
    
    def abs_to_rel(self, pos:np.ndarray):
        x, y = pos.copy()
        center_pos = self._maze_size_scaling / 2
        offset_x, offset_y = (self._init_torso_x - center_pos), (self._init_torso_y - center_pos)
        new_x, new_y = (x - offset_x, y - offset_y)
        return np.array([new_x, new_y], dtype=np.float32)
    
    def abs_to_rowcol(self, pos:np.ndarray):
        x, y = pos.copy()
        row, col = x // self._maze_size_scaling, y // self._maze_size_scaling
        row, col = (row + 1, col + 1)
        return np.array([col, row], dtype=np.int32)
    
    def rowcol_to_abs(self, point:np.ndarray):
        row, col = point.copy()
        row, col = self._maze_size_scaling * (row - 1), self._maze_size_scaling * (col - 1)
        center_pos = self._maze_size_scaling / 2
        row, col = (row + center_pos, col + center_pos)
        return np.array([col, row], dtype=np.float32)
        
    
    def get_abs_pos(self):
        rel_pos = self._get_obs()[:2]
        return self.rel_to_abs(rel_pos)
    
    def get_rowcol(self):
        abs_pos = self.get_abs_pos()
        return self.abs_to_rowcol(abs_pos)
        
    def reset_model(self):
        if self._non_zero_reset:
            raise NotImplementedError("non_zero_reset is not implemented")
        if self.task is None:
            raise RuntimeError("task is not set")
        
        
        qpos = self.init_qpos.copy()
        start_pos = self.abs_to_rel(self.rowcol_to_abs(self.task.start_point))
        qpos[:2] = start_pos
        qpos = qpos + self.np_random.uniform(
            size=self.model.nq, low=-self.task.start_point_noise, high=self.task.start_point_noise
        )
        qvel = self.init_qvel.copy() + self.np_random.randn(self.model.nv) * .1
        
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        if self.mode_absolute_obs:
            obs = obs.copy()
            obs[:2] = self.rel_to_abs(obs[:2])
        return obs
    
    def step(self, action):
        if self.task is None:
            raise RuntimeError("task is not set")
        next_obs, reward, done, info = super().step(action)
        if self.mode_absolute_obs:
            next_obs = next_obs.copy()
            next_obs[:2] = self.rel_to_abs(next_obs[:2])
        return next_obs, reward, done, info
    
    @contextmanager
    def set_task(self, task:MazeTask):
        if type(task) != MazeTask:
            raise TypeError(f'task should be MazeTask but {type(task)} is given')
        prev_task = self.task
        target_pos = self.abs_to_rel(self.rowcol_to_abs(task.end_point))
        self.target_goal = target_pos
        self._goal = target_pos
        self.task = task
        yield
        self.task = prev_task



# NOTE: maze_size = 10
size10_offline_points = [
    (( 1,  4), ( 6,  4)), (( 1, 10), ( 7,  3)), (( 9,  8), ( 1,  6)), ((10,  3), ( 1,  8)), (( 9,  3), ( 9,  8)),
    (( 5, 10), ( 1,  5)), (( 9,  4), ( 3,  9)), (( 4,  7), ( 5,  3)), (( 1,  3), ( 8,  6)), (( 4, 10), ( 5,  4)),
    (( 4, 10), ( 5,  5)), (( 8, 10), ( 3,  7)), (( 9,  7), ( 5,  3)), (( 2, 10), ( 1,  2)), (( 5, 10), ( 1,  4)),
    ((10,  1), ( 3,  6)), (( 7,  9), ( 2,  2)), (( 7,  1), ( 9,  5)), (( 6,  4), (10,  7)), (( 7,  7), ( 5,  7)),
    (( 2, 10), (10,  1)), (( 2,  4), ( 9,  7)), ((10,  7), ( 2,  1)), (( 3,  4), ( 8, 10)), (( 1,  9), (10,  8)),
    (( 5,  2), ( 1,  2)), ((10,  5), ( 3,  5)), (( 9,  5), ( 8,  3)), ((10,  5), ( 5,  2)), (( 8,  5), ( 9,  6)),
    (( 9,  8), ( 9,  5)), (( 3, 10), ( 2,  8)), (( 7,  4), (10,  6)), (( 6,  1), (10, 10)), (( 9,  4), (10, 10)),
    (( 4,  5), ( 5,  4)), (( 1, 10), ( 4,  7)), (( 5,  9), ( 8,  3)), (( 7, 10), ( 7,  4)), (( 9,  9), ( 2,  4)),
    (( 7,  2), ( 1,  2)), (( 2,  1), (10,  1)), (( 3,  4), ( 7,  4)), (( 7,  2), (10,  7)), (( 7,  2), ( 2,  4)),
    (( 6,  7), ( 2,  2)), (( 4,  9), ( 2, 10)), (( 2, 10), ( 4,  9)), (( 2,  8), ( 1,  3)), (( 1, 10), ( 4,  1))
]

size10_start_point = (5, 5)
size10_train_points = [
    ( 1,  5), ( 1,  1), ( 5,  2), ( 9,  5), ( 6,  5),
    ( 4, 10), ( 7,  3), ( 9,  8), ( 9,  9), ( 1,  8),
    ( 9,  2), ( 8,  6), ( 1,  2), ( 7,  2), ( 2,  8),
    ( 2,  2), ( 7,  1), ( 3,  8), ( 9,  3), ( 8,  3)
]
size10_test_points = [
    ( 5,  3), ( 1,  6), ( 5,  9), ( 6,  7), (10,  1),
    ( 5,  1), ( 1, 10), ( 7,  9), ( 7,  4), ( 3, 10),
]

size10_tasks = AttrDict(
    offline = [MazeTask(start_point=sp, end_point=ep) for sp, ep in size10_offline_points],
    train = [MazeTask(start_point=size10_start_point, end_point=end_point) for end_point in size10_train_points],
    test    = [MazeTask(start_point=size10_start_point, end_point=end_point) for end_point in size10_test_points],
)

# ----------------------------------------------

class D4RLPolicy(Stochastic, Module):
    @Base.save_input()
    def __init__(self):
        super().__init__(
            dist="tanh", 
            mode_limit_mean="none",
            mode_limit_std="hard", log_std_range=(-20, 2)
        )
        self.fc0 = nn.Linear(in_features=29, out_features=256)
        self.fc1 = nn.Linear(in_features=256, out_features=256)
        self.last_fc = nn.Linear(in_features=256, out_features=8)
        self.last_fc_log_std = nn.Linear(in_features=256, out_features=8)
        
    def forward(self, x):
        h = x
        h = F.relu(self.fc0(h))
        h = F.relu(self.fc1(h))
        mean = self.last_fc(h)
        log_std = self.last_fc_log_std(h)
        return mean, log_std

def _goal_reaching_policy_fn(obs, goal_pos, policy):
    obs_new = obs[2:-2].copy()
    goal_pos = goal_pos / np.linalg.norm(goal_pos) * 5.0
    obs_new = np.concatenate([obs_new, goal_pos], axis=-1)
    with policy.inference_mode():
        action = policy(obs_new, deterministic=False).cpu().numpy()
    return action, (goal_pos[0] + obs[0], goal_pos[1] + obs[1])

def policy_fn(obs, env, policy):
    if not isinstance(env, MTAntMazeEnv):
        env = env.env._wrapped_env
    obs_to_robot = lambda obs: obs[:2]
    
    robot_pos = env.rel_to_abs(obs_to_robot(obs))
    robot_point = env.abs_to_rowcol(robot_pos)
    target_pos = env.rel_to_abs(env.target_goal)
    target_point = env.abs_to_rowcol(target_pos)
    
    way_point = env._get_best_next_rowcol(robot_point, target_point)
    way_point = np.array(way_point)
    if (way_point == target_point).all():
        way_pos = target_pos
    else:
        way_pos = env.rowcol_to_abs(way_point)
        way_pos = way_pos + np.random.uniform(low=-0.5, high=0.5, size=len(way_pos))
    
    goal_pos = way_pos - robot_pos
    return _goal_reaching_policy_fn(obs, goal_pos, policy)

def available_point(env):
    if not isinstance(env, MTAntMazeEnv):
        env = env.env.wrapped_env
    map_np = map_to_numpy(env._maze_map)
    not_wall = map_np == 0
    empty_point = np.where(not_wall)
    empty_point = np.stack(empty_point, axis=0).T
    return empty_point
