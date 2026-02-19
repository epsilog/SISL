import numpy as np
from typing import Union

from utils.buffer import Episode

import warnings # warning ignore
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys, os
stdout, stderr = sys.stdout, sys.stderr
sys.stdout, sys.stderr = open(os.devnull, "w"), open(os.devnull, "w")
try:
    import d4rl
except:
    sys.stdout, sys.stderr = stdout, stderr
    print("D4RL import error")
finally:
    sys.stdout, sys.stderr = stdout, stderr

import gym
gym.register(
    id = "simpl-maze-size20-seed0-v0",
    entry_point="utils_maze:MazeEnv",
    kwargs={
        "size": 20,
        "seed": 0,
        "reward_type": "sparse",
        "done_on_completed": True,
    }
)

from utils.utils import AttrDict
import numpy as np
from scipy.signal import convolve2d
import re

def compute_sampling_probs(maze_layout, filter, temp):
    probs = convolve2d(maze_layout, filter, 'valid')
    return np.exp(-temp*probs) / np.sum(np.exp(-temp*probs))

def sample_2d(probs, rng):
    flat_probs = probs.flatten()
    sample = rng.choice(np.arange(flat_probs.shape[0]), p=flat_probs)
    sampled_2d = np.zeros_like(flat_probs)
    sampled_2d[sample] = 1
    idxs = np.where(sampled_2d.reshape(probs.shape))
    return idxs[0][0], idxs[1][0]

def place_wall(maze_layout, rng, min_len_frac, max_len_frac, temp):
    """Samples wall such that overlap with other walls is minimized (overlap is determined by temperature).
       Also adds one door per wall."""
    size = maze_layout.shape[0]
    sample_vert_hor = 0 if rng.random() < 0.5 else 1
    sample_len = int(max((max_len_frac-min_len_frac) * size * rng.random() + min_len_frac*size, 3))
    sample_door_offset = rng.choice(np.arange(1, sample_len - 1))

    if sample_vert_hor == 0:
        filter = np.ones((sample_len, 5)) / (5*sample_len)
        probs = compute_sampling_probs(maze_layout, filter, temp)
        middle_idxs = sample_2d(probs, rng)
        sample_pos1 = middle_idxs[0]
        sample_pos2 = middle_idxs[1] + 2

        maze_layout[sample_pos1 : sample_pos1 + sample_len, sample_pos2] = 1
        maze_layout[sample_pos1 + sample_door_offset, sample_pos2] = 0
        maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 + 1] = 1
        maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 - 1] = 1
        maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 + 1] = 1
        maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 - 1] = 1
    else:
        filter = np.ones((5, sample_len)) / (5 * sample_len)
        probs = compute_sampling_probs(maze_layout, filter, temp)
        middle_idxs = sample_2d(probs, rng)
        sample_pos1 = middle_idxs[1]
        sample_pos2 = middle_idxs[0] + 2

        maze_layout[sample_pos2, sample_pos1: sample_pos1 + sample_len] = 1
        maze_layout[sample_pos2, sample_pos1 + sample_door_offset] = 0
        maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset - 1] = 1
        maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset - 1] = 1
        maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset + 1] = 1
        maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset + 1] = 1
    return maze_layout

def sample_layout(seed=None,
                  size=20,
                  max_len_frac=0.5,
                  min_len_frac=0.3,
                  coverage_frac=0.25,
                  temp=20):
    """
    Generates maze layout with randomly placed walls.
    :param seed: if not None, makes maze layout reproducible
    :param size: number of cells per side in maze
    :param max_len_frac: maximum length of walls, as fraction of total maze side length
    :param min_len_frac: minimum length of walls, as fraction of total maze side length
    :param coverage_frac: fraction of cells that is covered with walls in randomly generated layout
    :param temp: controls overlap of walls in maze, the higher the temp the less the overlap of walls
    :return: layout matrix (where 1 indicates wall, 0 indicates free space)
    """
    rng = np.random.default_rng(seed=seed)
    maze_layout = np.zeros((size, size))

    while np.mean(maze_layout) < coverage_frac:
        maze_layout = place_wall(maze_layout, rng, min_len_frac, max_len_frac, temp)

    return maze_layout


def layout2str(layout):
    """Transfers a layout matrix to string format that is used by MazeEnv class."""
    h, w = layout.shape
    padded_layout = np.ones((h+2, w+2))
    padded_layout[1:-1, 1:-1] = layout
    output_str = ""
    for row in padded_layout:
        for cell in row:
            output_str += "O" if cell == 0 else "#"
        output_str += "\\"
    output_str = output_str[:-1]    # remove last line break
    output_str = re.sub("O", "G", output_str, count=1)   # add goal at random position
    return output_str

def rand_layout(seed=None, **kwargs):
    """Generates random layout with specified params (see 'sample_layout' function)."""
    rand_layout = sample_layout(seed, **kwargs)
    layout_str = layout2str(rand_layout)
    return layout_str


from contextlib import contextmanager
from d4rl.pointmaze import MazeEnv
import mujoco_py

init_loc_noise = 0.1
complete_threshold = 1.0


class MazeTask:
    def __init__(self, init_loc, goal_loc):
        self.init_loc = np.array(init_loc, dtype=np.float32)
        self.goal_loc = np.array(goal_loc, dtype=np.float32)

    def __repr__(self):
        return f'MTMazeTask(start:{self.init_loc}+-{init_loc_noise}, end: {self.goal_loc})'


class MazeEnv(MazeEnv):
    reward_types = ['sparse', 'dense']
    
    render_width = 100
    render_height = 100
    render_device = -1

    def __init__(self, size, seed, reward_type, done_on_completed):
        if reward_type not in self.reward_types:
            raise f'reward_type should be one of {self.reward_types}, but {reward_type} is given'
        
        self.maze_size = size
        self.maze_spec = rand_layout(size=size, seed=seed)
        
        # for initialization
        self.task = MazeTask([0, 0], [0, 0])
        self.done_on_completed = False
        
        super().__init__(self.maze_spec, reward_type, reset_target=False)
        
        self.task = None
        self.done_on_completed = done_on_completed
        
        gym.utils.EzPickle.__init__(self, size, seed, reward_type, done_on_completed)
        
    @contextmanager
    def set_task(self, task):
        if type(task) != MazeTask:
            raise TypeError(f'task should be MazeTask but {type(task)} is given')

        prev_task = self.task
        self.task = task
        self.set_target(task.goal_loc)
        yield
        self.task = prev_task
        
    def set_render_options(self, width, height, device, fps=30, frame_drop=1):
        self.render_width = width
        self.render_height = height
        self.render_device = device
        self.metadata['video.frames_per_second'] = fps
        self.metadata['video.frame_drop'] = frame_drop

    def reset_model(self):
        if self.task is None:
            raise RuntimeError('task is not set')
        init_loc = self.task.init_loc
        qpos = init_loc + self.np_random.uniform(low=-init_loc_noise, high=init_loc_noise, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):
        if self.task is None:
            raise RuntimeError('task is not set')
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()
        self.do_simulation(action, self.frame_skip)
        self.set_marker()
        
        ob = self._get_obs()
        goal_dist = np.linalg.norm(ob[0:2] - self._target)
        completed = (goal_dist <= complete_threshold)
        done = self.done_on_completed and completed
        
        if self.reward_type == 'sparse':
            reward = float(completed)
        elif self.reward_type == 'dense':
            reward = np.exp(-goal_dist)
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)

        return ob, reward, done, {}

    def render(self, mode='rgb_array'):
        return super().render(mode, self.render_width, self.render_height)
        
    def _get_viewer(self, mode):
        if self._viewers.get(mode) is None and mode in ['rgb_array', 'depth_array']:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=self.render_device)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return super()._get_viewer(mode)
    
    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] += 0.5
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0.5
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 0
            
    

offline_tasks_num = [
    (( 8, 13), ( 5, 12)), ((18, 14), ( 6, 19)), ((14,  9), (15,  1)), ((16,  4), ( 1,  2)), (( 5,  6), (10, 15)),
    ((16, 13), ( 2,  2)), (( 6, 19), (16,  1)), ((14,  9), (14,  5)), (( 2, 14), ( 4,  4)), ((16,  2), ( 9, 10)),
    ((20, 16), ( 2,  6)), (( 4, 10), (13,  7)), ((10,  9), (14, 19)), (( 1,  4), ( 5,  7)), ((16,  7), (20, 15)),
    ((18,  2), ( 5, 11)), ((15,  7), ( 7, 19)), (( 6,  6), (20, 10)), ((13,  1), ( 8,  1)), ((10, 17), (20,  4)),
    ((12, 11), (11, 14)), ((14, 13), ( 8, 14)), (( 1,  9), (12, 14)), ((18,  1), ( 1,  7)), ((15, 11), ( 2,  3)),
    (( 1, 11), (17, 19)), (( 1, 10), (19,  6)), ((17, 19), ( 8, 11)), (( 9,  8), ( 4,  6)), ((10, 11), (15,  9)),
    ((20, 12), (15, 19)), ((19,  2), ( 6,  2)), ((16, 13), (19, 20)), ((11, 20), (18, 16)), ((20,  4), (12, 13)),
    ((11, 10), (19,  5)), ((18,  5), (18, 15)), (( 7, 20), (15, 19)), ((11,  9), ( 4,  8)), (( 5, 19), (12,  6)),
]

# NOTE: x, y
# NOTE: min ~ max = (1, 1) ~ (20, 20)
init_loc = (10, 10)
train_tasks_40_num = [
    ( 8,  2), (19, 19), ( 8,  6), ( 5, 11), (14,  2),
    ( 2,  4), ( 9,  6), (15, 20), ( 6, 14), ( 7, 19),
    (11,  8), ( 9, 19), (11,  6), (19,  5), (14,  6),
    (20,  5), (11, 18), ( 2,  6), (18, 16), (17, 16),
    ( 4,  4), ( 4, 18), (12,  9), (14,  1), (14,  4),
    (12, 19), ( 4,  8), ( 4, 12), (18,  5), (10,  8),
    (12,  5), (20,  2), ( 6, 15), ( 1, 16), (18,  2),
    (19, 12), ( 4, 10), ( 6, 18), ( 3,  2), (15,  9)
]
train_tasks_20_num = [
    (12, 13), ( 8, 20), (14,  7), (18, 15), (14, 12),
    ( 1, 11), (15, 15), (20,  6), (18,  4), ( 6, 14),
    ( 5, 10), ( 6, 15), (14,  9), ( 3, 14), (12, 19),
    (12,  4), ( 2, 11), (19, 13), ( 8, 18), (18,  5),
]
test_tasks_num = [
    (12,  7), ( 1, 19), ( 5, 12), ( 2,  8), ( 5, 19),
    (19, 11), (11, 20), ( 7,  8), (15, 19), (20, 16),
]

tasks = AttrDict(
    offline = [MazeTask(init_loc=il, goal_loc=gl) for il, gl in offline_tasks_num],
    train40=[MazeTask(init_loc=init_loc, goal_loc=task_num) for task_num in train_tasks_40_num],
    train20=[MazeTask(init_loc=init_loc, goal_loc=task_num) for task_num in train_tasks_20_num],
    test=[MazeTask(init_loc=init_loc, goal_loc=task_num) for task_num in test_tasks_num],
)



# ----------------------------------
from d4rl.pointmaze.maze_model import WALL

BLOCK_POINT = [
    [15, 17],
    [13, 17],
    [13, 16],
]
MAZE_SIZE = 20

def available_point(env):
    not_wall = env.maze_arr > WALL
    if BLOCK_POINT is not None:
        for pos in BLOCK_POINT:
            not_wall[pos[0], pos[1]] = False
    empty_point = np.where(not_wall)
    empty_point = np.stack(empty_point, axis=0).T
    return empty_point

def map_to_numpy(map_raw):
    map_np = []
    for row in map_raw:
        row_new = []
        for c in row:
            if c == WALL:
                row_new.append(1)
            else:
                row_new.append(0)
        map_np.append(row_new)
    return np.array(map_np)

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

def draw_point(ax, points:np.ndarray, marker="x", color="red", size=100, linewidths=2):
    if points.ndim < 2:
        points = points[None]
    x, y = points[:, 0], points[:, 1]
    ax.scatter(x, y, marker=marker, c=color, s=size, linewidths=linewidths, alpha=1, zorder=10)

def draw_trajectory(ax, episodes:Union[list, Episode], color="royalblue"):
    if isinstance(episodes, Episode):
        episodes = [episodes]
    
    for episode in episodes:
        xys = episode.export()[0][:, :2]
        last_xy = episode.export()[3][-1:, :2]
        xys = np.concatenate([xys, last_xy], axis=0)
        ax.plot(*xys.T, color=color, alpha=0.5)
