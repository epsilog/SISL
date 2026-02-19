import numpy as np

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
    id = "simpl-kitchen-v0",
    entry_point="utils_kitchen:KitchenEnv"
)

from utils.utils import AttrDict

from contextlib import contextmanager
from d4rl.kitchen.kitchen_envs import KitchenBase, OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS, BONUS_THRESH
from d4rl.kitchen.adept_envs import mujoco_env

mujoco_env.USE_DM_CONTROL = True

class KitchenTask:
    def __init__(self, subtasks):
        for subtask in subtasks:
            if subtask not in all_tasks:
                raise ValueError(f'{subtask} is not valid subtask')
        self.subtasks = subtasks

    def __repr__(self):
        return f"MTKitchenTask({' -> '.join(self.subtasks)})"


class KitchenEnv(KitchenBase):
    render_width = 400
    render_height = 400
    render_device = -1

    def __init__(self, *args, **kwargs):
        self.TASK_ELEMENTS = ['top burner']  # for initialization
        super().__init__(*args, **kwargs)
        
        self.task = None
        self.TASK_ELEMENTS = None
    
    @contextmanager
    def set_task(self, task):
        if type(task) != KitchenTask:
            raise TypeError(f'task should be KitchenTask but {type(task)} is given')

        prev_task = self.task
        prev_task_elements = self.TASK_ELEMENTS
        self.task = task
        self.TASK_ELEMENTS = task.subtasks
        yield
        self.task = prev_task
        self.TASK_ELEMENTS = prev_task_elements
        
    def set_render_options(self, width, height, device, fps=30, frame_drop=1):
        self.render_width = width
        self.render_height = height
        self.render_device = device
        self.metadata['video.frames_per_second'] = fps
        self.metadata['video.frame_drop'] = frame_drop

    def _get_task_goal(self, task=None):
        if task is None:
            task = ['microwave', 'kettle', 'bottom burner', 'light switch']
        new_goal = np.zeros_like(self.goal)
        for element in task:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal
        return new_goal

    def compute_reward(self, obs_dict):
        reward_dict = {}
        
        next_q_obs = obs_dict['qp']
        next_obj_obs = obs_dict['obj_qp']
        next_goal = self._get_task_goal(task=self.TASK_ELEMENTS)
        idx_offset = len(next_q_obs)
        completions = []
        all_completed_so_far = True
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                next_goal[element_idx])
            complete = distance < BONUS_THRESH
            if complete and all_completed_so_far:
                completions.append(element)
            all_completed_so_far = all_completed_so_far and complete
        for completion in completions:
            self.tasks_to_complete.remove(completion)
        reward = float(len(completions))
        return reward
    
    def reset_model(self):
        ret = super().reset_model()
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        return ret

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        if not self.initializing:
            a = self.act_mid + a * self.act_amp

        self.robot.step(self, a, step_duration=self.skip * self.model.opt.timestep)

        obs = self._get_obs()
        reward = self.compute_reward(self.obs_dict)
        done = not self.tasks_to_complete
        env_info = {
            'time': self.obs_dict['t'],
            'obs_dict': self.obs_dict,
        }
        return obs, reward, done, env_info


all_tasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
offline_tasks_num = [
    (5, 1, 2, 4), (5, 1, 0, 2), (5, 2, 4, 3), (5, 6, 2, 4), (5, 0, 2, 3),
    (6, 1, 0, 2), (6, 1, 0, 4), (6, 0, 2, 4), (1, 0, 2, 3), (1, 0, 4, 3),
    (5, 1, 0, 3), (5, 1, 0, 4), (5, 6, 1, 2), (5, 6, 1, 4), (5, 6, 2, 3),
    (5, 6, 4, 3), (5, 6, 0, 3), (5, 6, 0, 4), (5, 0, 2, 3), (5, 0, 4, 3),
    (6, 1, 2, 3), (6, 1, 0, 3), (6, 2, 4, 3), (6, 0, 2, 3), (6, 0, 4, 3),
]
train_tasks_num = [
    (5, 6, 0, 3), (5, 0, 1, 3), (5, 1, 2, 4), (6, 0, 2, 4), (5, 0, 4, 1),
    (6, 1, 2, 3), (5, 6, 3, 0), (6, 2, 3, 0), (5, 6, 0, 1), (5, 6, 3, 4),
    (5, 0, 3, 1), (6, 0, 2, 1), (5, 6, 1, 2), (5, 6, 2, 4), (5, 0, 2, 3),
    (6, 0, 1, 2), (5, 2, 3, 4), (5, 0, 1, 4), (6, 0, 3, 4), (0, 1, 3, 2),
    (5, 6, 2, 3), (6, 0, 1, 4), (0, 1, 2, 3), (1, 4, 5, 3), (0, 4, 2, 6),
]
test_tasks_num = [
    (5, 0, 2, 1), (5, 0, 1, 2), (6, 0, 2, 3), (5, 6, 1, 4), (6, 0, 3, 1),
    (6, 2, 3, 4), (6, 0, 1, 3), (5, 0, 3, 4), (0, 1, 3, 4), (5, 6, 0, 4),
]
tasks = AttrDict(
    offline_tasks=[KitchenTask([all_tasks[c] for c in offline_task]) for offline_task in offline_tasks_num],
    train_tasks=[KitchenTask([all_tasks[c] for c in train_task]) for train_task in train_tasks_num],
    test_tasks=[KitchenTask([all_tasks[c] for c in test_task]) for test_task in test_tasks_num]
)