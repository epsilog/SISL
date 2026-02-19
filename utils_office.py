import roboverse
import roboverse.bullet as bullet
from roboverse.assets.shapenet_object_lists import PICK_PLACE_TRAIN_OBJECTS
from roboverse.policies.pick_place import PickPlaceTarget
from roboverse.policies.drawer_open import DrawerOpen, DrawerClose
from roboverse.envs.widow250_office import Widow250OfficeEnv
import gym
import numpy as np
from contextlib import contextmanager
from typing import List

gym.register(
    id = "office-v0",
    entry_point="utils_office:MTOfficeEnv",
    kwargs={
        # "gui": False,
        "reward_type": "pick_place",
        "control_mode": "discrete_gripper",
        "observation_mode": "noimage",
        "random_shuffle_object": False,
        "random_shuffle_target": False,
        "possible_objects": PICK_PLACE_TRAIN_OBJECTS,
    }
)

object_names = ["eraser", "shed", "pepsi_bottle", "gatorade", "eraser_2", "shed_2", "pepsi_bottle_2"]
object_targets = ["tray", "container", "drawer_inside"]

class OfficeTask:
    def __init__(self, subtask_objects:List[str], subtask_targets:List[str]):
        assert len(subtask_objects) == len(subtask_targets)
        for name in subtask_objects:
            if name not in object_names:
                raise ValueError(f'{name} is not valid object')
        for name in subtask_targets:
            if name not in object_targets:
                raise ValueError(f'{name} is not valid object_target')
        self.subtask_objects = subtask_objects
        self.subtask_targets = subtask_targets
        
    def __repr__(self):
        subtask = [f'({subtask_object}, {subtask_target})' for subtask_object, subtask_target in zip(self.subtask_objects, self.subtask_targets)]
        return f'MTOfficeTask({" -> ".join(subtask)})'

class MTOfficeEnv(Widow250OfficeEnv):
    def __init__(self, **kwargs):
        # dummy task for init
        self.task = OfficeTask(["eraser"], ["tray"])
        super().__init__(**kwargs)
        self.task = None
        
    def get_observation(self):
        gripper_state = self.get_gripper_state()
        gripper_binary_state = [float(self.is_gripper_open)]
        # ee_pos, ee_quat = bullet.get_link_state(self.robot_id, self.end_effector_index)
        ee_pos, ee_quat = bullet.get_link_state(self.p, self.robot_id, self.end_effector_index)
        
        object_states = self.get_states(self.objects) # NOTE: 7 * 7 = 49
        drawer_handle_state = np.asarray(self.get_drawer_handle_pos()) # NOTE: (3)
        arm_state = np.concatenate((ee_pos, ee_quat, gripper_state, gripper_binary_state)) # NOTE: (3, 4, 2, 1)
        desk_object_states = self.get_states(self.desk_objects) # NOTE: (14)
        
        # NOTE: 49 + 3 + 10 + 14 = 76
        state = np.concatenate((object_states, drawer_handle_state, arm_state, desk_object_states))
        return {"state": state}
    
    def reset(self):
        if self.task is None:
            raise ValueError("task is not set")
        items = super().reset()
        return items["state"]
    
    def step(self, action):
        if self.task is None:
            raise ValueError("task is not set")
        items, reward, done, info = super().step(action)
        return items["state"], reward, done, info
    
    @contextmanager
    def set_task(self, task:OfficeTask):
        if type(task) != OfficeTask:
            raise TypeError(f'task should be OfficeTask but {type(task)} is given')
        prev_task = self.task
        self.task = task
        self.task_object_names = task.subtask_objects
        self.object_targets = task.subtask_targets
        yield
        self.task = prev_task

    def _set_observation_space(self):
        if self.observation_mode == 'pixels':
            raise NotImplementedError
        else:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (76,), dtype=np.float32)




class PickPlaceTarget2(PickPlaceTarget):
    def get_action(self):
        ee_pos, ee_orientation = bullet.get_link_state(self.env.p, self.env.robot_id, self.env.end_effector_index)
        ee_deg = bullet.quat_to_deg(self.env.p, ee_orientation)
        
        object_pos, _ = bullet.get_object_position(self.env.p, self.env.objects[self.object_to_target])
        
        object_lifted = object_pos[2] > self.pick_height_thresh_noisy
        
        gripper_pickpoint_dist = np.linalg.norm(self.pick_point - ee_pos)
        
        gripper_droppoint_dist = np.linalg.norm((self.drop_point - ee_pos)[:2])
        gripper_drop_point_dist_z = (self.drop_point - ee_pos)[2]
        origin_dist = self.env.ee_pos_init - ee_pos 

        pickpoint_dist = np.linalg.norm(self.pick_point - ee_pos)
        droppoint_dist = np.linalg.norm(self.drop_point - ee_pos)
        done = False
        noise = True
        noise_thresh = 0.015
        if self.place_attempted:
            # Avoid pick and place the object again after one attempt

            # first lift arm keep xy unchanged
            if np.abs(gripper_drop_point_dist_z) < 0.01:
                # print("lifted")
                action_xyz = [0., 0., gripper_drop_point_dist_z * self.xyz_action_scale] 
                action_angles = (self.drop_angle - ee_deg) * self.angle_action_scale
                action_gripper = [0.0]
            else:
                action_xyz = [0., 0., 0.]
                action_angles = [0., 0., 0.]
                action_gripper = [0.]
                done = True
                self.done = done

        elif gripper_pickpoint_dist > 0.02 and self.env.is_gripper_open:
            
            diff = self.pick_point - ee_pos
            xy_diff = np.linalg.norm(diff[:2])
            if xy_diff > 0.02:
                diff[2] = 0.0
                xyz_scale = self.xyz_action_scale
                angle_scale = self.angle_action_scale
            else:
                xyz_scale = self.xyz_action_scale + 0.5
                angle_scale = self.angle_action_scale + 0.1
            
            action_xyz = diff * xyz_scale
            action_angles = (self.pick_angle - ee_deg)
            action_angles = action_angles * angle_scale
            
            action_gripper = [0.0]
        
        elif self.env.is_gripper_open:
            noise = False
            action_xyz = (self.pick_point  - ee_pos) * self.xyz_action_scale
            action_angles = (self.pick_angle - ee_deg) * self.angle_action_scale
            action_gripper = [-0.9]
        elif not object_lifted:
            xyz_scale = self.xyz_action_scale
            angle_scale = self.angle_action_scale
            
            action_xyz = (self.env.ee_pos_init - ee_pos) * xyz_scale
            action_angles = (self.pick_angle - ee_deg) * angle_scale
            action_gripper = [0.]
        elif gripper_droppoint_dist > 0.02:
            if droppoint_dist < noise_thresh:
                noise = False
            action_xyz = (self.drop_point - ee_pos) * self.xyz_action_scale
            action_angles = (self.drop_angle - ee_deg) * self.angle_action_scale
            action_gripper = [0.]
        else:
            action_xyz = (0., 0., 0.)
            action_angles = [0., 0., 0.]
            action_gripper = [0.9]
            self.place_attempted = True
        
        
        agent_info = dict(place_attempted=self.place_attempted, done=self.done)
        neutral_action = [0.]
        action = np.concatenate(
            (action_xyz, action_angles, action_gripper, neutral_action))
        
        return action, agent_info, noise

class DrawerClose2(DrawerClose):
    def get_action(self):
        ee_pos, ee_orientation = bullet.get_link_state(self.env.p, self.env.robot_id, self.env.end_effector_index)
        ee_deg = bullet.quat_to_deg(self.env.p, ee_orientation)
        handle_pos = self.env.get_drawer_handle_pos() + self.handle_offset
        gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
        gripper_handle_xy_dist = np.linalg.norm(handle_pos[:2] - ee_pos[:2])
        drawer_pos = self.env.get_drawer_pos("drawer")
        drawer_push_target_pos = (
            self.env.get_drawer_handle_pos() + np.array([0.1, 0.0, 0.12]))
        
        is_gripper_ready_to_push = (
            ee_pos[0] > drawer_push_target_pos[0] - 0.05 and
            np.linalg.norm(ee_pos[1] - drawer_push_target_pos[1]) < 0.1 and
            ee_pos[2] < drawer_push_target_pos[2] + 0.05
        )
        done = False
        neutral_action = [0.0]
        noise = False
        if (not self.env.is_drawer_closed() and
                not self.reached_pushing_region and
                not is_gripper_ready_to_push):
            action_xyz = (drawer_push_target_pos  - ee_pos) * self.xyz_action_scale

            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not self.env.is_drawer_closed():
            # print("close top drawer")
            self.reached_pushing_region = True
            action_xyz = [0,0,0]
            
            ratio = 3
            action_xyz[0] = (drawer_pos  - ee_pos)[0] * ratio
            action_xyz[1] = (drawer_pos  - ee_pos)[1] * ratio
            action_xyz[2] = (drawer_pos  - ee_pos)[2] * ratio
            action_angles = [0., 0., 0.]
            action_gripper = [0.7]
            self.begin_closing = True
        if self.env.is_drawer_closed() and self.begin_closing:
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
            done = True
        
        if done:
            if np.linalg.norm(ee_pos - self.env.ee_pos_init) < self.return_origin_thresh:
                self.done = done
            else:
                action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
        
        agent_info = dict(done=self.done)
        action = np.concatenate((action_xyz, action_angles, action_gripper, neutral_action))
        return action, agent_info, noise


# controller
class TableClean:
    def __init__(self, env, pick_height_thresh=-0.31, xyz_action_scale=4.5, angle_action_scale=0.2,
                 pick_point_noise=0.00, drop_point_noise=0.00, return_origin_thresh=0.2, return_origin_thresh_drawer=0.1):
        self.env = env
        self.done = False
        self.pick_height_thresh = pick_height_thresh
        self.xyz_action_scale = xyz_action_scale
        self.angle_action_scale = angle_action_scale
        self.pick_point_noise = pick_point_noise
        self.drop_point_noise = drop_point_noise
        self.return_origin_thresh = return_origin_thresh
        self.return_origin_thresh_drawer = return_origin_thresh_drawer
        self.policies = None
    
    def reset(self):
        self.done = False
        self.object_names = self.env.task_object_names
        self.object_targets = self.env.object_targets
        self.policies = []
        for object_name, object_target in zip(self.object_names, self.object_targets):
            if object_target in ["drawer_inside"]:
                open_policy = DrawerOpen(
                    self.env,
                    xyz_action_scale=self.xyz_action_scale,
                    return_origin_thresh=self.return_origin_thresh,
                )
                self.policies.append(open_policy)
                pick_policy = PickPlaceTarget2(
                    self.env,
                    pick_height_thresh=self.pick_height_thresh, 
                    xyz_action_scale=self.xyz_action_scale,
                    angle_action_scale=self.angle_action_scale,
                    pick_point_noise=self.pick_point_noise, 
                    drop_point_noise=self.drop_point_noise,
                    object_name=object_name,
                    object_target=object_target,
                    return_origin_thresh=self.return_origin_thresh_drawer,
                )
                pick_policy.reset(object_name=object_name, object_target=object_target)
                self.policies.append(pick_policy)
                close_policy = DrawerClose2(
                    self.env,
                    xyz_action_scale=self.xyz_action_scale,
                    return_origin_thresh=self.return_origin_thresh,
                )
                close_policy.reset()
                self.policies.append(close_policy)
            elif object_target in ["tray", "container"]:
                pick_policy = PickPlaceTarget2(
                    self.env,
                    pick_height_thresh=self.pick_height_thresh,
                    xyz_action_scale=self.xyz_action_scale,
                    angle_action_scale=self.angle_action_scale,
                    pick_point_noise=self.pick_point_noise, 
                    drop_point_noise=self.drop_point_noise,
                    object_name=object_name,
                    object_target=object_target,
                    return_origin_thresh=self.return_origin_thresh,
                )
                pick_policy.reset(object_name=object_name, object_target=object_target)
                self.policies.append(pick_policy)
            else: raise NotImplementedError

    def get_action(self):
        for sub_policy in self.policies:
            if sub_policy.done:
                continue
            
            action, agent_info, noise = sub_policy.get_action()
            agent_info["done"] = False
            return action, agent_info, noise
        self.done = True
        raise Exception("policy done")


offline_tasks_num = [ # NOTE: task 25 + 25
    ((0, 1, 3), (0, 1, 2)), ((1, 5, 4), (2, 1, 0)), ((0, 5, 2), (1, 2, 0)), ((1, 4, 3), (1, 2, 0)), ((2, 0, 3), (1, 2, 0)),
    ((2, 6, 4), (1, 2, 0)), ((6, 0, 2), (1, 2, 0)), ((1, 5, 6), (2, 0, 1)), ((2, 4, 0), (0, 1, 2)), ((2, 6, 3), (1, 0, 2)),
    ((2, 6, 4), (0, 1, 2)), ((6, 1, 0), (2, 1, 0)), ((4, 6, 1), (1, 0, 2)), ((3, 2, 0), (2, 1, 0)), ((6, 2, 0), (1, 2, 0)),
    ((3, 2, 4), (2, 0, 1)), ((0, 2, 1), (0, 2, 1)), ((5, 6, 2), (1, 2, 0)), ((4, 2, 3), (0, 2, 1)), ((4, 5, 1), (1, 2, 0)),
    ((5, 0, 6), (0, 1, 2)), ((5, 1, 6), (1, 0, 2)), ((4, 2, 3), (0, 1, 2)), ((3, 2, 1), (1, 0, 2)), ((2, 0, 3), (2, 1, 0)),    
    ((4, 0, 6), (1, 0, 2)), ((0, 4, 3), (0, 1, 2)), ((6, 4, 3), (0, 2, 1)), ((1, 0, 3), (1, 0, 2)), ((2, 5, 4), (2, 1, 0)),
    ((0, 5, 1), (0, 2, 1)), ((5, 4, 2), (1, 2, 0)), ((2, 4, 5), (1, 0, 2)), ((3, 1, 5), (0, 1, 2)), ((6, 0, 4), (1, 2, 0)),
    ((6, 2, 5), (1, 0, 2)), ((2, 6, 4), (2, 1, 0)), ((0, 4, 5), (0, 2, 1)), ((2, 4, 0), (2, 0, 1)), ((3, 6, 5), (1, 0, 2)),
    ((4, 1, 2), (2, 0, 1)), ((3, 5, 1), (2, 0, 1)), ((5, 6, 1), (2, 0, 1)), ((6, 3, 5), (2, 0, 1)), ((4, 1, 0), (1, 2, 0)),
    ((0, 6, 1), (2, 0, 1)), ((3, 5, 0), (2, 0, 1)), ((1, 4, 5), (2, 0, 1)), ((2, 0, 3), (2, 0, 1)), ((5, 3, 6), (0, 2, 1)),
]
train_tasks_num = [
    ((5, 0, 6), (2, 1, 0)), ((5, 0, 2), (1, 2, 0)), ((0, 5, 3), (0, 2, 1)), ((2, 0, 4), (0, 1, 2)), ((1, 5, 6), (0, 2, 1)),
    ((2, 1, 4), (1, 0, 2)), ((3, 4, 0), (0, 1, 2)), ((6, 5, 0), (1, 2, 0)), ((5, 3, 6), (2, 1, 0)), ((0, 6, 1), (1, 2, 0)),
    ((4, 5, 6), (2, 0, 1)), ((6, 5, 1), (1, 2, 0)), ((5, 2, 0), (0, 1, 2)), ((3, 0, 2), (0, 2, 1)), ((0, 1, 3), (0, 2, 1)),
    ((4, 3, 5), (2, 1, 0)), ((5, 6, 1), (0, 2, 1)), ((2, 6, 0), (1, 0, 2)), ((5, 3, 1), (0, 2, 1)), ((3, 2, 6), (0, 1, 2)),
    ((0, 5, 6), (0, 2, 1)), ((0, 3, 5), (0, 2, 1)), ((2, 5, 4), (1, 2, 0)), ((3, 1, 6), (2, 0, 1)), ((4, 2, 0), (1, 2, 0)),
]
test_tasks_num = [
    ((3, 0, 6), (2, 0, 1)), ((0, 4, 2), (2, 1, 0)), ((4, 2, 3), (2, 0, 1)), ((5, 6, 2), (2, 0, 1)), ((5, 3, 0), (1, 0, 2)),
    ((3, 4, 6), (1, 2, 0)), ((3, 1, 0), (0, 1, 2)), ((6, 1, 2), (2, 0, 1)), ((2, 6, 5), (0, 1, 2)), ((3, 2, 4), (2, 1, 0)),
]
from utils.utils import AttrDict
tasks = AttrDict(
    offline = [OfficeTask([object_names[n1] for n1 in obj_nums], [object_targets[n2] for n2 in tgt_nums]) for obj_nums, tgt_nums in offline_tasks_num],
    train   = [OfficeTask([object_names[n1] for n1 in obj_nums], [object_targets[n2] for n2 in tgt_nums]) for obj_nums, tgt_nums in train_tasks_num],
    test    = [OfficeTask([object_names[n1] for n1 in obj_nums], [object_targets[n2] for n2 in tgt_nums]) for obj_nums, tgt_nums in test_tasks_num],
)
