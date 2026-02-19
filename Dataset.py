import os
import argparse
import torch
import numpy as np

from utils.utils import AttrDict
from utils.base import Environment
from utils.buffer import Episode, FlatMaxTransitionEpisodeBuffer
from utils.dataset import make_dataset


def rollout_kitchen(
    env, noise, low_policy, high_policy, skill_length
):
    with low_policy.inference_mode(), high_policy.inference_mode():
        ep = Episode()
        state, _info = env.reset()
        while True:
            if env.n_step % skill_length == 0:
                z = high_policy(state, deterministic=False)
            action = low_policy(state, z).cpu().numpy()
            if noise > 0:
                action = action + np.random.randn(len(action)) * noise
                action = action.clip(min=-1, max=1)
            next_state, reward, done, truncated, _info = env.step(action)
            done = done or truncated
            ep.add_transition(state, action, reward, next_state, done)
            state = next_state
            if done: break
    return ep

def rollout_office(
    env, noise, policy
):
    while True:
        ep = Episode()
        state, _info = env.reset()
        policy.reset()
        d_action = env.action_space.shape[0]
        
        is_error = False
        while True:
            try:
                action, agent_info, add_noise = policy.get_action()
            except:
                print(f'error occured')
                is_error = True
                break
            if d_action - action.shape[0] == 1:
                action = np.append(action, 0)
            if noise > 0:
                action = action + np.random.randn(len(action)) * noise
            action = action.clip(min=-1, max=1)
            
            next_state, reward, done, truncated, _info = env.step(action)
            done = done or truncated
            ep.add_transition(state, action, reward, next_state, done)
            state = next_state
            if done: break
            
        if not is_error: break
    return ep

def rollout_maze(
    env, noise, controller, task
):
    ep = Episode()
    state, _info = env.reset()
    while True:
        action, _done = controller.get_action(state[:2], state[2:], task.goal_loc)
        if noise > 0:
            action = action + np.random.randn(len(action)) * noise
            action = action.clip(min=-1, max=1)
        next_state, reward, done, truncated, _info = env.step(action)
        done = done or truncated
        ep.add_transition(state, action, reward, next_state, done)
        state = next_state
        if done: break
    return ep

def rollout_antmaze(
    env, noise, policy, policy_fn
):
    ep = Episode()
    state, _info = env.reset()
    while True:
        action, goal = policy_fn(state, env._env, policy)
        if noise > 0:
            action = action + np.random.randn(len(action)) * noise
            action = action.clip(min=-1, max=1)
        next_state, reward, done, truncated, _info = env.step(action)
        done = done or truncated or (reward == 1)
        ep.add_transition(state[:-2], action, reward, next_state[:-2], done)
        state = next_state
        if done: break
    return ep


def main(args):
    # make folder to save dataset
    save_dir = f"./environments/{args.env}/dataset"
    os.makedirs(save_dir, exist_ok=True)
    
    conf = AttrDict(
        ENV = args.env,
        DEVICE = int(args.device) if args.device.isnumeric() else args.device,
        NOISE = args.noise,
        MAX_TRANSITION = args.max_transition,
    )
    if conf.ENV == "kitchen":
        env_conf = AttrDict(
            env_id              = "simpl-kitchen-v0",
            episode_max_step    = 280,
            expert_noise        = 0.05,
            skill_length        = 10,
            low_path            = "./environments/kitchen/controller/low/low-policy.pt",
            high_paths          = [f'./environments/kitchen/controller/high/' + p for p in sorted(os.listdir("./environments/kitchen/controller/high"))],
        )
        low_policy = torch.load(env_conf.low_path).eval().to(conf.DEVICE)
        high_policys = [torch.load(high_path).eval().to(conf.DEVICE) for high_path in env_conf.high_paths]
        
        from utils_kitchen import tasks
        offline_tasks = tasks.offline_tasks
    elif conf.ENV == "office":
        env_conf = AttrDict(
            env_id = "office-v0",
            episode_max_step = 300,
            expert_noise = 0.05,
        )
        from utils_office import tasks, TableClean
        offline_tasks = tasks.offline
    elif conf.ENV == "maze":
        env_conf = AttrDict(
            env_id              = "simpl-maze-size20-seed0-v0", 
            episode_max_step    = 2000,
            expert_noise        = 0.1,
        )
        from d4rl.pointmaze import waypoint_controller
        from utils_maze import tasks
        offline_tasks = tasks.offline
    elif conf.ENV == "antmaze":
        env_conf = AttrDict(
            env_id              = "antmaze-size10-v0",
            episode_max_step    = 1000,
            expert_noise        = 0.1,
        )
        from utils_antmaze import policy_fn, size10_tasks, D4RLPolicy
        offline_tasks = size10_tasks.offline
        policy = D4RLPolicy.load("./environments/antmaze/controller/antmaze_d4rl_policy.pt").eval().to(conf.DEVICE)
    else: raise NotImplementedError
        
    max_transition_per_task = conf.MAX_TRANSITION // len(offline_tasks)
    print(f'--- Collect max {max_transition_per_task} transition per {len(offline_tasks)} tasks...')
    if conf.ENV == "antmaze":
        env = Environment(env=env_conf.env_id, episode_max_step=env_conf.episode_max_step, truncated_done=True, eval=False)
    else:
        env = Environment(env=env_conf.env_id, episode_max_step=env_conf.episode_max_step, truncated_done=True)
    buffers = []
    noise = conf.NOISE if conf.NOISE != 0 else env_conf.expert_noise
    
    for task_idx in range(len(offline_tasks)):
        task = offline_tasks[task_idx]
        buffer = FlatMaxTransitionEpisodeBuffer(max_transition=conf.MAX_TRANSITION, name=["state", "action", "reward", "next_state", "done"])
        
        with env._env.set_task(task):
            while True:
                if conf.ENV == "kitchen":
                    high_policy = high_policys[task_idx]
                    ep = rollout_kitchen(env, noise, low_policy, high_policy, env_conf.skill_length)
                elif conf.ENV == "office":
                    policy = TableClean(env._env)
                    ep = rollout_office(env, noise, policy)
                elif conf.ENV == "maze":
                    controller = waypoint_controller.WaypointController(env._env.str_maze_spec)
                    ep = rollout_maze(env, noise, controller, task)
                elif conf.ENV == "antmaze":
                    ep = rollout_antmaze(env, noise, policy, policy_fn)
                else: raise NotImplementedError
                
                if buffer.n_transition + ep.n_transition > max_transition_per_task:
                    break
                else:
                    buffer.set_sample(ep)
        
        buffers.append(buffer)
        n_episode = buffer.n_episode
        n_transition = buffer.n_transition
        mean_horizon = np.array([ep.n_transition for ep in buffer]).mean()
        mean_score = np.array([ep.export()[2].sum() for ep in buffer]).mean()
        print(f'task_idx: {task_idx}, n_episode: {n_episode}, n_transition: {n_transition}, mean_horizon: {mean_horizon:.2f}, mean_score: {mean_score:.2f}')
    
    total_n_episode = sum([buf.n_episode for buf in buffers])
    total_n_transition = sum([buf.n_transition for buf in buffers])
    total_mean_horizon = total_n_transition / total_n_episode
    total_mean_score = sum([sum([ep.export()[2].sum() for ep in buf]) for buf in buffers]) / total_n_episode
    print(f'[total] n_episode: {total_n_episode}, n_transition: {total_n_transition}, mean_horizon: {total_mean_horizon:.2f}, mean_score: {total_mean_score:.2f}')
    
    dataset = make_dataset(buffers)
    save_path = f'{save_dir}/Dataset-Env={conf.ENV}_Noise={conf.NOISE}_T={conf.MAX_TRANSITION}.pt'
    torch.save(dataset, save_path)
    print(f'--- Save dataset to {save_path} ---')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["kitchen", "office", "maze", "antmaze"], required=True)
    parser.add_argument("--noise", required=True, type=float)
    parser.add_argument("--max_transition", required=True, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()
    main(args)
    print("--- Done!!! ---")