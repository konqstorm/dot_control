import argparse
import os
import random
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.env import PointEnv
from src.agent import ReplayBuffer
from src.sac import SAC


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def build_env(cfg, seed):
    env_cfg = cfg["env"]
    reward_cfg = cfg["reward"]
    return PointEnv(
        dt=env_cfg["dt"],
        u_max=env_cfg["u_max"],
        x_init_max=env_cfg["x_init_max"],
        v_init_max=env_cfg["v_init_max"],
        x_limit=env_cfg["x_limit"],
        goal_pos_tol=env_cfg["goal_pos_tol"],
        goal_vel_tol=env_cfg["goal_vel_tol"],
        goal_hold_steps=env_cfg["goal_hold_steps"],
        max_steps=env_cfg["max_steps"],
        reward_alpha=reward_cfg["alpha"],
        reward_beta=reward_cfg["beta"],
        reward_gamma=reward_cfg["gamma"],
        seed=seed,
    )


def evaluate_policy(agent, env, episodes):
    returns = []
    steps_to_done = []
    successes = 0

    for _ in range(episodes):
        obs = env.reset()
        ep_return = 0.0

        for step in range(env.max_steps):
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action[0])
            ep_return += reward
            if done:
                steps_to_done.append(step + 1)
                if info.get("success", False):
                    successes += 1
                break

        returns.append(ep_return)

    success_rate = successes / episodes if episodes > 0 else 0.0
    avg_steps = float(np.mean(steps_to_done)) if steps_to_done else float(env.max_steps)
    avg_return = float(np.mean(returns)) if returns else 0.0
    return success_rate, avg_steps, avg_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    env = build_env(cfg, seed=cfg["train"]["seed"])
    eval_env = build_env(cfg, seed=cfg["train"]["seed"] + 1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sac_cfg = cfg["sac"]
    agent = SAC(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        action_scale=env.u_max,
        hidden_sizes=sac_cfg["hidden_sizes"],
        gamma=sac_cfg["gamma"],
        tau=sac_cfg["tau"],
        actor_lr=sac_cfg["actor_lr"],
        critic_lr=sac_cfg["critic_lr"],
        alpha_lr=sac_cfg["alpha_lr"],
        init_alpha=sac_cfg["init_alpha"],
        target_entropy=sac_cfg["target_entropy"],
        device=device,
    )

    buffer = ReplayBuffer(env.observation_dim, env.action_dim, sac_cfg["replay_size"])

    total_steps = 0
    returns = []
    success_flags = []
    best_success = -1.0
    best_avg_steps = float("inf")
    last_model_path = None
    best_model_path = None

    for ep in range(cfg["train"]["episodes"]):
        obs = env.reset()
        ep_return = 0.0
        ep_len = 0
        ep_success = False

        for _ in range(env.max_steps):
            if total_steps < sac_cfg["start_steps"]:
                action = np.random.uniform(-env.u_max, env.u_max, size=(env.action_dim,))
            else:
                action = agent.select_action(obs, deterministic=False)

            next_obs, reward, done, info = env.step(action[0])
            ep_return += reward 
            ep_len += 1

            scaled_reward = reward * env.dt

            is_success = float(info.get("success", False))

            buffer.store(obs, action, scaled_reward, next_obs, is_success)
            obs = next_obs
            total_steps += 1

            if total_steps >= sac_cfg["update_after"] and buffer.size >= sac_cfg["batch_size"]:
                if total_steps % sac_cfg["update_every"] == 0:
                    for _ in range(sac_cfg["updates_per_step"]):
                        batch = buffer.sample_batch(sac_cfg["batch_size"])
                        agent.update(batch)

            if done:
                ep_success = info.get("success", False)
                break

        returns.append(ep_return)
        success_flags.append(1 if ep_success else 0)

        if (ep + 1) % cfg["train"]["log_every"] == 0:
            eval_episodes = cfg["eval"]["episodes"]
            eval_success, eval_avg_steps, eval_return = evaluate_policy(
                agent, eval_env, eval_episodes
            )

            avg_return = np.mean(returns[-cfg["train"]["log_every"] :])
            print(
                f"Episode {ep + 1:4d} | Return {ep_return:8.2f} | Length {ep_len:3d} | "
                f"AvgReturn {avg_return:8.2f}"
            )
            print(
                f"--> Eval Return: {eval_return:8.2f} | "
                f"Eval SuccessRate: {eval_success:5.2f} | "
                f"Eval AvgSteps: {eval_avg_steps:6.1f}"
            )

            ckpt_dir = os.path.dirname(cfg["paths"]["checkpoint"])
            ensure_dir(cfg["paths"]["checkpoint"])

            sr_str = f"{eval_success:.2f}"
            steps_int = int(round(eval_avg_steps))
            ep_num = ep + 1
            last_name = f"sac_point_last_ep{ep_num}_sr{sr_str}_st{steps_int}.pt"
            last_path = os.path.join(ckpt_dir, last_name)
            if last_model_path and os.path.exists(last_model_path) and last_model_path != last_path:
                os.remove(last_model_path)
            agent.save(last_path)
            last_model_path = last_path

            is_better = False
            if eval_success > best_success:
                is_better = True
            elif eval_success == best_success and eval_avg_steps <= best_avg_steps:
                is_better = True

            if is_better:
                best_success = eval_success
                best_avg_steps = eval_avg_steps
                best_name = f"sac_point_best_ep{ep_num}_sr{sr_str}_st{steps_int}.pt"
                best_path = os.path.join(ckpt_dir, best_name)
                if best_model_path and os.path.exists(best_model_path) and best_model_path != best_path:
                    os.remove(best_model_path)
                agent.save(best_path)
                best_model_path = best_path
                print(f"Saved new best model to {best_path}")
            print(f"Saved last model to {last_path}")
            
    ckpt_path = cfg["paths"]["checkpoint"]
    ensure_dir(ckpt_path)
    agent.save(ckpt_path)

    plt.figure(figsize=(8, 4))
    plt.plot(returns, label="Episode return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    curve_path = cfg["paths"]["curve_plot"]
    ensure_dir(curve_path)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()

    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved learning curve to {curve_path}")


if __name__ == "__main__":
    main()
