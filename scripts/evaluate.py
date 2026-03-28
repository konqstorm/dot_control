import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.env import PointEnv
from src.sac import SAC


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    env_cfg = cfg["env"]
    reward_cfg = cfg["reward"]
    env = PointEnv(
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
        seed=cfg["train"]["seed"],
    )

    sac_cfg = cfg["sac"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    agent.load(cfg["paths"]["checkpoint"], map_location=device)

    returns = []
    success_steps = []
    steps_to_done = []
    successes = 0

    for _ in range(cfg["eval"]["episodes"]):
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
                    success_steps.append(step + 1)
                break

        returns.append(ep_return)

    avg_return = float(np.mean(returns))
    success_rate = float(successes / cfg["eval"]["episodes"])
    if success_steps:
        avg_steps = float(np.mean(success_steps))
        avg_time = avg_steps * env.dt
    else:
        avg_steps = None
        avg_time = None
    avg_steps_done = float(np.mean(steps_to_done)) if steps_to_done else None
    avg_time_done = avg_steps_done * env.dt if avg_steps_done is not None else None

    metrics = {
        "average_return": avg_return,
        "success_rate": success_rate,
        "average_steps_to_success": avg_steps,
        "average_time_to_success_sec": avg_time,
        "average_steps_to_done": avg_steps_done,
        "average_time_to_done_sec": avg_time_done,
        "episodes": cfg["eval"]["episodes"],
    }

    metrics_path = cfg["paths"]["metrics"]
    ensure_dir(metrics_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
