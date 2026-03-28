import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
        seed=cfg["run"]["seed"],
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

    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.set_xlim(-env.x_limit * 1.1, env.x_limit * 1.1)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel("Position x")
    ax.set_title("Point stabilization rollout (live)")

    ax.hlines(0, -env.x_limit * 1.05, env.x_limit * 1.05, colors="gray", linewidth=2)
    point = ax.scatter([], [], s=80, color="#1f77b4")
    info_text = ax.text(0.02, 0.85, "", transform=ax.transAxes)
    status_text = ax.text(0.02, 0.70, "", transform=ax.transAxes)

    obs = env.reset()
    episode_idx = 1
    step_idx = 0
    last_status = "running"

    def init():
        point.set_offsets(np.empty((0, 2)))
        info_text.set_text("")
        status_text.set_text("")
        return point, info_text, status_text

    def step_env():
        nonlocal obs, episode_idx, step_idx, last_status
        action = agent.select_action(obs, deterministic=True)
        obs, _, done, info = env.step(action[0])
        step_idx += 1

        if done:
            if info.get("success", False):
                last_status = "success"
            elif info.get("out_of_bounds", False):
                last_status = "out_of_bounds"
            elif info.get("timeout", False):
                last_status = "timeout"
            else:
                last_status = "done"

            obs = env.reset()
            episode_idx += 1
            step_idx = 0
        else:
            last_status = "running"

    def update(_):
        step_env()
        x, v = float(obs[0]), float(obs[1])
        point.set_offsets([[x, 0.0]])
        info_text.set_text(f"ep={episode_idx:3d}  step={step_idx:3d}  x={x: .3f}  v={v: .3f}")
        status_text.set_text(f"status={last_status}")
        return point, info_text, status_text

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        interval=env.dt * 1000,
        blit=False,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
