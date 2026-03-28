import numpy as np
import torch
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.max_size = size
        self.ptr = 0
        self.size = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes, action_scale):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_sizes[-1], hidden_sizes[:-1])
        self.mean_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.action_scale = float(action_scale)

    def forward(self, obs):
        x = self.backbone(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        scaled_action = action * self.action_scale
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return scaled_action, log_prob

    def act(self, obs, deterministic=False):
        mean, log_std = self(obs)
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)
        return action * self.action_scale


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super().__init__()
        self.net = MLP(obs_dim + action_dim, 1, hidden_sizes)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)
