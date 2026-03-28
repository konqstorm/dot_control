import torch
from .agent import Actor, Critic


class SAC:
    def __init__(
        self,
        obs_dim,
        action_dim,
        action_scale,
        hidden_sizes,
        gamma,
        tau,
        actor_lr,
        critic_lr,
        alpha_lr,
        init_alpha,
        target_entropy,
        device=None,
    ):
        self.device = device or torch.device("cpu")
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy

        self.actor = Actor(obs_dim, action_dim, hidden_sizes, action_scale).to(self.device)
        self.critic1 = Critic(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.critic2 = Critic(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.critic1_target = Critic(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.critic2_target = Critic(obs_dim, action_dim, hidden_sizes).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor([init_alpha]).log().to(self.device)
        self.log_alpha.requires_grad = True
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.act(obs_t, deterministic=deterministic)
        return action.cpu().numpy()[0]

    def update(self, batch):
        obs = batch["obs"].to(self.device)
        act = batch["act"].to(self.device)
        rew = batch["rew"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_obs)
            q1_t = self.critic1_target(next_obs, next_action)
            q2_t = self.critic2_target(next_obs, next_action)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target = rew + self.gamma * (1.0 - done) * q_t

        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        critic1_loss = ((q1 - target).pow(2)).mean()
        critic2_loss = ((q2 - target).pow(2)).mean()

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        new_action, logp = self.actor.sample(obs)
        q1_pi = self.critic1(obs, new_action)
        q2_pi = self.critic2(obs, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def _soft_update(self, source, target):
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.copy_(self.tau * src_p.data + (1 - self.tau) * tgt_p.data)

    def save(self, path):
        payload = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
        }
        torch.save(payload, path)

    def load(self, path, map_location=None):
        payload = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.actor.load_state_dict(payload["actor"])
        self.critic1.load_state_dict(payload["critic1"])
        self.critic2.load_state_dict(payload["critic2"])
        self.critic1_target.load_state_dict(payload["critic1_target"])
        self.critic2_target.load_state_dict(payload["critic2_target"])
        self.log_alpha = torch.tensor(payload["log_alpha"], device=self.device)
        self.log_alpha.requires_grad = True
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.alpha_opt.param_groups[0]["lr"])
