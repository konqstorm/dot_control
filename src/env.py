import numpy as np


class PointEnv:
    def __init__(
        self,
        dt=0.05,
        u_max=5.0,
        x_init_max=10.0,
        v_init_max=5.0,
        x_limit=20.0,
        goal_pos_tol=0.05,
        goal_vel_tol=0.05,
        goal_hold_steps=10,
        max_steps=500,
        reward_alpha=1.0,
        reward_beta=0.1,
        reward_gamma=0.001,
        seed=None,
    ):
        self.dt = float(dt)
        self.u_max = float(u_max)
        self.x_init_max = float(x_init_max)
        self.v_init_max = float(v_init_max)
        self.x_limit = float(x_limit)
        self.goal_pos_tol = float(goal_pos_tol)
        self.goal_vel_tol = float(goal_vel_tol)
        self.goal_hold_steps = int(goal_hold_steps)
        self.max_steps = int(max_steps)
        self.reward_alpha = float(reward_alpha)
        self.reward_beta = float(reward_beta)
        self.reward_gamma = float(reward_gamma)

        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.consecutive_goal = 0
        x = self.rng.uniform(-self.x_init_max, self.x_init_max)
        v = self.rng.uniform(-self.v_init_max, self.v_init_max)
        self.state = np.array([x, v], dtype=np.float32)
        return self.state.copy()

    def step(self, action):
        u = float(np.clip(action, -self.u_max, self.u_max))
        x, v = float(self.state[0]), float(self.state[1])

        reward = -(
            self.reward_alpha * x * x
            + self.reward_beta * v * v
            + self.reward_gamma * u * u
        )

        x_next = x + v * self.dt
        v_next = v + u * self.dt
        self.state = np.array([x_next, v_next], dtype=np.float32)
        self.t += 1

        done = False
        info = {"success": False, "timeout": False, "out_of_bounds": False}

        if abs(x_next) < self.goal_pos_tol and abs(v_next) < self.goal_vel_tol:
            self.consecutive_goal += 1
        else:
            self.consecutive_goal = 0

        if self.consecutive_goal >= self.goal_hold_steps:
            done = True
            info["success"] = True

        if abs(x_next) > self.x_limit:
            done = True
            info["out_of_bounds"] = True

        if self.t >= self.max_steps:
            done = True
            info["timeout"] = True

        return self.state.copy(), float(reward), done, info

    @property
    def observation_dim(self):
        return 2

    @property
    def action_dim(self):
        return 1
