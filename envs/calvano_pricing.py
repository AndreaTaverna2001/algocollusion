from __future__ import annotations
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

class CalvanoPricingEnv(ParallelEnv):
    metadata = {"name": "CalvanoPricing-v0", "is_parallelizable": True}

    def __init__(
        self, n=2, m=15, k=1, a_i=2.0, c_i=1.0, a0=0.0, mu=0.25,
        p_min=None, p_max=None, max_steps=5000, seed=None
    ):
        super().__init__()
        self.n, self.k, self.mu, self.a0, self.max_steps = n, k, float(mu), float(a0), int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.c = np.full(n, c_i, dtype=float) if np.isscalar(c_i) else np.array(c_i, dtype=float)
        self.a = np.full(n, a_i, dtype=float) if np.isscalar(a_i) else np.array(a_i, dtype=float)

        # price grid (simple default bounds; refine later if you like)
        if p_min is None or p_max is None:
            lo = float(np.min(self.c) + 0.2)
            hi = float(np.max(self.c) + 1.5)
        else:
            lo, hi = float(p_min), float(p_max)
        self.grid = np.linspace(lo, hi, m).astype(float)
        self.m = m

        self.agents = [f"firm_{i}" for i in range(n)]
        self.possible_agents = self.agents[:]
        self.action_spaces = {ag: spaces.Discrete(self.m) for ag in self.agents}

        obs_low  = np.array([lo]*n + [0.0], dtype=np.float32)
        obs_high = np.array([hi]*n + [1.0], dtype=np.float32)
        self.observation_spaces = {ag: spaces.Box(obs_low, obs_high, dtype=np.float32) for ag in self.agents}

        self._t = 0
        self._last_prices = None
        self._terminated = False

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self._last_prices = self.rng.choice(self.grid, size=self.n)
        self._terminated = False
        obs = self._obs(start_flag=True)
        infos = {ag: {} for ag in self.agents}
        return obs, infos

    def step(self, actions: dict[str, int]):
        assert not self._terminated, "Call reset() first."
        self._t += 1
        p_now = np.array([self.grid[actions[ag]] for ag in self.agents], dtype=float)
        q = self._logit_shares(p_now)
        profits = (p_now - self.c) * q

        self._last_prices = p_now.copy()
        obs = self._obs(start_flag=False)

        terminated = self._t >= self.max_steps
        truncs = {ag: terminated for ag in self.agents}
        terms = {ag: False for ag in self.agents}
        self._terminated = terminated

        rewards = {ag: float(profits[i]) for i, ag in enumerate(self.agents)}
        infos = {ag: {"prices": p_now.copy(), "quantities": q.copy()} for ag in self.agents}
        return obs, rewards, terms, truncs, infos

    def _obs(self, start_flag: bool):
        vec = np.concatenate([self._last_prices.astype(np.float32),
                              np.array([1.0 if start_flag else 0.0], dtype=np.float32)])
        return {ag: vec.copy() for ag in self.agents}

    def _logit_shares(self, p: np.ndarray) -> np.ndarray:
        z = np.exp((self.a - p) / self.mu)
        denom = np.sum(z) + np.exp(self.a0 / self.mu)
        return z / denom

    def render(self):
        print(f"t={self._t} last_prices={self._last_prices}")
