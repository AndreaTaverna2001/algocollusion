# my_envs/inventory_duopoly_pz.py
from gymnasium import spaces
import numpy as np
from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

# ---------- simple MNL helpers ----------
def _mnl_shares(prices, alpha_i=2.0, alpha0=0.0):
    u = np.array([alpha_i - prices[0], alpha_i - prices[1], alpha0], dtype=float)
    e = np.exp(u - u.max())
    s = e / e.sum()
    return s  # [s0, s1, s_out]

def _integer_sales(prices, inventories, lam=1000, alpha_i=2.0, alpha0=0.0):
    inv = np.array(inventories, dtype=int)
    active = (inv > 0).astype(float)
    s = _mnl_shares(prices, alpha_i=alpha_i, alpha0=alpha0)
    s_firms = s[:2] * active
    s_out = s[2]
    total = s_firms.sum() + s_out
    if total > 0:
        s_firms = s_firms / total
    d = np.floor(lam * s_firms).astype(int)
    d = np.minimum(d, inv)
    return d  # [sales0, sales1]

# ---------- PettingZoo ParallelEnv ----------
class InventoryDuopolyParallelEnv:
    metadata = {"name": "inventory_duopoly_pz_v0"}

    def __init__(
        self,
        T=20,
        I_per_period=440,
        lam=1000,
        c=1.0,
        alpha_i=2.0,
        alpha0=0.0,
        price_grid=None,
        obs_include_time=True,
        obs_include_rival_inventory=True,
        seed=None,
    ):
        self.possible_agents = ["firm_0", "firm_1"]
        self.agents = self.possible_agents[:]
        self.T = int(T)
        self.I0 = int(I_per_period * T)
        self.lam = lam
        self.c = float(c)
        self.alpha_i = alpha_i
        self.alpha0 = alpha0
        self.obs_include_time = obs_include_time
        self.obs_include_rival_inventory = obs_include_rival_inventory
        self.rng = np.random.default_rng(seed)

        if price_grid is None:
            price_grid = np.linspace(1.4, 2.2, 15)
        self.price_grid = np.array(price_grid, dtype=float)

        self.action_spaces = {
            a: spaces.Discrete(len(self.price_grid)) for a in self.possible_agents
        }
        obs_dim = 4 + (1 if obs_include_time else 0)
        self.observation_spaces = {
            a: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.reset(seed=seed)

    # ---- API ----
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.inventories = {a: self.I0 for a in self.possible_agents}
        self.last_prices = {a: 1.6 for a in self.possible_agents}
        self.agents = self.possible_agents[:]
        obs = {a: self._obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        p0 = self.price_grid[int(actions["firm_0"])]
        p1 = self.price_grid[int(actions["firm_1"])]
        prices = np.array([p0, p1], dtype=float)
        inv = np.array([self.inventories["firm_0"], self.inventories["firm_1"]], dtype=int)

        sales = _integer_sales(prices, inv, lam=self.lam, alpha_i=self.alpha_i, alpha0=self.alpha0)
        profits = (prices - self.c) * sales
        r0, r1 = float(profits[0]), float(profits[1])

        self.inventories["firm_0"] -= int(sales[0])
        self.inventories["firm_1"] -= int(sales[1])
        self.last_prices["firm_0"] = p0
        self.last_prices["firm_1"] = p1
        self.t += 1

        terminated = self.t >= self.T
        rewards = {"firm_0": r0, "firm_1": r1}
        terminations = {"firm_0": terminated, "firm_1": terminated}
        truncations = {"firm_0": False, "firm_1": False}
        infos = {"firm_0": {"price": p0, "sales": int(sales[0])},
                 "firm_1": {"price": p1, "sales": int(sales[1])}}
        obs = {a: self._obs(a) for a in self.agents}

        if terminated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def _obs(self, agent):
        me = agent
        other = "firm_1" if agent == "firm_0" else "firm_0"
        feats = [
            self.inventories[me] / max(1, self.I0),
            (self.inventories[other] / max(1, self.I0)) if self.obs_include_rival_inventory else 0.0,
            self.last_prices[me],
            self.last_prices[other],
        ]
        if self.obs_include_time:
            feats.append(self.t / self.T)
        return np.array(feats, dtype=np.float32)

    # PettingZoo helpers
    def observation_space(self, agent): return self.observation_spaces[agent]
    def action_space(self, agent): return self.action_spaces[agent]
    def render(self): pass
    def close(self): pass

# ---------- FACTORIES EXPECTED BY BENCHMARL ----------
def parallel_env(**kwargs):
    """Return the PettingZoo ParallelEnv instance."""
    return InventoryDuopolyParallelEnv(**kwargs)

def env(**kwargs):
    """Return the AEC-wrapped env (for compatibility with some libs)."""
    return OrderEnforcingWrapper(parallel_to_aec(parallel_env(**kwargs)))
