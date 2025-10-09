# eval/eval_metrics.py
import numpy as np
from envs.make_env import make_calvano_env

def profit_single_step(p, c, a, a0, mu):
    z = np.exp((a - p) / mu)
    denom = np.sum(z) + np.exp(a0 / mu)
    q = z / denom
    return (p - c) * q

def profit_at_prices(p_vec, c, a, a0, mu):
    return profit_single_step(np.array(p_vec), np.array(c), np.array(a), a0, mu)

def compute_delta(pi_learned, pi_B, pi_M):
    return (pi_learned - pi_B) / (pi_M - pi_B)

def example_compute_baselines():
    # For now, use a rough guess for Bertrand/Monopoly prices.
    # Later, replace with numerical solvers (best-response FOCs).
    a = np.array([2.0, 2.0]); c = np.array([1.0, 1.0]); a0 = 0.0; mu = 0.25
    pB = np.array([1.6, 1.6])  # placeholder; refine via solver/grid-search
    pM = np.array([1.9, 1.9])
    pi_B = profit_at_prices(pB, c, a, a0, mu).mean()
    pi_M = profit_at_prices(pM, c, a, a0, mu).mean()
    return pi_B, pi_M
