# envs/make_env.py
from envs.calvano_pricing import CalvanoPricingEnv

def make_calvano_env(
    n=2, m=15, k=1, a_i=2.0, c_i=1.0, a0=0.0, mu=0.25,
    max_steps=2048, seed=None
):
    """Factory function BenchMARL will call to create a fresh PettingZoo env."""
    return CalvanoPricingEnv(
        n=n, m=m, k=k, a_i=a_i, c_i=c_i, a0=a0, mu=mu,
        max_steps=max_steps, seed=seed
    )
