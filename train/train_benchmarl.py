# train/train_benchmarl.py
import os
from dataclasses import dataclass

# 1) Your PettingZoo env factory
from envs.make_env import make_calvano_env

# 2) Try the two most common BenchMARL APIs (version safe)
try:
    # recent versions
    from benchmarl.runners.ippo_runner import IPPORunner
except Exception:
    IPPORunner = None

try:
    # older layout
    from benchmarl.runners import ippo as ippo_module
except Exception:
    ippo_module = None

@dataclass
class TrainCfg:
    total_env_steps: int = 5_000_000     # total timesteps across workers
    rollout_len: int = 2048              # steps per update
    gamma: float = 0.95
    lr: float = 3e-4
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    clip_range: float = 0.2
    n_envs: int = 4                      # parallel envs
    batch_size: int = 65536              # adjust per VRAM/CPU
    minibatch_size: int = 16384
    update_epochs: int = 4
    seed: int = 0
    logdir: str = "runs/ippo_calvano"

def main():
    cfg = TrainCfg()
    os.makedirs(cfg.logdir, exist_ok=True)

    # 3) Build an env-creation closure for BenchMARL
    def env_fn(seed=None):
        return make_calvano_env(
            n=2, m=15, k=1, a_i=2.0, c_i=1.0, a0=0.0, mu=0.25,
            max_steps=cfg.rollout_len, seed=seed
        )

    # 4) Run IPPO depending on BenchMARL version
    if IPPORunner is not None:
        runner = IPPORunner(
            env_fn=env_fn,
            n_envs=cfg.n_envs,
            total_env_steps=cfg.total_env_steps,
            rollout_len=cfg.rollout_len,
            gamma=cfg.gamma,
            lr=cfg.lr,
            entropy_coef=cfg.entropy_coef,
            vf_coef=cfg.vf_coef,
            clip_range=cfg.clip_range,
            update_epochs=cfg.update_epochs,
            batch_size=cfg.batch_size,
            minibatch_size=cfg.minibatch_size,
            log_dir=cfg.logdir,
            seed=cfg.seed,
            share_policy=False,   # independent PPO: one policy per agent
        )
        runner.train()
        runner.save(os.path.join(cfg.logdir, "checkpoints"))
        print("Training finished (IPPORunner).")
        return

    if ippo_module is not None:
        # Fallback API example
        algo = ippo_module.IPPO(
            env_fn=env_fn,
            n_envs=cfg.n_envs,
            total_env_steps=cfg.total_env_steps,
            rollout_len=cfg.rollout_len,
            gamma=cfg.gamma,
            lr=cfg.lr,
            entropy_coef=cfg.entropy_coef,
            vf_coef=cfg.vf_coef,
            clip_range=cfg.clip_range,
            update_epochs=cfg.update_epochs,
            batch_size=cfg.batch_size,
            minibatch_size=cfg.minibatch_size,
            log_dir=cfg.logdir,
            seed=cfg.seed,
            share_policy=False,
        )
        algo.train()
        algo.save(os.path.join(cfg.logdir, "checkpoints"))
        print("Training finished (ippo_module).")
        return

    raise ImportError(
        "Could not import BenchMARL IPPO. Check that BenchMARL installed correctly."
    )

if __name__ == "__main__":
    main()
