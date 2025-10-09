# train/train_benchmarl.py
# train/train_benchmarl.py
import os
from dataclasses import dataclass
from envs.make_env import make_calvano_env

# BenchMARL 1.5.0 API
from benchmarl.algorithms.ippo import Ippo, IppoConfig

@dataclass
class TrainCfg:
    total_env_steps: int = 2_000_000   # reduce if laptop struggles
    rollout_len: int = 2048
    gamma: float = 0.95
    lr: float = 3e-4
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    clip_range: float = 0.2
    n_envs: int = 1            # start with 1 env on laptop
    batch_size: int = 32768    # tune down if RAM is tight
    minibatch_size: int = 8192
    update_epochs: int = 4
    seed: int = 0
    logdir: str = "runs/ippo_calvano"

def main():
    cfg = TrainCfg()
    os.makedirs(cfg.logdir, exist_ok=True)

    def env_fn(seed=None):
        # rollout_len as episode horizon
        return make_calvano_env(n=2, m=15, k=1, a_i=2.0, c_i=1.0, a0=0.0,
                                mu=0.25, max_steps=cfg.rollout_len, seed=seed)

    algo_cfg = IppoConfig(
        gamma=cfg.gamma,
        lr=cfg.lr,
        entropy_coef=cfg.entropy_coef,
        vf_coef=cfg.vf_coef,
        clip_range=cfg.clip_range,
        rollout_len=cfg.rollout_len,
        batch_size=cfg.batch_size,
        minibatch_size=cfg.minibatch_size,
        update_epochs=cfg.update_epochs,
        share_policy=False,      # independent PPO: one policy per agent
        seed=cfg.seed,
        log_dir=cfg.logdir,
        n_envs=cfg.n_envs,
    )

    algo = Ippo(env_fn=env_fn, config=algo_cfg)
    algo.train(total_env_steps=cfg.total_env_steps)
    algo.save(os.path.join(cfg.logdir, "checkpoints"))
    print("Training finished (BenchMARL Ippo).")

if __name__ == "__main__":
    main()

