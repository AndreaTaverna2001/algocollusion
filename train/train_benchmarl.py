# train/train_benchmarl.py
import os, inspect
from dataclasses import dataclass
from envs.make_env import make_calvano_env
from benchmarl.algorithms.ippo import Ippo, IppoConfig

@dataclass
class TrainCfg:
    # generic knobs; we'll pass only the ones supported by your Ippo.train()
    total_env_steps: int = 200_000    # small smoke test; increase later
    rollout_len: int = 512
    n_envs: int = 1
    batch_size: int = 8192
    minibatch_size: int = 2048
    update_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.95
    seed: int = 0
    logdir: str = "runs/ippo_calvano"

def main():
    cfg = TrainCfg()
    os.makedirs(cfg.logdir, exist_ok=True)
    print(f"Logging to {cfg.logdir}")

    # env factory (PettingZoo)
    def env_fn(seed=None):
        return make_calvano_env(n=2, m=15, k=1, a_i=2.0, c_i=1.0, a0=0.0,
                                mu=0.25, max_steps=cfg.rollout_len, seed=seed)

    # minimal IPPO config (these names match your IppoConfig signature)
    algo_cfg = IppoConfig(
        share_param_critic=False,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        critic_coef=0.5,
        loss_critic_type="smooth_l1",
        lmbda=0.95,
        scale_mapping="affine",
        use_tanh_normal=False,
        minibatch_advantage=True,
    )

    algo = Ippo(env_fn=env_fn, config=algo_cfg)

    # Build kwargs and keep only those Ippo.train actually supports
    desired = dict(
        total_env_steps=cfg.total_env_steps,
        rollout_len=cfg.rollout_len,
        n_envs=cfg.n_envs,
        batch_size=cfg.batch_size,
        minibatch_size=cfg.minibatch_size,
        update_epochs=cfg.update_epochs,
        lr=cfg.lr,
        gamma=cfg.gamma,
        seed=cfg.seed,
        log_dir=cfg.logdir,
    )
    sig = inspect.signature(Ippo.train)
    allowed = {k: v for k, v in desired.items() if k in sig.parameters}

    print("Calling Ippo.train with:", sorted(allowed.keys()))
    algo.train(**allowed)

    # save if supported
    if hasattr(algo, "save"):
        ckpt_dir = os.path.join(cfg.logdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            algo.save(ckpt_dir)
            print("Saved checkpoints to", ckpt_dir)
        except Exception as e:
            print("Save skipped:", e)

if __name__ == "__main__":
    main()


