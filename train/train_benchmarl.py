# train/train_benchmarl.py
# train/train_benchmarl.py
import os, inspect
from dataclasses import dataclass
from envs.make_env import make_calvano_env
from benchmarl.algorithms.ippo import Ippo  # no IppoConfig on your build

@dataclass
class TrainCfg:
    total_env_steps: int = 200_000   # small smoke test; increase later
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

    def env_fn(seed=None):
        return make_calvano_env(n=2, m=15, k=1, a_i=2.0, c_i=1.0, a0=0.0,
                                mu=0.25, max_steps=cfg.rollout_len, seed=seed)

    # Your BenchMARL wants these 9 args POSITIONALLY in Ippo.__init__:
    # (share_param_critic, clip_epsilon, entropy_coef, critic_coef,
    #  loss_critic_type, lmbda, scale_mapping, use_tanh_normal, minibatch_advantage)
    algo = Ippo(
        env_fn,
        False,        # share_param_critic
        0.2,          # clip_epsilon
        0.01,         # entropy_coef
        0.5,          # critic_coef
        "smooth_l1",  # loss_critic_type
        0.95,         # lmbda (GAE)
        "affine",     # scale_mapping
        False,        # use_tanh_normal
    )

    # Build train kwargs, but only pass what this version supports
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
    allowed = {k: v for k, v in desired.items() if k in inspect.signature(Ippo.train).parameters}

    print("Calling Ippo.train with:", sorted(allowed.keys()))
    algo.train(**allowed)

    # save if available
    if hasattr(algo, "save"):
        ckpt = os.path.join(cfg.logdir, "checkpoints")
        os.makedirs(ckpt, exist_ok=True)
        try:
            algo.save(ckpt)
            print("Saved checkpoints to", ckpt)
        except Exception as e:
            print("Save skipped:", e)

if __name__ == "__main__":
    main()



