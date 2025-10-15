import sys
import os

# Add the current project directory to Python's path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import hydra
from omegaconf import DictConfig
from benchmarl.run import hydra_experiment
import my_envs.inventory_duopoly_pz

@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config",
)
def run(cfg: DictConfig) -> None:
    """
    Runs the BenchMARL experiment.
    """
    hydra_experiment(cfg)

if __name__ == "__main__":
    run()