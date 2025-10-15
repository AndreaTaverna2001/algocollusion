import hydra
from omegaconf import DictConfig
from benchmarl.run import hydra_experiment

# This is the entry point for the experiment.
# The @hydra.main decorator takes care of loading the configuration.
@hydra.main(
    version_base=None,
    config_path="conf",  # Look for configs in the 'conf' directory
    config_name="config",    # Load 'config.yaml' by default
)
def run(cfg: DictConfig) -> None:
    """
    Runs the BenchMARL experiment.
    """
    hydra_experiment(cfg)

if __name__ == "__main__":
    run()