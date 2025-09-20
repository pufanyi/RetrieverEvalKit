import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../../config", version_base=None, config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
