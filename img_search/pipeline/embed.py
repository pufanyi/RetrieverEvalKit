import hydra
from omegaconf import DictConfig

from img_search.utils.logging import print_config, setup_logger


@hydra.main(
    config_path="pkg://img_search/config", version_base=None, config_name="embed_config"
)
def main(cfg: DictConfig):
    setup_logger(cfg.logging)
    print_config(cfg)


if __name__ == "__main__":
    main()
