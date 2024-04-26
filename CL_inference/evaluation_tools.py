import os, sys
import yaml
from pathlib import Path


def load_config_file_wandb_format(path_to_config, config_file_name):
    config_wandb = yaml.safe_load(Path(os.path.join(path_to_config, config_file_name)).read_text())
    config = {}
    for ii, key in enumerate(config_wandb.keys()):
        try:
            config[key] = config_wandb[key]['value']
        except:
            print("not value for key", key)
    return config