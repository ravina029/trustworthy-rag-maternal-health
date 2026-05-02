import yaml
from pathlib import Path

_CONFIG_CACHE = {}

def get_config(path):
    path = str(Path(path).resolve())

    if path not in _CONFIG_CACHE:
        with open(path, "r") as f:
            _CONFIG_CACHE[path] = yaml.safe_load(f)

    return _CONFIG_CACHE[path]