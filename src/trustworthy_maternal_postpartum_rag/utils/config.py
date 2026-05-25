#merging both yaml files from configs
import yaml
from pathlib import Path

_CONFIG_CACHE = None


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(a, b):
    out = dict(a)

    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v

    return out


def get_config(extra_path=None):
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    project_root = Path(__file__).resolve().parents[3]
    config_dir = project_root / "configs"

    pipeline_path = config_dir / "pipeline_config.yaml"
    experiment_path = config_dir / "experiment_config.yaml"

    config = {}

    if pipeline_path.exists():
        config = _deep_merge(config, _load_yaml(pipeline_path))

    if experiment_path.exists():
        config = _deep_merge(config, _load_yaml(experiment_path))

    if extra_path:
        config = _deep_merge(config, _load_yaml(extra_path))

    _CONFIG_CACHE = config
    return config