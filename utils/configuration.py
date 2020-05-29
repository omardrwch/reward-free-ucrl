import importlib
import json
import argparse
from pathlib import Path
import numpy as np


def load_config(path: str) -> dict:
    # Load parameters
    with open(path) as f:
        params = json.load(f)

    # Seed randomness
    np.random.seed(params["seed"])

    # Make environment
    path = params['env_class'].split("'")[1]
    module_name, class_name = path.rsplit(".", 1)
    env_class = getattr(importlib.import_module(module_name), class_name)
    params["env"] = env_class(**params['env_params'])

    # Create output directory
    params["out"] = Path(params["out"]) / class_name
    params["out"].mkdir(parents=True, exist_ok=True)
    return params


def load_config_from_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("parameters", help="Path to the experiment parameters")
    args = parser.parse_args()
    return load_config(args.parameters)
