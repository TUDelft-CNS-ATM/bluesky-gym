"""
bluesky_gym/experiment/train.py
-------------------------------
Logic for initialising environments and executing the SB3 training loop.

Functions
---------
  train()        - Programmatic entry point; instantiates the experiment class.
  run_train_cli() - CLI parser for hyper-parameters and session settings.
"""

from __future__ import annotations
import os
import sys
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_experiment import BaseExperiment, ExperimentConfig

def train(
    experiment_cls: Type[BaseExperiment],
    cfg: "ExperimentConfig"
) -> None:
    """Programmatic entry point to initialise and run an experiment."""
    experiment_cls(cfg).run()

def run_train_cli(experiment_cls: Type[BaseExperiment]) -> None:
    """CLI wrapper that parses arguments and executes training."""
    from .config import ExperimentConfig
    
    model_cls = experiment_cls.model_config_cls
    env_cls   = experiment_cls.env_config_cls

    # Use the existing parser builder from ExperimentConfig
    parser = ExperimentConfig._build_parser(
        model_config_cls=model_cls,
        env_config_cls=env_cls,
        description=f"Train a new model for {experiment_cls.__name__}.",
    )

    # In this context, the 'train' command has already been stripped by runner.py
    args, _ = parser.parse_known_args()

    # 1. Load or Create Config
    run_id = getattr(args, "run_id", None)
    if run_id:
        # Resume/Override existing run
        cfg = ExperimentConfig.load(run_id, model_cls, env_cls)
        cfg = _apply_cli_overrides(cfg, args, model_cls, env_cls)
    else:
        # New run from scratch
        cfg = ExperimentConfig.from_args(args, model_cls, env_cls)

    # 2. Execute
    train(experiment_cls, cfg)

def _apply_cli_overrides(cfg, args, model_cls, env_cls):
    """Applies explicit CLI arguments on top of a loaded configuration."""
    from dataclasses import fields
    from .config import SessionConfig, _field_dest, _MISSING

    # Iterate through session, model, and environment settings to apply overrides
    for section, dc_cls in [("session", SessionConfig), ("model", model_cls), ("env", env_cls)]:
        sub = getattr(cfg, section)
        for f in fields(dc_cls):
            dest = _field_dest(section, f.name)
            val  = getattr(args, dest, _MISSING)
            if val is not _MISSING and val is not None:
                setattr(sub, f.name, val)

    cfg._build_paths()
    return cfg