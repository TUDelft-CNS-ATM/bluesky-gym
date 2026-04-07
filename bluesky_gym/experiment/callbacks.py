"""
bluesky_gym/experiment/callbacks.py
-------------------------------------
SB3 callbacks for experiment monitoring.

Classes
-------
  CheckpointCallback   - periodic model save (overwrites; keeps only latest)
  SuccessRateLogger    - per-group win/loss tracking, printed at training end
  TrainingEvalLogger   - evaluation metrics written to CSV during training

Factories
---------
  callback_registry    - global registry of callback classes, keyed by name
  get_callbacks()      - assembles the full CallbackList for a given config
"""

from __future__ import annotations

import csv
import os

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from bluesky_gym.utils import logger as b_logger

from .config import ExperimentConfig

# ---------------------------------------------------------------------------
# Callback registry
# ---------------------------------------------------------------------------


class CallbackRegistry:
    def __init__(self):
        self._callbacks = {}

    def register(self, name: str):
        """Decorator to register a callback class."""
        def wrapper(cls):
            self._callbacks[name] = cls
            return cls
        return wrapper

    def get(self, name: str):
        callback = self._callbacks.get(name, None)
        if callback is None:
            raise ValueError(f"Callback '{name}' not found in registry. Available callbacks: {self.list_available()}")
        return callback

    def list_available(self):
        return list(self._callbacks.keys())

# Create a global instance
callback_registry = CallbackRegistry()

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

@callback_registry.register("checkpoint")
class CheckpointCallback(BaseCallback):
    """Saves a checkpoint zip at a fixed step interval.

    Always writes to the same filename (checkpoint_model.zip) so only the
    latest checkpoint is kept on disk.
    """

    def __init__(self, save_freq: int, save_path: str) -> None:
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(f"{self.save_path}/checkpoint_model")
        return True
    
    @classmethod
    def from_config(cls, cfg: ExperimentConfig, **kwargs) -> CheckpointCallback:
        """Picks only checkpoint-related fields from the config."""
        return cls(save_freq=cfg.save_freq, save_path=cfg.save_path)


# ---------------------------------------------------------------------------
# Per-group success tracking
# ---------------------------------------------------------------------------

@callback_registry.register("success_rate")
class SuccessRateLogger(BaseCallback):
    """Tracks per-group success rates across all training episodes.

    Reads the group key and 'is_success' from the step info dict.
    The group key checked in order: 'current_group', 'current_runway'.
    Both keys are optional — missing values fall back to 'unknown'.

    A formatted summary is printed at the end of training.
    """

    def __init__(self, success_key: str,group_key: str) -> None:
        super().__init__()
        self.success: str = success_key
        self.group_key = group_key
        # { group_id: {"wins": int, "eps": int} }
        self.group_stats: dict[str, dict[str, int]] = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            group = info.get(self.group_key, "unknown")
            win   = bool(info.get(self.success, False))
            stat  = self.group_stats.setdefault(group, {"wins": 0, "eps": 0})
            stat["eps"] += 1
            if win:
                stat["wins"] += 1
        return True

    def _on_training_end(self) -> None:
        if not self.group_stats:
            return
        print("\n📈 Final Training Success Rates:")
        for grp, stat in sorted(self.group_stats.items()):
            eps  = stat["eps"]
            wins = stat["wins"]
            rate = wins / eps if eps > 0 else 0.0
            print(f"  {grp}: {wins}/{eps}  ({rate:.1%})")

    @property
    def overall_success_rate(self) -> float:
        total_eps  = sum(s["eps"]  for s in self.group_stats.values())
        total_wins = sum(s["wins"] for s in self.group_stats.values())
        return total_wins / total_eps if total_eps > 0 else 0.0
    
    @classmethod
    def from_config(cls, cfg: ExperimentConfig, **kwargs) -> SuccessRateLogger:
        """Picks environment-specific keys for success tracking."""
        return cls(
            success_key=cfg.env.success_key, 
            group_key=cfg.env.group_key or "unknown"
        )


# ---------------------------------------------------------------------------
# Evaluation Classes
# ---------------------------------------------------------------------------

class StandardEval(EvalCallback):
    """Simple wrapper for standard SB3 EvalCallback."""
    @classmethod
    def from_config(cls, cfg: ExperimentConfig, eval_env, **kwargs) -> EvalCallback:
        return cls(
            eval_env=eval_env,
            log_path=cfg.log_dir,
            eval_freq=cfg.session.eval_freq,
            best_model_save_path=cfg.save_path,
            deterministic=True,
            verbose=0,
        )


class TrainingEvalLogger(EvalCallback):
    """Custom logger that writes results to CSV."""
    def __init__(self, eval_env, log_path: str, csv_filename: str, **kwargs) -> None:
        super().__init__(eval_env, log_path=log_path, **kwargs)
        self._csv_path = os.path.join(log_path, csv_filename) if log_path else f"./{csv_filename}"
        self._last_logged_timestep = -1

    @classmethod
    def from_config(cls, cfg: ExperimentConfig, eval_env, **kwargs) -> TrainingEvalLogger:
        return cls(
            eval_env=eval_env,
            log_path=cfg.log_dir,
            eval_freq=cfg.session.eval_freq,
            best_model_save_path=cfg.save_path,
            deterministic=True,
            csv_filename="training_evals.csv",
            verbose=0,
        )

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if not self.evaluations_timesteps:
            return continue_training

        last_eval_step = self.evaluations_timesteps[-1]
        if last_eval_step <= self._last_logged_timestep:
            return continue_training

        self._last_logged_timestep = last_eval_step
        recent_results = self.evaluations_results[-1]
        
        success_rate = (
            np.mean(self.evaluations_successes[-1])
            if self.evaluations_successes
            else float("nan")
        )

        self._append_csv({
            "timestep": last_eval_step,
            "mean_reward": round(float(np.mean(recent_results)), 4),
            "std_reward": round(float(np.std(recent_results)), 4),
            "success_rate": round(float(success_rate), 4),
        })
        return continue_training

    def _append_csv(self, row: dict) -> None:
        write_header = not os.path.exists(self._csv_path)
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header: writer.writeheader()
            writer.writerow(row)


@callback_registry.register("eval")
class Eval:
    """
    Factory class for Evaluation. 
    It doesn't inherit from BaseCallback because it just dispatches 
    to a concrete implementation.
    """
    #NOTE: This is a factory method and I know it's not the cleanest solution but for now it works
    @classmethod
    def from_config(cls, cfg: ExperimentConfig, eval_env, **kwargs) -> EvalCallback:
        if cfg.session.track_training_evals:
            return TrainingEvalLogger.from_config(cfg, eval_env=eval_env)
        return StandardEval.from_config(cfg, eval_env=eval_env)

@callback_registry.register("csv_logger")
class CSVLogger(b_logger.CSVLoggerCallback):
    @classmethod
    def from_config(cls, cfg: ExperimentConfig, **kwargs) -> CSVLogger:
        return cls(cfg.log_dir)

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_callbacks(
    cfg: ExperimentConfig,
    eval_env,
) -> CallbackList:
    """Assemble and return the full callback list for a training run.

    Returns
    -------
    callbacks      : CallbackList       — pass to model.learn()
    """
    callbacks = set(cfg.session.callbacks or [])

    cb_list = [
        callback_registry.get(cb).from_config(cfg, eval_env=eval_env) for cb in callbacks
    ]

    return CallbackList(cb_list)