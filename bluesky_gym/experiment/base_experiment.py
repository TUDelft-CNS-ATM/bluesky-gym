"""
bluesky_gym/experiment/base_experiment.py
------------------------------------------
Abstract experiment base class.

Classes
-------
  BaseExperiment   - lifecycle (train / evaluate / run) + abstract interface

Subclasses must implement
-------------------------
  make_env(env_kwargs, render_mode)  - build a (wrapped) Gymnasium env
  make_model(env)                    - build an SB3 model

Subclasses may override
-----------------------
  model_config_cls   - dataclass for model hyper-parameters (default: ModelConfig)
  env_config_cls     - EnvConfig subclass for this experiment
  metric_extractor() - returns a MetricExtractor or None
  get_callbacks()    - returns a CallbackList
  evaluate()         - post-training evaluation loop
  train_log_interval() - how often SB3 logs to stdout
"""

from __future__ import annotations

import abc
from typing import Optional, Type, TYPE_CHECKING

import gymnasium as gym

from .config import ExperimentConfig, ModelConfig, EnvConfig
from .callbacks import get_callbacks

if TYPE_CHECKING:
    from .evaluate import MetricExtractor


class BaseExperiment(abc.ABC):
    """Abstract base for all bluesky-gym experiments.
 
    Class variables
    ---------------
    model_config_cls : Type[ModelConfig]
        Dataclass describing the algorithm and its hyper-parameters.
        CLI flags are generated automatically from its fields.
        Must set algorithm to a non-None default in the subclass.
 
    env_config_cls : Type[EnvConfig]
        Dataclass describing the environment.  Wraps an EnvKwargsConfig
        (forwarded to gym.make()) and holds env_name, group_key,
        success_key.  CLI flags for both EnvConfig and the nested
        EnvKwargsConfig are generated automatically.
        Must set env_name (and ideally group_key) in the subclass.
    """

    model_config_cls: Type[ModelConfig]     = ModelConfig
    env_config_cls:   Type[EnvConfig]   = EnvConfig

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # Abstract interface — must be implemented by every subclass          #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def make_env(
        self,
        env_kwargs:  dict,
        render_mode: Optional[str] = None,
    ) -> gym.Env:
        """Construct and return a single (optionally wrapped) Gymnasium env."""
        ...

    @abc.abstractmethod
    def make_model(self, env: gym.Env):
        """Construct and return a model bound to `env`."""
        ...

    # ------------------------------------------------------------------ #
    # Extension points — override as needed                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def metric_extractor(cls) -> "MetricExtractor | None":
        """Return a MetricExtractor for env-specific metrics, or None.

        The extractor tells evaluate.py which extra fields to pull from the
        info dict and how to aggregate them.  Returning None means only the
        built-in metrics (is_success, total_reward) are collected.
        """
        return None

    def get_callbacks(self, eval_env: gym.Env):
        """Return a CallbackList for model.learn().

        Delegates to callbacks.get_callbacks() by default.  Override to add
        experiment-specific callbacks.
        """
        return get_callbacks(self.cfg, eval_env)

    def evaluate(self, model) -> dict[str, list[bool]]:
        """Run eval episodes and print per-group success rates.
 
        Uses cfg.env.success_key and cfg.env.group_key to read results
        from the info dict, so no env-specific strings are hardcoded here.
        """
        cfg         = self.cfg
        success_key = cfg.env.success_key
        group_key   = cfg.env.group_key
 
        print(f"\n📊 Evaluating model from {cfg.save_path}/final_model.zip …")
 
        eval_env = self.make_env(cfg.eval_env_kwargs, render_mode="human")
        model.set_env(eval_env)
 
        results: dict[str, list[bool]] = {}
 
        for _ in range(cfg.session.eval_episodes):
            done = truncated = False
            obs, info = eval_env.reset()
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = eval_env.step(action)
 
            group = str(info.get(group_key, "unknown")) if group_key else "all"
            win   = bool(info.get(success_key, False))
            results.setdefault(group, []).append(win)
 
        eval_env.close()
 
        print("\nEvaluation Results:")
        for grp, outcomes in sorted(results.items()):
            n    = len(outcomes)
            wins = sum(outcomes)
            print(f"  {grp}: {wins}/{n}  ({wins/n:.1%})")
 
        return results

    def train_log_interval(self, total_timesteps: int) -> int:
        """
        Default Logging cadence passed to model.learn(log_interval=...).  
        Subclasses may override.
        """
        return max(1_000, total_timesteps // 100)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        """Build envs → model → learn → save."""
        cfg          = self.cfg
        log_interval = self.train_log_interval(cfg.session.total_timesteps)
 
        train_env = self.make_env(cfg.train_env_kwargs)
        eval_env  = self.make_env(cfg.eval_env_kwargs)

        if cfg.session.pretrained_model_path:
            print(f"📥 Loading pretrained model from {cfg.session.pretrained_model_path}...")
            model = cfg.model.get_algorithm().load(cfg.session.pretrained_model_path, env=train_env)

            if model is None:
                raise ValueError("Model load failed!")
            
            # Compatibility Checks
            if model.observation_space != train_env.observation_space:
                raise ValueError("Pretrained model observation space does not match the environment!")
            if model.action_space != train_env.action_space:
                raise ValueError("Pretrained model action space does not match the environment!")
        else:
            model = self.make_model(train_env)
            if model is None:
                raise ValueError("make_model() returned None!")

        callbacks = self.get_callbacks(eval_env)

        cfg.save()
 
        print(f"\n🏋️  Training for {cfg.session.total_timesteps:,} steps …")
        model.learn(
            total_timesteps=cfg.session.total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
        )
 
        model.save(f"{cfg.save_path}/final_model")
        print(f"✅ Training complete.  Model saved → {cfg.save_path}")
 
        train_env.close()
        eval_env.close()
        self._model = model

    def run(self) -> None:
        """Convenience entry point: train (optional) → evaluate (optional)."""
        import bluesky_gym
        bluesky_gym.register_envs()
 
        cfg = self.cfg

        algo_name = cfg.model.algorithm.__name__ if cfg.model.algorithm else "Unspecified"

        print(
            f"▶️  Experiment {cfg.run_id}"
            f"  |  env={cfg.env.env_name}"
            f"  |  algo={algo_name}"
        )
 
        model = None
 
        if cfg.session.do_train:
            self.train()
            model = getattr(self, "_model", None)
 
        if cfg.session.do_evaluate:
            if model is None:
                if cfg.model.algorithm is None:
                    raise ValueError("No algorithm specified in config, so can't load model for evaluation.  Please specify an algorithm or set do_evaluate=False.")
                model = cfg.model.algorithm.load(f"{cfg.save_path}/final_model")
            self.evaluate(model)