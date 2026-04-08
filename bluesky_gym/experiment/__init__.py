"""
bluesky_gym/experiment/__init__.py
------------------------------------
Public API for the experiment framework.

Import surface
--------------
  from bluesky_gym.experiment import (
      # Entry point
      run_experiment,

      # Base classes to subclass
      BaseExperiment,
      EnvKwargsConfig,
      EnvConfig,
      ModelConfig,
      SessionConfig,
      ExperimentConfig,

      # Registry extension point
      BaseRegistry,
      register_command,

      # Evaluation extension point
      MetricExtractor,

      # Plotting extension point
      plot_training_curves,
      plot_eval_summary,
      plot_eval_episodes,

      # Standalone entry points
      train,
      evaluate,
      enjoy,
      compare,
      plot

      # Standalone CLIs (rarely needed directly)
      run_train_cli,
      run_evaluate_cli,
      run_enjoy_cli,
      run_compare_cli,
      run_plot_cli
  )
"""

from .runner          import run_experiment
from .base_experiment import BaseExperiment
from .config          import (
    EnvKwargsConfig,
    EnvConfig,
    ModelConfig,
    SessionConfig,
    ExperimentConfig,
)
from .train           import run_train_cli, train
from .evaluate        import MetricExtractor, run_evaluate_cli, evaluate
from .enjoy           import run_enjoy_cli, enjoy
from .compare_runs    import run_compare_cli, compare
from .plot            import (
    run_plot_cli, 
    plot_training_curves,
    plot_eval_summary, 
    plot_eval_episodes,
    plot
)
from .registry        import BaseRegistry, register_command

__all__ = [
    "run_experiment",
    "BaseExperiment",
    "EnvKwargsConfig",
    "EnvConfig",
    "ModelConfig",
    "SessionConfig",
    "ExperimentConfig",
    "MetricExtractor",
    "train",
    "evaluate",
    "enjoy",
    "compare",
    "plot",
    "run_train_cli",
    "run_evaluate_cli",
    "run_enjoy_cli",
    "run_compare_cli",
    "run_plot_cli",
    "plot_training_curves",
    "plot_eval_summary",
    "plot_eval_episodes",
    "BaseRegistry",
    "register_command",
]