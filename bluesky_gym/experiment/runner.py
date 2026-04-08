from __future__ import annotations

import os
import sys
import textwrap
from typing import Type, TYPE_CHECKING, Dict, Tuple, Callable, Optional

if TYPE_CHECKING:
    from .base_experiment import BaseExperiment

CustomCommandMap = Dict[str, Tuple[Callable[[Type["BaseExperiment"]], None], str]]

def _print_global_help(experiment_cls: "Type[BaseExperiment]", custom_commands: Optional[CustomCommandMap] = None) -> None:
    """Prints a clean, top-level CLI menu including custom commands."""
    script_name = os.path.basename(sys.argv[0])
    
    # Define standard commands
    core_commands = {
        "train": "Train (and optionally evaluate) a new model. [default]",
        "evaluate": "Run detailed evaluation on a saved model.",
        "enjoy": "Watch or record a saved model.",
        "plot": "Plot training curves or evaluation results.",
        "compare": "Compare training metrics across multiple runs.",
        "generate-config": "Generate a default config.yaml for this experiment.",
    }

    # Build the command list string
    max_len = max(len(c) for c in list(core_commands.keys()) + list((custom_commands or {}).keys()))
    
    cmd_help = ""

    # Add core commands
    for cmd, desc in core_commands.items():
        cmd_help += f"      {cmd:<{max_len}}  {desc}\n"
    
    # Add custom commands if they exist
    if custom_commands:
        cmd_help += "\n    Custom Commands:\n"
        for cmd, (fn, desc) in custom_commands.items():
            cmd_help += f"      {cmd:<{max_len}}  {desc}\n"

    help_text = f"""\
    Usage: python {script_name} <command> [options]

    CLI for {experiment_cls.__name__}.

    Commands:
        {cmd_help}
    Type 'python {script_name} <command> --help' for details on a specific command.
    """
    print(textwrap.dedent(help_text))

def run_generate_config_cli(experiment_cls: Type[BaseExperiment]) -> None:
    """CLI wrapper to generate and save a default configuration file."""
    from .config import ExperimentConfig
    
    model_cls = experiment_cls.model_config_cls
    env_cls   = experiment_cls.env_config_cls
    
    # 1. Get the base parser (includes all model/env hyperparameters)
    parser = ExperimentConfig._build_parser(
        model_config_cls=model_cls,
        env_config_cls=env_cls,
        description=f"Generate a default config.yaml for {experiment_cls.__name__}."
    )

    # 2. Add specific arguments for the generation process
    parser.add_argument(
        "--path", 
        type=str, 
        default=".", 
        help="Directory where the config file should be saved (defaults to script directory).")
    parser.add_argument(
        "--filename", 
        type=str, 
        default="config.yaml", 
        help="Name of the output configuration file. Note: the file will be overwritten if it already exists and needs to end in '.yaml'."
    )

    # 3. Parse arguments
    args, _ = parser.parse_known_args()
    
    # 4. Initialize config and save to the specified location
    cfg = ExperimentConfig.from_args(args, model_cls, env_cls)
    
    # 5. Save
    caller_dir = os.path.dirname(os.path.abspath(sys.argv[0]))    
    target_dir = os.path.abspath(os.path.join(caller_dir, args.path))
    os.makedirs(target_dir, exist_ok=True)
    # Manually trigger the save to the specific path
    cfg.save_path = target_dir
    cfg.save(filename=args.filename, include_metadata=False)
        
    print(f"✅ Default config saved to {os.path.join(target_dir, args.filename)}")

def run_experiment(
    experiment_cls: "Type[BaseExperiment]", 
    custom_commands: Optional[CustomCommandMap] = None
) -> None:
    """Single entry point for the CLI with support for custom commands."""
    from .evaluate import run_evaluate_cli
    from .enjoy import run_enjoy_cli
    from .plot import run_plot_cli
    from .compare_runs import run_compare_cli
    from .train import run_train_cli

    # Build the dispatch map
    dispatch_map = {
        "train": run_train_cli,
        "evaluate": run_evaluate_cli,
        "enjoy": run_enjoy_cli,
        "plot": run_plot_cli,
        "compare": run_compare_cli,
        "generate-config": run_generate_config_cli,
    }
    
    # Merge custom commands into the dispatch map
    if custom_commands:
        for name, (fn, _) in custom_commands.items():
            dispatch_map[name] = fn

    known_commands = set(dispatch_map.keys())
    
    # Find the Command
    command = None
    for arg in sys.argv[1:]:
        if arg in known_commands:
            command = arg
            break

    # Global Help Intercept
    if command is None and any(arg in ["-h", "--help"] for arg in sys.argv):
        _print_global_help(experiment_cls, custom_commands)
        sys.exit(0)

    # Default to 'train'
    if command is None:
        command = "train"

    if command in dispatch_map:
        return _reparse_and_run(dispatch_map[command], experiment_cls, command)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reparse_and_run(sub_cli_fn, experiment_cls, command_name: str) -> None:
    """Strip the command from sys.argv and hand off to a sub-CLI function."""
    argv = sys.argv[:]
    for i in range(1, len(argv)):
        if argv[i] == command_name:
            argv.pop(i)
            break
    sys.argv = argv
    sub_cli_fn(experiment_cls)