import os
import csv
import abc
import sys
import argparse
import inspect
from datetime import datetime
from typing import List, Dict, Any, Type, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_experiment import BaseExperiment
    from .runner import CustomCommandMap

def register_command(help_text: Optional[str] = None, **arg_configs):
    """
    Decorator to mark a method as a CLI command for the registry.
    
    Usage:
        @register_command("Label a run", status={"choices": ["good", "bad", "meh"]})
        def label(self, run_id: str, status: str): ...
    """
    def wrapper(func):
        func._is_command = True
        func._command_help = help_text or func.__doc__ or "No description provided."
        func._arg_configs = arg_configs
        return func
    return wrapper

class BaseRegistry(abc.ABC):
    run_id: str = "run_id"
    timestamp: str = "timestamp"

    def __init__(self, filepath: str = "./experiments/registry.csv"):
        self.filepath = filepath
        self._ensure_exists()

    @property
    @abc.abstractmethod
    def headers(self) -> List[str]:
        """User-defined columns. 'run_id' is mandatory. 'timestamp' is optional but is auto-set when used. """
        pass

    def _run_experiment(self, experiment_cls: Type["BaseExperiment"], custom_commands: Optional[CustomCommandMap] = None):
        """
        Internal entry point that injects registry logic into the runner.
        Do not override this method in subclasses.
        """
        from .runner import run_experiment

        if custom_commands is None:
            custom_commands = {}
        
        custom_commands["registry"] = (
            self._dispatch,
            "Experiment metadata management suite"
        )

        run_experiment(experiment_cls, custom_commands=custom_commands)

    def run_experiment(self, experiment_cls: Type["BaseExperiment"], custom_commands: Optional[CustomCommandMap] = None):
        """
        Entry point for the experiment framework CLI.

        This method acts as a wrapper for the framework's runner. It can be 
        overridden in subclasses to perform setup tasks (e.g., cloud syncing, 
        pre-run validation) or teardown tasks.

        Note:
            If you override this method, you MUST call `self._run_experiment(...)` 
            within your implementation. Failure to do so will prevent the 
            registry commands from being injected into the CLI.
        """
        self._run_experiment(experiment_cls, custom_commands)

    def _dispatch(self, _experiment_cls):
        """Internal dispatcher that builds the CLI from decorated methods."""
        parser = argparse.ArgumentParser(prog="registry", formatter_class=argparse.RawTextHelpFormatter)
        subparsers = parser.add_subparsers(dest="subcmd", required=True)

        # Collect all methods marked with @register_command
        cmds = {name: method for name, method in inspect.getmembers(self, predicate=inspect.ismethod) 
                if hasattr(method, "_is_command")}

        for name, fn in cmds.items():
            command_name = name.replace("_", "-")
            sub = subparsers.add_parser(command_name, help=getattr(fn, "_command_help", "No description provided."))
            
            sig = inspect.signature(fn)
            overrides = getattr(fn, "_arg_configs", {})

            for p_name, p in sig.parameters.items():
                if p_name == "self": 
                    continue
                
                # Automatic Header Discovery for **kwargs (used in 'add')
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    for h in self.headers:
                        if h != self.run_id:
                            flag = f"--{h.replace('_', '-')}"
                            sub.add_argument(flag, type=str, default="")
                    continue

                # Build Argument
                is_optional = p.default is not inspect.Parameter.empty
                arg_name = f"--{p_name.replace('_', '-')}" if is_optional else p_name
                
                # Merge logic: Decorator overrides > Signature inference
                kwargs = overrides.get(p_name, {})
                if "type" not in kwargs and is_optional:
                    kwargs["type"] = type(p.default)
                if "default" not in kwargs and is_optional:
                    kwargs["default"] = p.default

                sub.add_argument(arg_name, **kwargs)

        # Parse starting from the registry sub-command
        args = parser.parse_args(sys.argv[2:])
        cmd_fn = cmds[args.subcmd.replace("-", "_")]
        
        # Prepare function arguments
        func_args = vars(args).copy()
        func_args.pop("subcmd")
        
        # Signature safety: only pass what the function expects unless it has **kwargs
        sig = inspect.signature(cmd_fn)
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            func_args = {k: v for k, v in func_args.items() if k in sig.parameters}
            
        cmd_fn(**func_args)

    # --- Built-in Commands ---

    @register_command("Add a new experiment run. Extra headers can be passed as --header-name.")
    def add(self, run_id: str, **kwargs):
        row = {h: "" for h in self.headers}
        row[self.run_id] = run_id
        
        if self.timestamp in self.headers:
            row[self.timestamp] = datetime.now().strftime("%Y-%m-%d %H:%M")

        for key, value in kwargs.items():
            clean_key = key.replace("-", "_")
            if clean_key in self.headers:
                row[clean_key] = value

        self._append_row(row)
        print(f"✨ Registered new run: {run_id}")

    @register_command("List all experiments in the registry.")
    def list(self):
        rows = self._read_all()
        if not rows: 
            return print("Registry is empty.")
        
        # Dynamic column sizing for a clean terminal output
        print(f"\n{'RUN ID':<20} | {'GOOD':<4} | {'INTENT'}")
        print("-" * 75)
        for r in rows:
            print(f"{r.get('run_id',''):<20} | {r.get('is_good',''):<4} | {r.get('intent','')}")
        print()

    @register_command("Delete a run from the registry.")
    def delete(self, run_id: str):
        rows = self._read_all()
        filtered = [r for r in rows if r.get('run_id') != run_id]
        if len(filtered) < len(rows):
            self._write_all(filtered)
            print(f"🗑️ Removed {run_id}")
        else:
            print(f"⚠️ Run {run_id} not found.")

    @register_command("Migrate the registry CSV to match the current headers.")
    def migrate(self):
        """Migrate the registry CSV to match the current headers.
        
        - Adds any new headers with an empty default value.
        - Drops any headers that have been removed.
        - Safe to run multiple times (idempotent).
        """
        rows = self._read_all()
        migrated = [
            {h: row.get(h, "") for h in self.headers}
            for row in rows
        ]
        self._write_all(migrated)
        print(f"✅ Migrated registry to headers: {self.headers}")

    # --- File I/O Helpers ---
    def update_run(self, run_id: str, updates: dict):
        """
        Standard update logic to modify specific rows.
        """
        rows = self._read_all()
        found = False
        for row in rows:
            if row.get(self.run_id) == run_id:
                row.update(updates)
                found = True
                break
        if not found:
            print(f"⚠️  Run '{run_id}' not found in registry.")
            return
        self._write_all(rows)

    def _ensure_exists(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.headers).writeheader()

    def _read_all(self) -> List[Dict[str, str]]:
        if not os.path.exists(self.filepath): return []
        with open(self.filepath, 'r', newline='') as f:
            return list(csv.DictReader(f))

    def _write_all(self, rows: List[Dict[str, str]]):
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(rows)

    def _append_row(self, row: Dict[str, Any]):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(row)