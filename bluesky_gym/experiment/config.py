"""
bluesky_gym/experiment/config.py
---------------------------------
Environment-independent configuration layer.
 
Structure
---------
  EnvKwargsConfig   - base class; subclass to add env-specific kwargs
  EnvConfig         - wraps EnvKwargsConfig; adds env_name, success_key, group_key
  ModelConfig       - base class; subclass to add algo-specific fields
  SessionConfig     - generic session settings (timesteps, eval, etc.)
  ExperimentConfig  - root object; owns paths, YAML save/load, and
                      automatic CLI generation from dataclass fields
 
Design notes
------------
- EnvConfig and ModelConfig are intentionally sparse base classes.
  Users subclass them to declare env/algo-specific knobs:
 
      class MyExperiment(BaseExperiment):
          model_config_cls = MyModelConfig   # must subclass ModelConfig
          env_config_cls   = MyEnvConfig     # must subclass EnvConfig
 
- ModelConfig.algorithm defaults to None — subclasses must set a concrete
  default (e.g. algorithm: Type = SAC).  Leaving it None raises a clear
  error at startup so users are never confused about which algo is running.
 
- EnvConfig.env_name defaults to None — subclasses must set it.
  EnvConfig.group_key defaults to None — set it to the info-dict key your
  env uses for grouping episodes (e.g. "current_runway").  Leaving it None
  disables per-group reporting without raising an error.
 
- CLI arguments are generated automatically from dataclass fields at
  runtime (see _build_parser()).  EnvConfig fields and the nested
  EnvKwargsConfig fields are both exposed under the "env" prefix.
  You never write argparse boilerplate.
 
- Priority chain: dataclass defaults → YAML file → CLI flags
 
- YAML round-trips through _to_dict() / _make().  The `algorithm`
  field (a class object) is serialised as its __name__ string.
"""

from __future__ import annotations

import dataclasses
import glob
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, get_args, get_origin

import yaml

#TODO: add support for loading in a pre-trained model from disk for the session, we should also add a check that the model is compatible with the environment!

# ---------------------------------------------------------------------------
# Base config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EnvKwargsConfig:
    """Base class for environment kwargs forwarded verbatim to gym.make().
 
    Subclass to declare env-specific knobs:
 
        @dataclass
        class PathPlanningEnvKwargsConfig(EnvKwargsConfig):
            action_mode: str          = "hdg"
            use_rta:     bool         = False
            runways:     list[str]    = field(default_factory=list)
 
    All fields are forwarded to gym.make() via as_kwargs().
    """

    def get_group_kwarg_name(self) -> Optional[str]:
        """Return the name of the kwarg field used for grouping (e.g., 'runways').
        
        Subclasses should override this if the environment accepts a list of 
        groups to train/evaluate on.
        """
        return None

    def as_kwargs(self) -> Dict[str, Any]:
        """Return all fields as a plain dict for gym.make(**kwargs)."""
        return asdict(self)

@dataclass
class EnvConfig:
    """Base class for full environment configuration.
 
    Wraps an EnvKwargsConfig (the raw gym.make kwargs) and adds the
    metadata the framework itself needs: env_name, success_key, group_key.
 
    Subclass to attach your env-specific kwargs class and set the required
    fields:
 
        @dataclass
        class PathPlanningEnvConfig(EnvConfig):
            env_kwargs:  PathPlanningEnvKwargsConfig = field(
                default_factory=PathPlanningEnvKwargsConfig
            )
            env_name:    str = "PathPlanningGoalEnv-v0"
            group_key:   str = "current_runway"
            success_key: str = "is_success"   # this is already the default
 
    Attributes
    ----------
    env_kwargs : EnvKwargsConfig
        Forwarded verbatim to gym.make().
    env_name : str | None
        Gymnasium environment ID.  Must be set by subclasses.
    success_key : str
        Key in the episode info dict that signals success.
        Defaults to "is_success" — the Gymnasium standard.
    group_key : str | None
        Key in the episode info dict that identifies the episode group
        (e.g. runway, level, map).  Set by subclasses to enable per-group
        tracking and reporting.  None disables grouping silently.
    """
 
    env_kwargs:  EnvKwargsConfig = field(default_factory=EnvKwargsConfig)
    env_name:    Optional[str]   = None
    success_key: str             = "is_success"
    group_key:   Optional[str]   = None
 
    def as_kwargs(self) -> Dict[str, Any]:
        """Return the nested env_kwargs as a plain dict for gym.make()."""
        return self.env_kwargs.as_kwargs()
 
    def validate(self) -> None:
        """Raise a clear error if required fields have not been set.
 
        Called by ExperimentConfig.__post_init__ so misconfiguration is
        caught at construction time, not mid-training.
        """
        if self.env_name is None:
            raise ValueError(
                f"{type(self).__name__}.env_name is None.\n"
                "Set it in your EnvConfig subclass:\n\n"
                "    @dataclass\n"
                "    class MyEnvConfig(EnvConfig):\n"
                "        env_name:  str = 'MyEnv-v0'\n"
                "        group_key: str = 'current_level'\n"
            )
        
@dataclass
class ModelConfig:
    """Base class for model / algorithm configuration.
 
    Intentionally minimal — only fields that are universal across all SB3
    algorithms live here.  Net architecture, policy kwargs, replay buffer
    settings, and any other algo-specific knobs belong in user subclasses.
 
    Subclass example:
 
        @dataclass
        class SACModelConfig(ModelConfig):
            algorithm: Type = SAC
            net_arch: Dict[str, List[int]] = field(
                default_factory=lambda: dict(pi=[256, 256], qf=[256, 256])
            )
 
            @property
            def policy_kwargs(self) -> dict:
                return dict(net_arch=self.net_arch)
 
    The `algorithm` field holds a class object but is serialised as its
    __name__ string for YAML round-trips.  It defaults to None here —
    subclasses MUST provide a concrete default or the framework raises a
    clear error at startup before any training begins.
    """
 
    algorithm:     Optional[Type] = None
    learning_rate: float          = 3e-4
    verbose:       int            = 1

    def __post_init__(self) -> None:
        """Framework-controlled lifecycle hook."""
        if isinstance(self.algorithm, str):
            self.algorithm = self.resolve_algorithm(self.algorithm)

    def resolve_algorithm(self, name: str) -> Type:
        """
        Hook for subclasses to convert a string name to a class Type.
        By default, it raises an error to force the user to define the source.
        """
        raise NotImplementedError(
            f"{type(self).__name__} received algorithm name '{name}' but "
            "does not implement resolve_algorithm()."
        )
 
    def validate(self) -> None:
        """Raise a clear error if algorithm has not been set."""
        if self.algorithm is None:
            raise ValueError(
                f"{type(self).__name__}.algorithm is None.\n"
                "Set it in your ModelConfig subclass:\n\n"
                "    from stable_baselines3 import SAC\n\n"
                "    @dataclass\n"
                "    class MyModelConfig(ModelConfig):\n"
                "        algorithm: Type = SAC\n"
            )

    def get_algorithm(self) -> Type:
        """Return the algorithm class to use. Subclasses MUST override this."""
        if self.algorithm is None:
            raise ValueError(
                f"{type(self).__name__}.algorithm is None.\n"
                "Set it in your ModelConfig subclass:\n\n"
                "    from stable_baselines3 import SAC\n\n"
                "    @dataclass\n"
                "    class MyModelConfig(ModelConfig):\n"
                "        algorithm: Type = SAC\n"
            )
        return self.algorithm


@dataclass
class SessionConfig:
    """Generic session settings — independent of env and algorithm."""
    total_timesteps: int = 250_000

    # List of callback keys to load from the registry
    callbacks: List[str] = field(default_factory=lambda: ["csv_logger", "checkpoint", "eval", "success_rate"])

    # Load a pretrained model (a .zip file)
    pretrained_model_path: Optional[str] = None

    # None  → use all available groups (runways, levels, …)
    train_groups: Optional[List[str]] = field(default_factory=lambda: None)
    eval_groups:  Optional[List[str]] = field(default_factory=lambda: None)

    eval_episodes: int  = 10
    do_train:      bool = True
    do_evaluate:   bool = True

    eval_freq:            int  = 5_000
    track_training_evals: bool = False

# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Root configuration object.  One instance per experiment run.

    Do not instantiate directly — use from_args(), from_yaml(), or load().

    Parameterised by env and model config subclasses — set these via class
    variables on your BaseExperiment subclass, not here directly.

    Path layout
    -----------
    ./experiments/<env_name>/<algo>/
        logs/<run_id>/
        models/<run_id>/
            final_model.zip
            checkpoint_model.zip
            config.yaml
    """

    model:   ModelConfig     = field(default_factory=ModelConfig)
    session: SessionConfig   = field(default_factory=SessionConfig)
    env:     EnvConfig       = field(default_factory=EnvConfig)

    run_id: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    def __post_init__(self) -> None:
        """Validate configs and build paths."""
        self.env.validate()
        self.model.validate()
        self._build_paths()

    def _build_paths(self) -> None:
        algo_name = self.model.algorithm.__name__ if self.model.algorithm else "UnknownAlgorithm"
        base = (
            f"./experiments"
            f"/{self.env.env_name}"
            f"/{algo_name}"
        )
        self.log_dir   = f"{base}/logs/{self.run_id}/"
        self.save_path = f"{base}/models/{self.run_id}/"
        self.save_freq = max(5_000, self.session.total_timesteps // 100) #TODO: make this more flexible / configurable if we want to support algorithms that don't use timesteps as their main training unit (e.g. PPO with update_epochs) or if we want a different default cadence for shorter/longer experiments.
        os.makedirs(self.log_dir,   exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Env kwargs properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def train_env_kwargs(self) -> Dict[str, Any]:
        """kwargs for the training environment.
 
        Starts from the env_kwargs fields and overwrites the groups field
        (the first field on EnvKwargsConfig ending with 'groups') with the
        session's train_groups value.
        """
        kwargs = self.env.as_kwargs()
        _inject_groups(kwargs, self.env.env_kwargs, self.session.train_groups)
        return kwargs

    @property
    def eval_env_kwargs(self) -> Dict[str, Any]:
        kwargs = self.env.as_kwargs()
        _inject_groups(kwargs, self.env.env_kwargs, self.session.eval_groups)
        return kwargs

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def _to_dict(self, include_metadata: bool = True) -> dict:
        """Serialise to a plain dict.
 
        The nested structure mirrors the dataclass hierarchy exactly so
        _make() can reconstruct it field-by-field.
        """
        d = asdict(self)

        # Convert the algorithm class object to its string name
        if self.model.algorithm is not None:
            # self.model.algorithm is a Type, e.g., <class 'SAC'>
            d["model"]["algorithm"] = self.model.algorithm.__name__

        if not include_metadata:
            d.pop("run_id", None)
            return d
        
        # Store concrete subclass names so load() can reconstruct them accurately
        d["_model_config_cls"] = type(self.model).__name__
        d["_env_config_cls"]   = type(self.env).__name__
        d["_env_kwargs_cls"]   = type(self.env.env_kwargs).__name__
        return d

    def save(self, filename: str = "config.yaml", include_metadata: bool = True) -> None:
        """Write config.yaml next to the model files."""
        path = os.path.join(self.save_path, filename)
        with open(path, "w") as f:
            yaml.dump(self._to_dict(include_metadata), f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------ #
    # Construction helpers (used by from_args / from_yaml / load)        #
    # ------------------------------------------------------------------ #

    @classmethod
    def _make(
        cls,
        d:                dict,
        model_config_cls: Type[ModelConfig],
        env_config_cls:   Type[EnvConfig],
        fresh_run_id:     bool = False,
    ) -> "ExperimentConfig":
        """Internal factory shared by from_yaml, from_args, and load."""
        
        # ── model ──────────────────────────────────────────────────────
        # We no longer pop "algorithm" because it's managed by the class method,
        # not necessarily a stored field in the dict.
        model_d   = dict(d.get("model", {}))
        model_cfg = _construct_dataclass(model_config_cls, model_d)
 
        # ── session ────────────────────────────────────────────────────
        session_cfg = _construct_dataclass(SessionConfig, d.get("session", {}))
 
        # ── env (two-level) ────────────────────────────────────────────
        env_d = dict(d.get("env", {}))
        
        # Pull out the nested env_kwargs dict and reconstruct separately
        env_kwargs_d   = env_d.pop("env_kwargs", {})
        env_kwargs_cls = _env_kwargs_cls_from(env_config_cls)
        env_kwargs_cfg = _construct_dataclass(env_kwargs_cls, env_kwargs_d)
        
        # Reconstruct the main EnvConfig with its specific kwargs object
        env_cfg = _construct_dataclass(
            env_config_cls, 
            env_d, 
            env_kwargs=env_kwargs_cfg
        )
 
        # ── run_id ─────────────────────────────────────────────────────
        run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            if fresh_run_id
            else d.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        )
 
        # Assemble the final ExperimentConfig object
        obj = cls.__new__(cls)
        object.__setattr__(obj, "model",   model_cfg)
        object.__setattr__(obj, "session", session_cfg)
        object.__setattr__(obj, "env",     env_cfg)
        object.__setattr__(obj, "run_id",  run_id)
        
        # This triggers validation, which now includes model_cfg.get_algorithm()
        obj.__post_init__()
        return obj

    @classmethod
    def from_yaml(
        cls,
        yaml_path:        str,
        model_config_cls: Type[ModelConfig] = ModelConfig,
        env_config_cls:   Type[EnvConfig]   = EnvConfig,
    ) -> "ExperimentConfig":
        """Load config from a (partial) YAML file.
 
        Missing sections/fields fall back to dataclass defaults.
        """
        with open(yaml_path) as f:
            d = yaml.safe_load(f) or {}
        return cls._make(d, model_config_cls, env_config_cls)

    @classmethod
    def load(
        cls,
        run_id:           str,
        model_config_cls: Type[ModelConfig] = ModelConfig,
        env_config_cls:   Type[EnvConfig]   = EnvConfig,
    ) -> "ExperimentConfig":
        """Reconstruct an ExperimentConfig from a previously saved run."""
        pattern = f"./experiments/*/*/models/{run_id}/config.yaml"
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(
                f"No config.yaml found for run_id='{run_id}'.\n"
                f"Searched: {pattern}"
            )
        with open(matches[0]) as f:
            d = yaml.safe_load(f)
        return cls._make(d, model_config_cls, env_config_cls)

    # ------------------------------------------------------------------ #
    # Automatic CLI                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def _build_parser(
        cls,
        model_config_cls: Type[ModelConfig] = ModelConfig,
        env_config_cls:   Type[EnvConfig]   = EnvConfig,
        description:      str               = "Train / evaluate an RL agent.",
    ):
        """Build an argparse.ArgumentParser from the dataclass fields.
 
        Flags are generated for SessionConfig, model_config_cls, EnvConfig,
        and the nested EnvKwargsConfig — all env-related flags share the
        "env" prefix.
 
        Type mapping
        ------------
          str           → --flag  type=str
          int           → --flag  type=int
          float         → --flag  type=float
          bool          → --flag / --no-flag  pair
          list / List   → --flag  nargs="+"
          Type          → --flag  type=str  (algorithm name)
          dataclass     → skipped (expanded separately)
        """
        import argparse
 
        p = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
 
        p.add_argument("--config", type=str, default=None, metavar="PATH",
                       help="Path to a YAML config file. CLI flags override it.")
        p.add_argument("--run-id", type=str, default=None, metavar="ID",
                       help="Existing run ID to load (used with --no-train).")
 
        for section, dc_cls in [
            ("session", SessionConfig),
            ("model",   model_config_cls),
        ]:
            for f in fields(dc_cls):
                _add_argument(p, section, f)
 
        # EnvConfig fields (skip the nested env_kwargs field itself)
        env_kwargs_cls = _env_kwargs_cls_from(env_config_cls)
        for f in fields(env_config_cls):
            if f.name == "env_kwargs":
                continue
            _add_argument(p, "env", f)
        # Nested EnvKwargsConfig fields — also under "env" prefix
        for f in fields(env_kwargs_cls):
            _add_argument(p, "env", f)
 
        return p

    @classmethod
    def from_args(
        cls,
        args,
        model_config_cls: Type[ModelConfig] = ModelConfig,
        env_config_cls:   Type[EnvConfig]   = EnvConfig,
    ) -> "ExperimentConfig":
        """Build an ExperimentConfig from parsed CLI args.
 
        Priority chain: dataclass defaults → YAML (--config) → CLI flags.
        """
        if getattr(args, "config", None):
            cfg = cls.from_yaml(args.config, model_config_cls, env_config_cls)
        else:
            cfg = cls._make({}, model_config_cls, env_config_cls, fresh_run_id=True)
 
        _apply_args_to_cfg(cfg, args, model_config_cls, env_config_cls)
        cfg._build_paths()
        return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MISSING = object()   # sentinel distinct from None

def _inject_groups(
    kwargs:     dict,
    env_cfg:    EnvKwargsConfig,
    group_list: Optional[List[str]],
) -> None:
    """Overwrite the target group kwarg with group_list, if the env supports it."""
    group_kwarg = env_cfg.get_group_kwarg_name()
    if group_kwarg is not None:
        kwargs[group_kwarg] = group_list

def _env_kwargs_cls_from(env_config_cls: Type[EnvConfig]) -> Type[EnvKwargsConfig]:
    """Return the concrete EnvKwargsConfig subclass used by env_config_cls.
 
    Inspects the default_factory of the 'env_kwargs' field, which is the
    most reliable source since it doesn't depend on annotation resolution.
    Falls back to the base EnvKwargsConfig if nothing is found.
    """
    for f in fields(env_config_cls):
        if f.name == "env_kwargs":
            if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                try:
                    return type(f.default_factory())  # type: ignore[misc]
                except Exception:
                    pass
            if isinstance(f.type, type) and issubclass(f.type, EnvKwargsConfig):
                return f.type
    return EnvKwargsConfig

def _construct_dataclass(cls, d: dict, **extra) -> Any:
    """Construct a dataclass from a dict, ignoring unknown keys.

    `extra` kwargs override values in `d` before construction.
    """
    known = {f.name for f in fields(cls)}
    merged = {k: v for k, v in d.items() if k in known}
    merged.update(extra)
    return cls(**merged)


def _field_dest(section: str, field_name: str) -> str:
    """argparse stores '--section-field-name' as 'section_field_name'."""
    return f"{section}_{field_name}"

def _apply_args_to_cfg(cfg, args, model_config_cls, env_config_cls) -> None:
    """Write explicit CLI values onto an existing ExperimentConfig in place."""
    env_kwargs_cls = _env_kwargs_cls_from(env_config_cls)
    _apply_section(args, cfg.session,          "session", SessionConfig)
    _apply_section(args, cfg.model,            "model",   model_config_cls)
    _apply_section(args, cfg.env,              "env",     env_config_cls,
                   skip=frozenset(["env_kwargs"]))
    _apply_section(args, cfg.env.env_kwargs,   "env",     env_kwargs_cls)
 
def _apply_section(
    args,
    target,
    section:  str,
    dc_cls,
    skip:     frozenset = frozenset(),
) -> None:
    """Apply CLI args for one section onto a target dataclass instance."""
    for f in fields(dc_cls):
        if f.name in skip:
            continue
        dest = _field_dest(section, f.name)
        val  = getattr(args, dest, _MISSING)
        if val is _MISSING or val is None:
            continue
        
        setattr(target, f.name, val)

def _add_argument(parser, section: str, f: dataclasses.Field) -> None:
    """Register one dataclass field as an argparse argument.
 
    Silently skips nested dataclass fields (they are expanded by the caller).
    """
    import argparse
 
    # Skip nested dataclass fields
    if isinstance(f.type, type) and dataclasses.is_dataclass(f.type):
        return
 
    flag = f"--{section}-{f.name.replace('_', '-')}"
    dest = _field_dest(section, f.name)
    hint = f.metadata.get("help", "")
 
    origin = get_origin(f.type)
    args_t = get_args(f.type)
 
    # bool → --flag / --no-flag pair
    if f.type is bool or f.type == "bool":
        parser.add_argument(flag,    dest=dest, action="store_true",
                            default=None, help=f"Enable {f.name}. {hint}")
        no_flag = f"--{section}-no-{f.name.replace('_', '-')}"
        parser.add_argument(no_flag, dest=dest, action="store_false",
                            help=f"Disable {f.name}. {hint}")
        return
 
    # Type field (algorithm class) → string name
    if f.type is Type or (isinstance(f.type, str) and "Type" in f.type):
        parser.add_argument(flag, dest=dest, type=str, default=None,
                            metavar="ALGO",
                            help=f"Algorithm name (e.g. SAC, TD3). {hint}")
        return
 
    # list / List[X]
    if origin in (list, List) or f.type in (list, List):
        inner = args_t[0] if args_t else str
        parser.add_argument(flag, dest=dest, nargs="+", type=inner,
                            default=None, metavar=f.name.upper(), help=hint)
        return
 
    # Optional[X] → unwrap
    if origin is type(None) or (
        hasattr(f.type, "__origin__") and type(None) in get_args(f.type)
    ):
        inner = next((a for a in get_args(f.type) if a is not type(None)), str)
        if isinstance(inner, type) and dataclasses.is_dataclass(inner):
            return
        parser.add_argument(flag, dest=dest, type=inner, default=None, help=hint)
        return
 
    # Primitives
    type_map = {int: int, float: float, str: str,
                "int": int, "float": float, "str": str}
    py_type  = type_map.get(f.type, str)
    parser.add_argument(flag, dest=dest, type=py_type, default=None, help=hint)