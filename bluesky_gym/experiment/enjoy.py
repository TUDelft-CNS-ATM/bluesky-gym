"""
bluesky_gym/experiment/enjoy.py
---------------------------------
Watch or record a previously trained model.

The config is reconstructed from the config.yaml saved alongside the
model, so enjoy always uses the exact settings the run was trained with.

Usage
-----
  python enjoy.py --run-id 20260331_134059
  python enjoy.py --run-id 20260331_134059 --episodes 10
  python enjoy.py --run-id 20260331_134059 --record
  python enjoy.py --run-id 20260331_134059 --groups 27 18R
"""

from __future__ import annotations

import argparse

import bluesky_gym
from gymnasium.wrappers import RecordVideo

from .config import ExperimentConfig


def run_enjoy_cli(experiment_cls) -> None:
    """Standalone enjoy CLI for a given experiment class.

    Called by run_experiment() when used as enjoy.py, or directly.
    """
    p = argparse.ArgumentParser(
        description="Watch or record a trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-id",       type=str, required=True)
    p.add_argument("--episodes",     type=int, default=5)
    p.add_argument("--record",       action="store_true",
                   help="Record episodes to video.")
    p.add_argument("--groups",       nargs="+", default=None, metavar="GROUP",
                   help="Override the eval groups from the saved config.")
    p.add_argument("--deterministic",action="store_true", default=True)
    args = p.parse_args()

    enjoy(
        experiment_cls=experiment_cls,
        run_id=args.run_id,
        episodes=args.episodes,
        record=args.record,
        groups=args.groups,
        deterministic=args.deterministic,
    )

def enjoy(
    experiment_cls,
    run_id:        str,
    episodes:      int  = 5,
    record:        bool = False,
    groups:        list | None = None,
    deterministic: bool = True,
) -> None:
    """Load a trained model and run/record episodes."""
    bluesky_gym.register_envs()

    cfg = ExperimentConfig.load(
        run_id,
        model_config_cls=experiment_cls.model_config_cls,
        env_config_cls=experiment_cls.env_config_cls,
    )

    env_name = cfg.env.env_name if cfg.env.env_name else "Unknown"
    algo_name = cfg.model.algorithm.__name__ if cfg.model.algorithm else "Unspecified"

    print(
        f"📺 Run: {run_id}"
        f"  |  env={env_name}"
        f"  |  algo={algo_name}"
    )

    eval_kwargs = cfg.eval_env_kwargs
    if groups is not None:
        from .config import _inject_groups
        _inject_groups(eval_kwargs, cfg.env.env_kwargs, groups)

    render_mode = "rgb_array" if record else "human"

    experiment = experiment_cls(cfg)
    env        = experiment.make_env(eval_kwargs, render_mode=render_mode)

    if record:
        video_dir = f"{cfg.save_path}/videos"
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda _: True)
        print(f"🎥 Recording to {video_dir}")

    model_path = f"{cfg.save_path}/final_model.zip"
    model      = cfg.model.get_algorithm().load(model_path, env=env)
    print(f"✅ Model loaded from {model_path}")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            if hasattr(action, "shape") and action.shape == ():
                action = action[()]
            obs, _, done, truncated, _ = env.step(action)
        print(f"  Episode {ep}/{episodes} done.")

    env.close()
    print("✅ Done.")