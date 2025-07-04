## bluesky_gym_marl/__init__.py

import gymnasium as gym
from gymnasium.envs.registration import register
from pettingzoo.utils.env import ParallelEnv

# Register multi-agent conflict resolution environment
register(
    id="MARLConflictEnv-v0",
    entry_point="bluesky_gym_marl.marl_conflict_env:MARLConflictEnv",
    max_episode_steps=2000,
    kwargs={
        "num_aircraft": 8,
        "airspace_size": 20.0,  # NM x NM square airspace
        "separation_lateral_nm": 5.0,
        "separation_vertical_ft": 1000.0,
        "max_episode_steps": 2000,
        "cooperative_reward": True,
        "sector_based": False
    }
)
