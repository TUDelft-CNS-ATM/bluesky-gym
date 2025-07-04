## bluesky_gym_hallucination/__init__.py

import gymnasium as gym
from gymnasium.envs.registration import register
from .hallucination_wrapper import HallucinationWrapper

# Register hallucination variants of all base BlueSky-Gym environments
BASE_ENVS = [
    "DescentEnv-v0",
    "AscentEnv-v0", 
    "SectorEnv-v0",
    "TaxiEnv-v0",
    "MergeEnv-v0",
    "ConflictEnv-v0",
    "StackEnv-v0"
]

def register_hallucination_envs():
    """Register all hallucination-enabled environment variants"""
    for base_env in BASE_ENVS:
        halluc_id = base_env.replace("-v0", "-Hallucination-v0")
        
        register(
            id=halluc_id,
            entry_point="bluesky_gym_hallucination.hallucination_wrapper:make_hallucination_env",
            kwargs={"base_env_id": base_env, "p_halluc": 0.05, "magnitude": 2.0}
        )

def make_hallucination_env(base_env_id, p_halluc=0.05, magnitude=2.0, **kwargs):
    """Factory function to create hallucination-wrapped environments"""
    base_env = gym.make(base_env_id, **kwargs)
    return HallucinationWrapper(base_env, p_halluc=p_halluc, magnitude=magnitude)

# Auto-register on import
register_hallucination_envs()
