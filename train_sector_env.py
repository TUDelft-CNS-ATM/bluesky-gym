import gymnasium as gym
import bluesky_gym
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.callbacks import CheckpointCallback

bluesky_gym.register_envs()

env = gym.make('SectorCREnv-v1', render_mode=None)
# model = DDPG("MultiInputPolicy",env,tensorboard_log="log")
model = PPO("MultiInputPolicy", env, tensorboard_log="log")

checkpoint_callback = CheckpointCallback(
    save_freq=100000,                 # Save every 10,000 steps
    save_path="./checkpoints/",      # Folder to save checkpoints
    name_prefix="ddpg_model"         # Prefix for checkpoint files
)

model.learn(total_timesteps=2e6, progress_bar=True, callback=checkpoint_callback)
model.save("model_altitude_test_ppo_sector")

