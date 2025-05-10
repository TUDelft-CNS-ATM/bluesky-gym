import gymnasium as gym
import bluesky_gym
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers import RecordVideo
import os

bluesky_gym.register_envs()

TRAIN = False
RECORD = False

if RECORD:
    env = gym.make('SectorCREnv-v1', render_mode='rgb_array')
    video_folder = "./videos/"
    os.makedirs(video_folder, exist_ok=True)
    env = RecordVideo(env, video_folder, episode_trigger=lambda e: True)
else:
    env = gym.make('SectorCREnv-v1', render_mode="human")

if TRAIN:
    model = DDPG("MultiInputPolicy",env,tensorboard_log="log")
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./checkpoints/",      # Folder to save checkpoints
        name_prefix="ppo_model"  
    )

    model.learn(total_timesteps=2e6, progress_bar=True, callback=checkpoint_callback)
    model.save("ppo_model_sector_final")
else:
    model = DDPG.load("checkpoints/ppo_model_1800000_steps.zip", env=env) 

obs, info = env.reset()
done = truncated = False
while not (done):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

env.close()

