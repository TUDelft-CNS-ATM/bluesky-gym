import gymnasium as gym
import bluesky_gym
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers import RecordVideo
import os

bluesky_gym.register_envs()

# env = gym.make('StaticObstacleEnv-v1', render_mode=None)
# model = DDPG("MultiInputPolicy",env,tensorboard_log="log")
# model = PPO("MultiInputPolicy", env, tensorboard_log="log")

# checkpoint_callback = CheckpointCallback(
#     save_freq=100000,                 # Save every 10,000 steps
#     save_path="./checkpoints/",      # Folder to save checkpoints
#     name_prefix="ppo_model"         # Prefix for checkpoint files
# )


# model.learn(total_timesteps=2e6, progress_bar=True, callback=checkpoint_callback)
# model.save("model_altitude_test_ppo")

env = gym.make('StaticObstacleEnv-v1', render_mode='rgb_array')

# model = DDPG.load("./model.zip", env=env) 
model = DDPG.load("checkpoints/ddpg_model_400000_steps.zip", env=env) 


video_folder = "./videos/"
os.makedirs(video_folder, exist_ok=True)
env = RecordVideo(env, video_folder, episode_trigger=lambda e: True)


obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action, _state = model.predict(obs, deterministic=True) # Your agent code here
    obs, reward, done, truncated, info = env.step(action)

env.close()
