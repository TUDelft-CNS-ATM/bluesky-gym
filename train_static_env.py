import gymnasium as gym
import bluesky_gym
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback

bluesky_gym.register_envs()

env = gym.make('StaticObstacleEnv-v0', render_mode=None)
model = DDPG("MultiInputPolicy",env,tensorboard_log="log")

checkpoint_callback = CheckpointCallback(
    save_freq=50000,                 # Save every 10,000 steps
    save_path="./checkpoints/",      # Folder to save checkpoints
    name_prefix="ddpg_model"         # Prefix for checkpoint files
)

print('learn')
# model.learn(total_timesteps=2e6, progress_bar=True, callback=checkpoint_callback)
# model.save("model")

env = gym.make('StaticObstacleEnv-v0', render_mode='human')

# model = DDPG.load("./model.zip", env=env) 
model = DDPG.load("checkpoints/ddpg_model_450000_steps.zip", env=env) 


obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action, _state = model.predict(obs, deterministic=True) # Your agent code here
    obs, reward, done, truncated, info = env.step(action)

