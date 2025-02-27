# BlueSky-Gym
A gymnasium style library for standardized Reinforcement Learning research in Air Traffic Management developed in Python.
Build on [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) and The Farama Foundation's [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)

<p align="center">
    <img src="https://github.com/user-attachments/assets/6ae83579-78af-4cb7-8096-3a10af54a5c5" width=50% height=50%><br/>
    <em>An example trained agent attempting the merge environment available in BlueSky-Gym.</em>
</p>

For a complete list of the currently available environments click [here](bluesky_gym/envs/README.md)

## Installation
**Update 27 February 2025:** *There is currently a bug when pip installing BlueSky-Simulator, which causes the pip install to fail on most machines (see [issue](https://github.com/TUDelft-CNS-ATM/bluesky/issues/543)). For now, users can clone the repository linked in [this](https://github.com/TUDelft-CNS-ATM/bluesky-gym/tree/main_bluesky) branch and pip install the requirements.txt file to circumvent this problem. This branch contains a local, barebones, version of BlueSky-Simulator from which the required functionality is retrieved.*

`pip install bluesky-gym`

Note that the pip package is `bluesky-gym`, for usage however, import as `bluesky_gym`.

## Usage
Using the environments follows the standard API from Gymnasium, an example of which is given below:

```python
import gymnasium as gym
import bluesky_gym
bluesky_gym.register_envs()

env = gym.make('MergeEnv-v0', render_mode='human')

obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = ... # Your agent code here
    obs, reward, done, truncated, info = env.step(action)
```

Additionally you can directly use algorithms from standardized libraries such as [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) or [RLlib](https://docs.ray.io/en/latest/rllib/index.html) to train a model:

```python
import gymnasium as gym
import bluesky_gym
from stable_baselines3 import DDPG
bluesky_gym.register_envs()

env = gym.make('MergeEnv-v0', render_mode=None)
model = DDPG("MultiInputPolicy",env)
model.learn(total_timesteps=2e6)
model.save()
```


## Citing

If you use BlueSky-Gym in your work, please cite it using:
```bibtex
@misc{bluesky-gym,
  author = {Groot, DJ and Leto, G and Vlaskin, A and Moec, A and Ellerbroek, J},
  title = {BlueSky-Gym: Reinforcement Learning Environments for Air Traffic Applications},
  year = {2024},
  journal = {SESAR Innovation Days 2024},
}
```

List of publications & preprints using `BlueSky-Gym` (please open a pull request to add missing entries):
*   _missing entry_
