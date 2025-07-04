## bluesky_gym_marl/README.md

# Multi-Agent Conflict Detection & Resolution Suite

## Overview

This module implements multi-agent reinforcement learning (MARL) environments for air traffic conflict detection and resolution research. Each aircraft acts as an independent agent, enabling study of distributed decision-making, coordination protocols, and emergent behavior in complex airspace scenarios.

## Key Features

### PettingZoo Integration
- **Parallel Environment**: All agents act simultaneously each step
- **Gymnasium Compatibility**: Seamless integration with single-agent RL libraries
- **Stable-Baselines3 Support**: Multi-agent training wrappers included

### Cooperative Objectives
- **Shared Responsibility**: All agents penalized for system-wide conflicts
- **Local Observations**: Each agent observes only nearby aircraft (k-nearest neighbors)
- **Global Safety**: Collective goal of maintaining airspace-wide separation

### Research Applications
- **Distributed ATC**: Study decentralized conflict resolution protocols
- **Hallucination Impact**: Test how phantom observations affect multi-agent coordination
- **Scalability Analysis**: Evaluate performance with varying numbers of agents
- **Emergent Behavior**: Analyze self-organizing traffic patterns

## Dependencies

```bash
pip install pettingzoo>=1.22.0
pip install stable-baselines3>=2.0.0
pip install supersuit>=3.7.0  # Multi-agent wrappers
pip install ray[rllib]>=2.0.0  # Optional: for advanced MARL algorithms
```

## Quick Start

### Basic Multi-Agent Training

```python
import gymnasium as gym
import bluesky_gym_marl
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from bluesky_gym_marl.utils import make_sb3_env

# Create multi-agent environment
env = gym.make("MARLConflictEnv-v0", num_aircraft=6, cooperative_reward=True)

# Convert to Stable-Baselines3 compatible format
sb3_env = make_sb3_env(env, num_envs=4)

# Train multi-agent PPO
model = PPO("MlpPolicy", sb3_env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate trained agents
obs = env.reset()
for step in range(1000):
    actions = {agent: model.predict(obs[agent])[0] for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    
    if all(terminations.values()) or all(truncations.values()):
        break
```

### Advanced MARL with RLlib

```python
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from bluesky_gym_marl.utils import RLLibWrapper

# Initialize Ray
ray.init()

# Configure multi-agent PPO
config = (
    PPOConfig()
    .environment("MARLConflictEnv-v0")
    .multi_agent(
        policies={"shared_policy": (None, None, None, {})},
        policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy"
    )
    .training(lr=3e-4, train_batch_size=4000)
    .resources(num_gpus=1)
)

# Train multi-agent system
algorithm = config.build()
for i in range(100):
    result = algorithm.train()
    print(f"Iteration {i}: Mean reward = {result['episode_reward_mean']}")
```

## Environment Details

### Observation Space (Per Agent)
- **Own State**: [lat, lon, alt, heading, speed, vertical_rate]
- **Nearest Neighbors**: Up to 5 nearest aircraft with relative states
- **Conflict Indicators**: Binary flags for detected conflicts with each neighbor
- **Communication**: Optional message passing between nearby agents

### Action Space (Per Agent)
- **Heading Change**: ±45° maximum per step
- **Vertical Rate**: ±3000 ft/min climb/descent
- **Speed Adjustment**: ±15% of current airspeed
- **Communication**: Optional discrete messages to broadcast

### Reward Structure

#### Individual Rewards
- **Conflict Avoidance**: -10.0 per conflict involving this agent
- **Efficiency Penalty**: -0.1 × (action magnitude) for large maneuvers
- **Goal Progress**: +0.5 for maintaining course toward destination
- **Separation Bonus**: +0.2 for maintaining safe distances

#### Cooperative Rewards (Optional)
- **System-Wide Safety**: -5.0 × (total conflicts) shared by all agents
- **Traffic Flow**: +1.0 × (average progress) for collective efficiency
- **Coordination Bonus**: +0.5 for synchronized conflict resolution

## Research Configuration Examples

### Hallucination Impact Study

```python
from bluesky_gym_hallucination import HallucinationWrapper

# Create base MARL environment
base_env = gym.make("MARLConflictEnv-v0", num_aircraft=10)

# Wrap with hallucination injection
halluc_env = HallucinationWrapper(base_env, p_halluc=0.1, magnitude=2.0)

# Study how phantom observations affect multi-agent coordination
results = evaluate_marl_with_hallucinations(halluc_env, episodes=100)
```

### Scalability Analysis

```python
# Test different numbers of agents
agent_counts = [4, 8, 12, 16, 20]
performance_results = {}

for num_agents in agent_counts:
    env = gym.make("MARLConflictEnv-v0", 
                   num_aircraft=num_agents,
                   airspace_size=num_agents * 2.5)  # Scale airspace
    
    performance = benchmark_marl_performance(env, episodes=50)
    performance_results[num_agents] = performance
```

### Sector-Based Operations

```python
# Enable sector-based agent grouping
env = gym.make("MARLConflictEnv-v0",
               num_aircraft=16,
               sector_based=True,
               airspace_size=40.0)

# Agents are grouped by geographic sectors
# Enables study of hierarchical control structures
sector_performance = evaluate_sectored_agents(env)
```

## Advanced Features

### Communication Protocols
- **Local Broadcasting**: Agents can send discrete messages to nearby aircraft
- **Intent Sharing**: Share planned maneuvers to improve coordination
- **Emergency Signals**: High-priority conflict alerts

### Dynamic Scenarios
- **Weather Avoidance**: Moving hazard areas that require coordination
- **Route Changes**: Mid-flight destination updates
- **Equipment Failures**: Agents with reduced capabilities

### Performance Metrics
- **Conflict Rate**: System-wide conflicts per episode
- **Coordination Efficiency**: Measure of synchronized behavior
- **Emergent Patterns**: Analysis of self-organizing traffic flows
- **Communication Overhead**: Message passing frequency and content

## Citation

```bibtex
@misc{bluesky_gym_marl,
    title={Multi-Agent Reinforcement Learning for Air Traffic Conflict Resolution},
    author={Panigrahi, Somnath},
    year={2025},
    note={BlueSky-Gym MARL Extension for Distributed ATC Research}
}
```
