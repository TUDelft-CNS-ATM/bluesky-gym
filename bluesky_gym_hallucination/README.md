
## bluesky_gym_hallucination/README.md

# BlueSky-Gym Hallucination Detection Add-On

## Overview

This module extends BlueSky-Gym with hallucination injection capabilities for ML safety research in air traffic control. It provides controlled simulation of the phantom intruder phenomenon that can cause dangerous decision-making in AI-based ATC systems.

## Key Research Applications

- **Safety Margin Quantification**: Measure how hallucinations affect separation distances and conflict resolution
- **Training Data Boundary Analysis**: Test ML model behavior outside operational envelopes  
- **Stress Testing**: Evaluate model robustness under adversarial sensor conditions
- **Certification Support**: Generate evidence for AI safety assurance in critical ATC applications

## Installation

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
import gymnasium as gym
import bluesky_gym_hallucination

# Create hallucination-enabled environment
env = gym.make("ConflictEnv-Hallucination-v0")

# Configure hallucination parameters
env.unwrapped.set_hallucination_params(p_halluc=0.1, magnitude=3.0)

# Training loop with hallucination analytics
for episode in range(100):
    obs, info = env.reset()
    
    while True:
        action = your_model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if hallucination occurred this step
        if info["hallucinated"]:
            print(f"Hallucination detected at step {info['hallucination_stats']['total_steps']}")
        
        if terminated or truncated:
            break
    
    # Analyze episode hallucination impact
    stats = env.unwrapped.get_detector().get_statistics()
    print(f"Episode {episode}: {stats['current_episode_hallucinations']} hallucinations")
```

### Advanced Research Configuration

```python
# Create custom hallucination environment
from bluesky_gym_hallucination import make_hallucination_env

env = make_hallucination_env(
    base_env_id="SectorEnv-v0",
    p_halluc=0.15,        # 15% hallucination probability
    magnitude=2.5,        # 2.5x magnitude scaling
    render_mode="human"   # Visual debugging
)

# Access detailed analytics
detector = env.get_detector()

# Training with hallucination correlation analysis
conflict_events = []
for step in range(1000):
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Log conflicts for safety impact analysis
    if 'conflicts' in info and info['conflicts']:
        conflict_events.extend([
            {'step': step, 'severity': len(info['conflicts'])}
        ])

# Compute safety impact metrics
safety_metrics = detector.compute_safety_impact_metrics(conflict_events)
print(f"Hallucination-conflict correlation: {safety_metrics['correlation']:.3f}")
```

## Available Environments

All base BlueSky-Gym environments have hallucination variants:

- `DescentEnv-Hallucination-v0`
- `AscentEnv-Hallucination-v0`
- `SectorEnv-Hallucination-v0`
- `TaxiEnv-Hallucination-v0`
- `MergeEnv-Hallucination-v0`
- `ConflictEnv-Hallucination-v0`
- `StackEnv-Hallucination-v0`

## Parameter Tuning Guidelines

### Hallucination Probability (`p_halluc`)
- **Low (0.01-0.05)**: Realistic operational conditions
- **Medium (0.05-0.15)**: Stress testing scenarios  
- **High (0.15-0.30)**: Extreme adversarial conditions
- **Research Range**: 0.0-1.0 (higher values for boundary testing)

### Magnitude Scaling (`magnitude`)
- **Conservative (1.0-2.0)**: Subtle sensor noise simulation
- **Moderate (2.0-3.5)**: Significant phantom readings
- **Aggressive (3.5-5.0+)**: Severe hallucination effects
- **Research Range**: 1.0+ (unbounded for extreme testing)

## Research Integration

### Training Data Boundary Testing

```python
# Test model behavior outside training envelope
boundary_test_configs = [
    {"p_halluc": 0.05, "magnitude": 1.5},  # Nominal
    {"p_halluc": 0.20, "magnitude": 3.0},  # Moderate stress
    {"p_halluc": 0.40, "magnitude": 5.0},  # Extreme stress
]

results = {}
for config in boundary_test_configs:
    env.set_hallucination_params(**config)
    results[str(config)] = run_evaluation(env, episodes=50)
```

### Monte Carlo Safety Analysis

```python
# Generate large-scale hallucination impact datasets
import numpy as np

halluc_probs = np.linspace(0.0, 0.3, 31)
magnitudes = np.logspace(0, 1, 11)  # 1.0 to 10.0

safety_matrix = np.zeros((len(halluc_probs), len(magnitudes)))

for i, p in enumerate(halluc_probs):
    for j, mag in enumerate(magnitudes):
        env.set_hallucination_params(p_halluc=p, magnitude=mag)
        safety_score = evaluate_safety_performance(env)
        safety_matrix[i, j] = safety_score
```

## Performance Metrics

The detector automatically tracks key research metrics:

- **Detection Rate**: Frequency of hallucination events
- **Magnitude Distribution**: Statistical analysis of hallucination severity
- **Temporal Patterns**: Time-series analysis of hallucination occurrence
- **Safety Correlation**: Relationship between hallucinations and conflicts
- **Episode Statistics**: Per-episode hallucination impact analysis

## Citation

If using this module for research, please cite:

```bibtex
@misc{bluesky_gym_hallucination,
    title={BlueSky-Gym Hallucination Detection Framework},
    author={Panigrahi, Somnath},
    year={2025},
    note={ML Hallucination Quantification for Air Traffic Control Safety}
}
```