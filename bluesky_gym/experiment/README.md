# Bluesky-Gym Experiment Framework

This system automates the "boilerplate" of RL: managing configurations, logging results, generating CLI flags and visualising performance.

## Key Features
* **Automatic CLI Generation**: Every field in your configuration dataclass automatically becomes a command-line argument.
* **Hierarchical Configs**: Separate control over the Session (timesteps, callbacks), the Model (hyperparameters, algorithms) and the Environment (gym kwargs).
* **Unified Entry Point**: One script handles training, evaluation, plotting and video recording.
* **Custom Metric Extraction**: Easily pull and aggregate domain-specific data from environment `info` dicts.

---

## 1. Project Structure
When you run experiments, the framework organises artifacts automatically:
```text
./experiments/
├── registry.csv                    # Global metadata log (Optional)
└── <EnvName>/
    └── <AlgorithmName>/
        ├── logs/
        │   └── <run_id>/           # Tensorboard & CSV logs
                └── training_log.csv
        └── models/
            └── <run_id>/           # Checkpoints & Config
                ├── final_model.zip
                ├── config.yaml      # Full experiment settings
                └── training_evals.csv
```

---
## 2. Implementing a New Experiment

To plug your environment into the framework, you only need to define one file (e.g., `experiment.py`) containing three configuration classes and one experiment class.

### Step 1: Define Configurations
Subclass the base configs to add your specific hyperparameters.

```python
from dataclasses import dataclass, field
from bluesky_gym.experiment import ModelConfig, EnvConfig, EnvKwargsConfig
from stable_baselines3 import SAC

@dataclass
class MyModelConfig(ModelConfig):
    algorithm: type = SAC
    learning_rate: float = 3e-4
    use_her: bool = True  # Custom field!

    def resolve_algorithm(self, name: str) -> Type:
        if name == "SAC":
            return SAC
        else:
            raise ValueError(f"Unsupported algorithm: {name}")    

@dataclass
class MyEnvKwargs(EnvKwargsConfig):
    # These are passed directly to gym.make(**kwargs)
    difficulty: str = "easy"
    obs_type: str = "lidar"

@dataclass
class MyEnvConfig(EnvConfig):
    env_name: str = "MyCustomEnv-v0"
    env_kwargs: MyEnvKwargs = field(default_factory=MyEnvKwargs)
    success_key: str = "is_success" # Key in the info dict
```

### Step 2: Create the Experiment Class
```python
from bluesky_gym.experiment import BaseExperiment, MetricExtractor

class MyExperiment(BaseExperiment):
    model_config_cls = MyModelConfig
    env_config_cls = MyEnvConfig

    def make_env(self, env_kwargs, render_mode=None):
        import gymnasium as gym
        # Wrappers (like Monitor) can be applied here
        return gym.make(self.cfg.env.env_name, render_mode=render_mode, **env_kwargs)

    def make_model(self, env):
        mcfg = self.cfg.model
        return mcfg.algorithm("MultiInputPolicy", env, verbose=1, learning_rate=mcfg.learning_rate)

    @classmethod
    def metric_extractor(cls) -> MetricExtractor:
        # Define how to pull custom data from the environment info dict
        return MetricExtractor(
            extractors={
                "distance_error": lambda info, ok: info.get("dist", 0.0),
            },
            display=["distance_error"]
        )
```

### Step 3: Create the Entry Point

Create an optional `main.py` file. This acts as the "switcher" that handles all the CLI commands (train, evaluate, etc.) for your specific experiment.

```python
from bluesky_gym.experiment import run_experiment
from my_project.experiment import MyExperiment

if __name__ == "__main__":
    # This single line enables the entire CLI (train, enjoy, plot, etc.)
    run_experiment(MyExperiment)
```
---

## 3. Command Line Usage

Now that your `main.py` is set up, you can interact with your experiment using the following commands. The framework automatically generates flags based on your dataclass fields using the format `--{section}-{field_name}`.

### Training
Start training using the default settings or override them via the CLI:
```bash
# Train with defaults
python main.py train

# Override session, model, and env parameters
python main.py train \
    --session-total-timesteps 1000000 \
    --model-learning-rate 1e-4 \
    --model-use-her \
    --env-difficulty hard
```

### Evaluation & Visualisation
Once you have a `run-id` (generated automatically during training), you can analyse the results:
```bash
# Detailed evaluation (generates summary table and CSV)
python main.py evaluate --run-id 20260301_120000 --episodes 100

# Watch the agent play
python main.py enjoy --run-id 20260301_120000

# Record video of the agent
python main.py enjoy --run-id 20260301_120000 --record
```

### Plotting & Comparisons
```bash
# Plot training reward for a run
python main.py plot training --run-id 20260301_120000

# Compare multiple runs side-by-side
python main.py compare --runs 20260301_120000 20260301_150000
```

### Pro Tip: Configuration Files

If your CLI commands are getting too long, you can generate a template configuration file, edit it and run your experiment from that instead:

```bash
# Generate: 
python main.py generate-config --filename my_config.yaml

# Run:
python main.py train --config my_config.yaml
```

You can still override specific values from a file using CLI flags:
```bash
python main.py train --config my_config.yaml --session-total-timesteps 2000000
```
---

## 4. Advanced: Callbacks
You can enable built-in callbacks via the CLI:
* `--session-callbacks checkpoint` : Saves the model periodically.
* `--session-callbacks eval` : Evaluates the model on a separate environment during training.
* `--session-callbacks csv_logger` : Saves training logs to a CSV.

Example:
```bash
python main.py train --session-callbacks checkpoint eval csv_logger
```

### Defining Your Own Callbacks

You can add custom logic to the training loop by registering a new callback class. Your class should implement a from_config class method to handle its own initialisation from the experiment settings.

```python
from bluesky_gym.experiment.callbacks import callback_registry, BaseCallback

@callback_registry.register("my_custom_cb")
class MyCustomCallback(BaseCallback):
    def __init__(self):
        pass

    @classmethod
    def from_config(cls, cfg, **kwargs):
        # Access any config value to initialise your callback
        return cls(verbose=cfg.model.verbose)

    def _on_step(self) -> bool:
        # Your custom logic here
        return True
```

### Usage

Once registered in your code, you can trigger it just like a built-in one:

```bash
python main.py train --session-callbacks my_custom_cb checkpoint
```

## 5. Optional: The Experiment Registry
The Registry is an optional metadata layer. Use it when you want a persistent CSV log of all your runs to track things like "Intent," "Priority," or qualitative "Quality" scores.

### Example: Biscuit Quality Tracker

If you are training an agent to bake biscuits, you might want to track different flour types and the resulting crunchiness.

**Step 1: Define your Registry**
```python
from bluesky_gym.experiment.registry import BaseRegistry, register_command

class BiscuitRegistry(BaseRegistry):
    @property
    def headers(self):
        # These columns automatically become flags for the 'add' command
        return ["run_id", "timestamp", "flour_type", "is_good", "crunch_score"]

    @register_command("Label the bake result", status={"choices": ["delicious", "burnt", "soggy"]})
    def label(self, run_id: str, status: str):
        icons = {"delicious": "🍪", "burnt": "🔥", "soggy": "💧"}
        self.update_run(run_id, {"is_good": icons.get(status, status)})
```

**Step 2: Swap the Runner**

Instead of using runner.run_experiment, use your registry instance to launch:
```python
if __name__ == "__main__":
    registry = BiscuitRegistry()
    # This replaces the standard runner.run_experiment()
    registry.run_experiment(MyExperiment)
```

**Step 3: CLI Usage**

The registry commands are now merged into your script:
```bash
# Register a new run with custom metadata
python main.py registry add 20260301_120000 --flour-type "Whole Wheat"

# Use your custom 'label' command with restricted choices
python main.py registry label 20260301_120000 delicious

# List all tracked experiments in a table
python main.py registry list
```

The registry is completely separate from the training logic, it’s just a smart way to manage your `registry.csv` file without leaving the terminal.