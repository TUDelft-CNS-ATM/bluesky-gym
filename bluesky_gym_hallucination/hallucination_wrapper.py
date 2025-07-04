## bluesky_gym_hallucination/hallucination_wrapper.py

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .detector import HallucinationDetector

class HallucinationWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that injects phantom intruder readings into environments
    containing 'intruder_distance' observations for ML hallucination research.
    
    This wrapper simulates the key hallucination phenomenon where ML models
    perceive non-existent threats or misinterpret sensor data, which is critical
    for safety margin quantification in air traffic control applications.
    """
    
    def __init__(self, env: gym.Env, p_halluc: float = 0.05, magnitude: float = 2.0):
        """
        Initialize hallucination wrapper
        
        Args:
            env: Base BlueSky-Gym environment to wrap
            p_halluc: Probability of hallucination injection per step (0.0-1.0)
            magnitude: Magnitude scaling factor for phantom readings (1.0-5.0)
        """
        super().__init__(env)
        
        self.p_halluc = max(0.0, min(1.0, p_halluc))  # Clamp to valid range
        self.magnitude = max(1.0, magnitude)
        
        # Initialize hallucination detector for research analytics
        self.detector = HallucinationDetector()
        
        # Cache observation space analysis
        self._has_intruder_data = self._check_intruder_observations()
        
        if not self._has_intruder_data:
            print(f"Warning: Environment {env.spec.id if hasattr(env, 'spec') else 'Unknown'} "
                  "does not contain 'intruder_distance' observations. "
                  "Hallucination injection will be disabled.")
    
    def _check_intruder_observations(self) -> bool:
        """Check if observation space contains intruder distance data"""
        if isinstance(self.observation_space, gym.spaces.Dict):
            return 'intruder_distance' in self.observation_space.spaces
        elif isinstance(self.observation_space, gym.spaces.Box):
            # Assume structured observation with intruder data in specific indices
            return self.observation_space.shape[0] > 4  # Basic assumption
        return False
    
    def _inject_phantom_intruder(self, observation: np.ndarray) -> np.ndarray:
        """
        Inject phantom intruder readings that simulate ML hallucination effects
        
        Args:
            observation: Original observation from environment
            
        Returns:
            Modified observation with phantom intruder data
        """
        if isinstance(observation, dict) and 'intruder_distance' in observation:
            # Dictionary-based observation space
            intruder_distances = observation['intruder_distance'].copy()
            
            # Create phantom intruder at critical distance (within separation minimum)
            # This simulates the most dangerous type of hallucination
            phantom_distance = np.random.uniform(0.5, 2.0) * self.magnitude  # NM
            phantom_bearing = np.random.uniform(0, 360)  # degrees
            phantom_alt_diff = np.random.uniform(-500, 500) * self.magnitude  # feet
            
            # Inject phantom reading (modify closest intruder or add new one)
            if len(intruder_distances) > 0:
                closest_idx = np.argmin(intruder_distances)
                observation['intruder_distance'][closest_idx] = phantom_distance
                if 'intruder_bearing' in observation:
                    observation['intruder_bearing'][closest_idx] = phantom_bearing
                if 'intruder_altitude_diff' in observation:
                    observation['intruder_altitude_diff'][closest_idx] = phantom_alt_diff
            
        elif isinstance(observation, np.ndarray) and len(observation) > 4:
            # Array-based observation space (assume first 4 are own-ship state)
            observation = observation.copy()
            
            # Inject phantom data in intruder observation slots
            for i in range(4, min(len(observation), 10)):  # Modify up to 6 intruder slots
                if np.random.random() < 0.3:  # 30% chance per intruder slot
                    phantom_value = np.random.uniform(0.1, 3.0) * self.magnitude
                    observation[i] = phantom_value
        
        return observation
    
    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with potential hallucination injection
        
        Returns observation, reward, terminated, truncated, info with hallucination metadata
        """
        # Execute base environment step
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Determine if hallucination occurs this step
        hallucination_triggered = (
            self._has_intruder_data and 
            np.random.random() < self.p_halluc
        )
        
        if hallucination_triggered:
            # Inject phantom intruder reading
            observation = self._inject_phantom_intruder(observation)
            
            # Log hallucination event for research analysis
            self.detector.log_hallucination_event(
                step=self.detector.total_steps,
                magnitude=self.magnitude,
                obs_type="phantom_intruder"
            )
        
        # Update detector statistics
        self.detector.update(hallucination_triggered)
        
        # Add hallucination metadata to info dict for research purposes
        info["hallucinated"] = hallucination_triggered
        info["hallucination_stats"] = self.detector.get_statistics()
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment and hallucination detector"""
        observation, info = self.env.reset(**kwargs)
        
        # Reset detector for new episode
        self.detector.reset_episode()
        
        # Add initial hallucination metadata
        info["hallucinated"] = False
        info["hallucination_stats"] = self.detector.get_statistics()
        
        return observation, info
    
    def set_hallucination_params(self, p_halluc: Optional[float] = None, 
                                magnitude: Optional[float] = None):
        """
        Dynamically adjust hallucination parameters during research
        
        Args:
            p_halluc: New hallucination probability (0.0-1.0)
            magnitude: New magnitude scaling factor (1.0+)
        """
        if p_halluc is not None:
            self.p_halluc = max(0.0, min(1.0, p_halluc))
        if magnitude is not None:
            self.magnitude = max(1.0, magnitude)
    
    def get_detector(self) -> HallucinationDetector:
        """Access hallucination detector for research analysis"""
        return self.detector

## bluesky_gym_hallucination/detector.py

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class HallucinationEvent:
    """Data class for individual hallucination events"""
    step: int
    magnitude: float
    obs_type: str
    timestamp: float = field(default_factory=lambda: 0.0)

class HallucinationDetector:
    """
    Analytics class for tracking and analyzing hallucination injection patterns.
    
    This detector provides essential metrics for quantifying the relationship
    between hallucination frequency/magnitude and ML model safety performance,
    supporting the core research objective of safety margin quantification.
    """
    
    def __init__(self):
        """Initialize hallucination detection and logging system"""
        self.reset()
    
    def reset(self):
        """Reset all tracking statistics"""
        self.total_steps = 0
        self.total_episodes = 0
        self.current_episode_steps = 0
        self.hallucination_events: List[HallucinationEvent] = []
        self.episode_stats: List[Dict[str, Any]] = []
        
        # Per-episode tracking
        self.current_episode_hallucinations = 0
        
        # Statistics caches
        self._stats_cache = {}
        self._cache_valid = False
    
    def reset_episode(self):
        """Reset episode-specific counters"""
        # Archive current episode stats
        if self.current_episode_steps > 0:
            self.episode_stats.append({
                'episode': self.total_episodes,
                'steps': self.current_episode_steps,
                'hallucinations': self.current_episode_hallucinations,
                'hallucination_rate': self.current_episode_hallucinations / self.current_episode_steps
            })
        
        # Reset for new episode
        self.total_episodes += 1
        self.current_episode_steps = 0
        self.current_episode_hallucinations = 0
        self._cache_valid = False
    
    def log_hallucination_event(self, step: int, magnitude: float, obs_type: str):
        """
        Log individual hallucination event for detailed analysis
        
        Args:
            step: Global step number when hallucination occurred
            magnitude: Magnitude scaling factor used
            obs_type: Type of observation modified (e.g., 'phantom_intruder')
        """
        event = HallucinationEvent(
            step=step,
            magnitude=magnitude,
            obs_type=obs_type,
            timestamp=step  # Simple timestamp approximation
        )
        self.hallucination_events.append(event)
        self.current_episode_hallucinations += 1
        self._cache_valid = False
    
    def update(self, hallucination_occurred: bool):
        """Update step counters"""
        self.total_steps += 1
        self.current_episode_steps += 1
        
        if not hallucination_occurred:
            self._cache_valid = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive hallucination statistics for research analysis
        
        Returns:
            Dictionary containing key metrics for safety margin quantification
        """
        if self._cache_valid and self._stats_cache:
            return self._stats_cache
        
        stats = {
            # Basic counters
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'total_hallucinations': len(self.hallucination_events),
            
            # Current episode metrics
            'current_episode_steps': self.current_episode_steps,
            'current_episode_hallucinations': self.current_episode_hallucinations,
        }
        
        if self.total_steps > 0:
            stats['overall_hallucination_rate'] = len(self.hallucination_events) / self.total_steps
        else:
            stats['overall_hallucination_rate'] = 0.0
        
        if self.current_episode_steps > 0:
            stats['current_episode_rate'] = self.current_episode_hallucinations / self.current_episode_steps
        else:
            stats['current_episode_rate'] = 0.0
        
        # Advanced analytics if sufficient data exists
        if len(self.hallucination_events) > 0:
            magnitudes = [event.magnitude for event in self.hallucination_events]
            stats.update({
                'magnitude_stats': {
                    'mean': float(np.mean(magnitudes)),
                    'std': float(np.std(magnitudes)),
                    'min': float(np.min(magnitudes)),
                    'max': float(np.max(magnitudes))
                }
            })
            
            # Observation type distribution for research insights
            obs_types = defaultdict(int)
            for event in self.hallucination_events:
                obs_types[event.obs_type] += 1
            stats['observation_type_distribution'] = dict(obs_types)
        
        # Episode-based statistics for training analysis
        if len(self.episode_stats) > 0:
            episode_rates = [ep['hallucination_rate'] for ep in self.episode_stats]
            stats['episode_rate_stats'] = {
                'mean': float(np.mean(episode_rates)),
                'std': float(np.std(episode_rates)),
                'recent_episodes': self.episode_stats[-5:] if len(self.episode_stats) >= 5 else self.episode_stats
            }
        
        # Cache results for performance
        self._stats_cache = stats
        self._cache_valid = True
        
        return stats
    
    def get_hallucination_timeline(self) -> List[Dict[str, Any]]:
        """
        Get chronological timeline of all hallucination events
        
        Returns:
            List of hallucination events with metadata for temporal analysis
        """
        return [
            {
                'step': event.step,
                'magnitude': event.magnitude,
                'obs_type': event.obs_type,
                'timestamp': event.timestamp
            }
            for event in self.hallucination_events
        ]
    
    def compute_safety_impact_metrics(self, conflict_data: List[Dict]) -> Dict[str, float]:
        """
        Correlate hallucination events with safety metrics like conflicts
        
        Args:
            conflict_data: List of conflict events with step numbers
            
        Returns:
            Safety impact correlation metrics
        """
        if not self.hallucination_events or not conflict_data:
            return {'correlation': 0.0, 'halluc_conflict_ratio': 0.0}
        
        # Find conflicts that occurred near hallucination events (±5 steps)
        halluc_steps = {event.step for event in self.hallucination_events}
        conflict_steps = {conflict['step'] for conflict in conflict_data}
        
        correlated_conflicts = 0
        for conflict_step in conflict_steps:
            for halluc_step in halluc_steps:
                if abs(conflict_step - halluc_step) <= 5:
                    correlated_conflicts += 1
                    break
        
        return {
            'correlation': correlated_conflicts / len(conflict_data) if conflict_data else 0.0,
            'halluc_conflict_ratio': correlated_conflicts / len(self.hallucination_events),
            'total_correlated_conflicts': correlated_conflicts
        }
