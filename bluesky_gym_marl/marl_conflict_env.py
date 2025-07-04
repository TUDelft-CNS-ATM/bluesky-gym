## bluesky_gym_marl/marl_conflict_env.py

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
import math
from dataclasses import dataclass
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

@dataclass
class Aircraft:
    """Aircraft state representation for MARL environment"""
    agent_id: str
    position: np.ndarray  # [lat, lon, alt] in degrees and feet
    velocity: np.ndarray  # [heading, speed, vertical_rate] in degrees, knots, ft/min
    destination: np.ndarray  # [lat, lon, alt] target position
    active: bool = True
    last_action: Optional[np.ndarray] = None
    conflicts: Set[str] = None  # Set of conflicting agent IDs
    
    def __post_init__(self):
        if self.conflicts is None:
            self.conflicts = set()

class MARLConflictEnv(ParallelEnv):
    """
    Multi-Agent Parallel Environment for Air Traffic Conflict Resolution
    
    Each aircraft operates as an independent agent with local observations
    and coordinated objectives for system-wide safety and efficiency.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "marl_conflict_v0",
    }
    
    def __init__(self, 
                 num_aircraft: int = 8,
                 airspace_size: float = 20.0,
                 separation_lateral_nm: float = 5.0,
                 separation_vertical_ft: float = 1000.0,
                 max_episode_steps: int = 2000,
                 cooperative_reward: bool = True,
                 sector_based: bool = False,
                 communication_enabled: bool = False,
                 observation_radius_nm: float = 15.0):
        """
        Initialize multi-agent conflict resolution environment
        
        Args:
            num_aircraft: Number of aircraft agents (4-20 recommended)
            airspace_size: Square airspace dimension in nautical miles
            separation_lateral_nm: Minimum lateral separation requirement
            separation_vertical_ft: Minimum vertical separation requirement
            max_episode_steps: Maximum steps per episode
            cooperative_reward: Enable shared responsibility rewards
            sector_based: Group agents by geographic sectors
            communication_enabled: Allow inter-agent message passing
            observation_radius_nm: Local observation range per agent
        """
        super().__init__()
        
        # Environment parameters
        self.num_aircraft = num_aircraft
        self.airspace_size = airspace_size
        self.separation_lateral_nm = separation_lateral_nm
        self.separation_vertical_ft = separation_vertical_ft
        self.max_episode_steps = max_episode_steps
        self.cooperative_reward = cooperative_reward
        self.sector_based = sector_based
        self.communication_enabled = communication_enabled
        self.observation_radius_nm = observation_radius_nm
        
        # Agent management
        self.possible_agents = [f"aircraft_{i}" for i in range(num_aircraft)]
        self.agents = self.possible_agents.copy()
        
        # Environment state
        self.aircraft: Dict[str, Aircraft] = {}
        self.current_step = 0
        self.total_conflicts = 0
        self.episode_conflicts = []
        
        # Sector configuration
        if sector_based:
            self.num_sectors = min(4, (num_aircraft + 3) // 4)  # 2x2 or 1x1 sectors
            self.sector_assignments = self._assign_sectors()
        else:
            self.num_sectors = 1
            self.sector_assignments = {}
        
        # Communication system
        self.message_buffer: Dict[str, List[Dict]] = {}
        self.max_messages_per_agent = 3 if communication_enabled else 0
        
        # Define spaces
        self._setup_observation_action_spaces()
        
        # Performance tracking
        self.episode_stats = {
            'conflicts_per_step': [],
            'total_conflicts': 0,
            'coordination_events': 0,
            'messages_sent': 0
        }
    
    def _setup_observation_action_spaces(self):
        """Define observation and action spaces for all agents"""
        # Observation space per agent:
        # - Own state: [lat, lon, alt, heading, speed, vertical_rate] (6)
        # - Nearest neighbors (max 5): [rel_lat, rel_lon, rel_alt, rel_heading, rel_speed, distance, conflict_flag] (7 each)
        # - Sector info: [sector_id, sector_density] (2)
        # - Communication: received messages (3 × message_size if enabled)
        
        max_neighbors = 5
        message_size = 4 if self.communication_enabled else 0
        obs_size = 6 + max_neighbors * 7 + 2 + self.max_messages_per_agent * message_size
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Action space per agent:
        # - Flight controls: [heading_change, vertical_rate, speed_change] (3)
        # - Communication: [message_type, message_value1, message_value2] (3 if enabled)
        
        action_size = 3 + (3 if self.communication_enabled else 0)
        
        action_low = np.array([-45.0, -3000.0, -0.15] + 
                             ([0, -1.0, -1.0] if self.communication_enabled else []))
        action_high = np.array([45.0, 3000.0, 0.15] + 
                              ([10, 1.0, 1.0] if self.communication_enabled else []))
        
        self.action_space = gym.spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )
        
        # Apply spaces to all agents
        self.observation_spaces = {agent: self.observation_space for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space for agent in self.possible_agents}
    
    def _assign_sectors(self) -> Dict[str, int]:
        """Assign agents to geographic sectors for hierarchical control"""
        assignments = {}
        agents_per_sector = max(1, self.num_aircraft // self.num_sectors)
        
        for i, agent in enumerate(self.possible_agents):
            sector_id = min(i // agents_per_sector, self.num_sectors - 1)
            assignments[agent] = sector_id
        
        return assignments
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment with new multi-agent traffic scenario"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset counters
        self.current_step = 0
        self.total_conflicts = 0
        self.episode_conflicts.clear()
        self.agents = self.possible_agents.copy()
        
        # Initialize aircraft positions
        self.aircraft.clear()
        self._initialize_aircraft_positions()
        
        # Reset communication
        self.message_buffer = {agent: [] for agent in self.agents}
        
        # Reset episode statistics
        self.episode_stats = {
            'conflicts_per_step': [],
            'total_conflicts': 0,
            'coordination_events': 0,
            'messages_sent': 0
        }
        
        # Generate initial observations
        observations = self._get_observations()
        infos = self._get_infos()
        
        return observations, infos
    
    def _initialize_aircraft_positions(self):
        """Initialize aircraft with realistic conflict-prone scenarios"""
        for i, agent_id in enumerate(self.agents):
            if self.sector_based:
                # Position aircraft within assigned sectors
                sector_id = self.sector_assignments[agent_id]
                sector_x = sector_id % 2
                sector_y = sector_id // 2
                
                lat_base = -self.airspace_size/4 + sector_x * self.airspace_size/2
                lon_base = -self.airspace_size/4 + sector_y * self.airspace_size/2
                
                lat = lat_base + np.random.uniform(-self.airspace_size/4, self.airspace_size/4)
                lon = lon_base + np.random.uniform(-self.airspace_size/4, self.airspace_size/4)
            else:
                # Random positioning across entire airspace
                lat = np.random.uniform(-self.airspace_size/2, self.airspace_size/2)
                lon = np.random.uniform(-self.airspace_size/2, self.airspace_size/2)
            
            # Altitude assignment with potential conflicts
            if i < self.num_aircraft // 2:
                # Lower group: potential for vertical conflicts
                alt = np.random.uniform(25000, 30000)
            else:
                # Upper group: may conflict during climbs/descents
                alt = np.random.uniform(30000, 35000)
            
            # Random heading and speed
            heading = np.random.uniform(0, 360)
            speed = np.random.uniform(250, 450)  # knots
            vertical_rate = np.random.uniform(-500, 500)  # ft/min
            
            # Generate destination for goal-oriented behavior
            dest_lat = np.random.uniform(-self.airspace_size/2, self.airspace_size/2)
            dest_lon = np.random.uniform(-self.airspace_size/2, self.airspace_size/2)
            dest_alt = np.random.uniform(25000, 35000)
            
            aircraft = Aircraft(
                agent_id=agent_id,
                position=np.array([lat, lon, alt]),
                velocity=np.array([heading, speed, vertical_rate]),
                destination=np.array([dest_lat, dest_lon, dest_alt])
            )
            
            self.aircraft[agent_id] = aircraft
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one step for all agents simultaneously"""
        self.current_step += 1
        
        # Process actions for all agents
        for agent_id in self.agents:
            if agent_id in actions:
                self._apply_action(agent_id, actions[agent_id])
        
        # Update aircraft positions
        self._update_positions()
        
        # Detect conflicts
        conflicts = self._detect_conflicts()
        self.episode_conflicts.append(conflicts)
        self.total_conflicts += len(conflicts)
        
        # Process communication
        if self.communication_enabled:
            self._process_communications()
        
        # Calculate rewards
        rewards = self._calculate_rewards(conflicts)
        
        # Check termination conditions
        terminations = self._get_terminations()
        truncations = self._get_truncations()
        
        # Generate observations and info
        observations = self._get_observations()
        infos = self._get_infos(conflicts)
        
        # Update episode statistics
        self.episode_stats['conflicts_per_step'].append(len(conflicts))
        self.episode_stats['total_conflicts'] = self.total_conflicts
        
        return observations, rewards, terminations, truncations, infos
    
    def _apply_action(self, agent_id: str, action: np.ndarray):
        """Apply agent action to aircraft state"""
        aircraft = self.aircraft[agent_id]
        
        # Extract flight control actions
        heading_change = np.clip(action[0], -45.0, 45.0)
        vertical_rate = np.clip(action[1], -3000.0, 3000.0)
        speed_change = np.clip(action[2], -0.15, 0.15)
        
        # Update aircraft velocity
        aircraft.velocity[0] = (aircraft.velocity[0] + heading_change) % 360  # heading
        aircraft.velocity[1] *= (1 + speed_change)  # speed
        aircraft.velocity[1] = np.clip(aircraft.velocity[1], 150, 600)  # speed limits
        aircraft.velocity[2] = vertical_rate  # vertical rate
        
        aircraft.last_action = action[:3]
        
        # Process communication action if enabled
        if self.communication_enabled and len(action) >= 6:
            message_type = int(np.clip(action[3], 0, 10))
            message_values = action[4:6]
            
            if message_type > 0:  # 0 = no message
                self._send_message(agent_id, message_type, message_values)
    
    def _update_positions(self):
        """Update aircraft positions based on current velocities"""
        dt = 1.0  # 1 second time step
        
        for aircraft in self.aircraft.values():
            # Current state
            lat, lon, alt = aircraft.position
            heading, speed, vertical_rate = aircraft.velocity
            
            # Convert speed from knots to degrees per second (rough approximation)
            speed_deg_per_sec = (speed * 0.000539957) / 3600  # knots to deg/sec
            
            # Update position
            heading_rad = math.radians(heading)
            lat_change = speed_deg_per_sec * math.cos(heading_rad) * dt
            lon_change = speed_deg_per_sec * math.sin(heading_rad) * dt / math.cos(math.radians(lat))
            alt_change = vertical_rate * dt / 60  # ft/min to ft/sec
            
            aircraft.position[0] += lat_change  # latitude
            aircraft.position[1] += lon_change  # longitude
            aircraft.position[2] += alt_change  # altitude
            
            # Constrain to airspace bounds
            aircraft.position[0] = np.clip(aircraft.position[0], -self.airspace_size/2, self.airspace_size/2)
            aircraft.position[1] = np.clip(aircraft.position[1], -self.airspace_size/2, self.airspace_size/2)
            aircraft.position[2] = np.clip(aircraft.position[2], 20000, 45000)
    
    def _detect_conflicts(self) -> List[Tuple[str, str, float, float]]:
        """
        Detect conflicts between all aircraft pairs
        
        Returns:
            List of (agent1_id, agent2_id, lateral_sep_nm, vertical_sep_ft) for conflicts
        """
        conflicts = []
        agents = list(self.aircraft.keys())
        
        for i, agent1_id in enumerate(agents):
            for agent2_id in agents[i+1:]:
                aircraft1 = self.aircraft[agent1_id]
                aircraft2 = self.aircraft[agent2_id]
                
                # Calculate separations
                lat_sep = self._geodetic_distance_nm(
                    aircraft1.position[0], aircraft1.position[1],
                    aircraft2.position[0], aircraft2.position[1]
                )
                vert_sep = abs(aircraft1.position[2] - aircraft2.position[2])
                
                # Check separation criteria
                if (lat_sep < self.separation_lateral_nm and 
                    vert_sep < self.separation_vertical_ft):
                    conflicts.append((agent1_id, agent2_id, lat_sep, vert_sep))
                    
                    # Update aircraft conflict sets
                    aircraft1.conflicts.add(agent2_id)
                    aircraft2.conflicts.add(agent1_id)
                else:
                    # Clear resolved conflicts
                    aircraft1.conflicts.discard(agent2_id)
                    aircraft2.conflicts.discard(agent1_id)
        
        return conflicts
    
    def _geodetic_distance_nm(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in nautical miles"""
        # Haversine formula
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return 3440.065 * c  # Earth radius in nautical miles
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Generate observations for all agents"""
        observations = {}
        
        for agent_id in self.agents:
            obs = self._get_agent_observation(agent_id)
            observations[agent_id] = obs.astype(np.float32)
        
        return observations
    
    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        """Generate observation for specific agent"""
        aircraft = self.aircraft[agent_id]
        obs_size = self.observation_space.shape[0]
        observation = np.zeros(obs_size)
        
        # Own state [6 elements]
        observation[0:6] = [
            aircraft.position[0],    # latitude
            aircraft.position[1],    # longitude
            aircraft.position[2],    # altitude
            aircraft.velocity[0],    # heading
            aircraft.velocity[1],    # speed
            aircraft.velocity[2]     # vertical rate
        ]
        
        # Find nearest neighbors
        neighbors = self._get_nearest_neighbors(agent_id, max_neighbors=5)
        
        # Neighbor observations [5 × 7 = 35 elements]
        for i, (neighbor_id, distance) in enumerate(neighbors):
            if i >= 5:
                break
            
            base_idx = 6 + i * 7
            neighbor = self.aircraft[neighbor_id]
            
            # Relative position and state
            observation[base_idx:base_idx+7] = [
                neighbor.position[0] - aircraft.position[0],  # rel_lat
                neighbor.position[1] - aircraft.position[1],  # rel_lon
                neighbor.position[2] - aircraft.position[2],  # rel_alt
                neighbor.velocity[0],                         # neighbor_heading
                neighbor.velocity[1],                         # neighbor_speed
                distance,                                     # distance_nm
                1.0 if neighbor_id in aircraft.conflicts else 0.0  # conflict_flag
            ]
        
        # Sector information [2 elements]
        sector_idx = 6 + 5 * 7
        if self.sector_based:
            sector_id = self.sector_assignments[agent_id]
            sector_density = self._get_sector_density(sector_id)
            observation[sector_idx:sector_idx+2] = [sector_id, sector_density]
        else:
            observation[sector_idx:sector_idx+2] = [0, len(self.agents) / self.num_aircraft]
        
        # Communication messages [3 × 4 = 12 elements if enabled]
        if self.communication_enabled:
            msg_idx = sector_idx + 2
            messages = self.message_buffer.get(agent_id, [])
            
            for i, msg in enumerate(messages[:self.max_messages_per_agent]):
                msg_base = msg_idx + i * 4
                observation[msg_base:msg_base+4] = [
                    msg['type'], msg['value1'], msg['value2'], msg['urgency']
                ]
        
        return observation
    
    def _get_nearest_neighbors(self, agent_id: str, max_neighbors: int = 5) -> List[Tuple[str, float]]:
        """Get nearest neighbor aircraft within observation radius"""
        aircraft = self.aircraft[agent_id]
        neighbors = []
        
        for other_id, other_aircraft in self.aircraft.items():
            if other_id == agent_id:
                continue
            
            distance = self._geodetic_distance_nm(
                aircraft.position[0], aircraft.position[1],
                other_aircraft.position[0], other_aircraft.position[1]
            )
            
            if distance <= self.observation_radius_nm:
                neighbors.append((other_id, distance))
        
        # Sort by distance and return top k
        neighbors.sort(key=lambda x: x[1])
        return neighbors[:max_neighbors]
    
    def _get_sector_density(self, sector_id: int) -> float:
        """Calculate aircraft density in specific sector"""
        if not self.sector_based:
            return len(self.agents) / self.num_aircraft
        
        sector_agents = [agent for agent, sid in self.sector_assignments.items() 
                        if sid == sector_id and agent in self.agents]
        sector_area = (self.airspace_size / 2) ** 2  # Each sector is quarter of total area
        
        return len(sector_agents) / sector_area
    
    def _calculate_rewards(self, conflicts: List[Tuple]) -> Dict[str, float]:
        """Calculate rewards for all agents"""
        rewards = {agent_id: 0.0 for agent_id in self.agents}
        
        for agent_id in self.agents:
            aircraft = self.aircraft[agent_id]
            
            # Individual conflict penalty
            agent_conflicts = sum(1 for c in conflicts if agent_id in c[:2])
            rewards[agent_id] -= agent_conflicts * 10.0
            
            # Efficiency penalty for large actions
            if aircraft.last_action is not None:
                action_penalty = -0.1 * np.sum(np.abs(aircraft.last_action))
                rewards[agent_id] += action_penalty
            
            # Goal progress reward
            dest_distance = self._geodetic_distance_nm(
                aircraft.position[0], aircraft.position[1],
                aircraft.destination[0], aircraft.destination[1]
            )
            
            # Reward for getting closer to destination
            if hasattr(aircraft, '_last_dest_distance'):
                progress = aircraft._last_dest_distance - dest_distance
                rewards[agent_id] += progress * 0.1
            aircraft._last_dest_distance = dest_distance
            
            # Separation maintenance bonus
            if len(aircraft.conflicts) == 0:
                rewards[agent_id] += 0.2
        
        # Cooperative rewards
        if self.cooperative_reward:
            system_penalty = -len(conflicts) * 5.0
            cooperative_bonus = system_penalty / len(self.agents)
            
            for agent_id in self.agents:
                rewards[agent_id] += cooperative_bonus
        
        return rewards
    
    def _send_message(self, sender_id: str, message_type: int, values: np.ndarray):
        """Send message to nearby agents"""
        sender = self.aircraft[sender_id]
        
        # Find agents within communication range
        comm_range = self.observation_radius_nm * 0.8  # Slightly less than observation range
        
        for receiver_id, other_aircraft in self.aircraft.items():
            if receiver_id == sender_id:
                continue
            
            distance = self._geodetic_distance_nm(
                sender.position[0], sender.position[1],
                other_aircraft.position[0], other_aircraft.position[1]
            )
            
            if distance <= comm_range:
                # Add message to receiver's buffer
                message = {
                    'sender': sender_id,
                    'type': message_type,
                    'value1': values[0],
                    'value2': values[1],
                    'urgency': 1.0 if len(sender.conflicts) > 0 else 0.5
                }
                
                if receiver_id not in self.message_buffer:
                    self.message_buffer[receiver_id] = []
                
                self.message_buffer[receiver_id].append(message)
                
                # Keep only recent messages
                if len(self.message_buffer[receiver_id]) > self.max_messages_per_agent:
                    self.message_buffer[receiver_id] = self.message_buffer[receiver_id][-self.max_messages_per_agent:]
        
        self.episode_stats['messages_sent'] += 1
    
    def _process_communications(self):
        """Process and decay message buffers"""
        # Clear old messages (simple decay mechanism)
        for agent_id in self.message_buffer:
            if len(self.message_buffer[agent_id]) > 0:
                # Remove oldest message with some probability
                if np.random.random() < 0.3:
                    self.message_buffer[agent_id].pop(0)
    
    def _get_terminations(self) -> Dict[str, bool]:
        """Check if any agents have terminated"""
        terminations = {}
        
        for agent_id in self.agents:
            aircraft = self.aircraft[agent_id]
            
            # Check if reached destination
            dest_distance = self._geodetic_distance_nm(
                aircraft.position[0], aircraft.position[1],
                aircraft.destination[0], aircraft.destination[1]
            )
            
            # Agent terminates if very close to destination or out of bounds
            terminated = (dest_distance < 1.0 or  # Within 1 NM of destination
                         abs(aircraft.position[0]) > self.airspace_size/2 or  # Out of bounds
                         abs(aircraft.position[1]) > self.airspace_size/2 or
                         aircraft.position[2] < 15000 or aircraft.position[2] > 50000)
            
            terminations[agent_id] = terminated
        
        return terminations
    
    def _get_truncations(self) -> Dict[str, bool]:
        """Check if episode should be truncated"""
        truncated = self.current_step >= self.max_episode_steps
        return {agent_id: truncated for agent_id in self.agents}
    
    def _get_infos(self, conflicts: Optional[List] = None) -> Dict[str, Dict]:
        """Generate info dictionaries for all agents"""
        if conflicts is None:
            conflicts = []
        
        infos = {}
        
        for agent_id in self.agents:
            aircraft = self.aircraft[agent_id]
            
            infos[agent_id] = {
                'conflicts': list(aircraft.conflicts),
                'num_conflicts': len(aircraft.conflicts),
                'position': aircraft.position.tolist(),
                'destination_distance': self._geodetic_distance_nm(
                    aircraft.position[0], aircraft.position[1],
                    aircraft.destination[0], aircraft.destination[1]
                ),
                'sector_id': self.sector_assignments.get(agent_id, 0) if self.sector_based else 0,
                'step': self.current_step
            }
        
        # Add global info to first agent
        if self.agents:
            first_agent = self.agents[0]
            infos[first_agent].update({
                'global_conflicts': len(conflicts),
                'total_conflicts': self.total_conflicts,
                'episode_stats': self.episode_stats.copy()
            })
        
        return infos
    
    def render(self, mode: str = "human"):
        """Render the environment (basic implementation)"""
        if mode == "human":
            print(f"Step {self.current_step}: {len(self.aircraft)} aircraft")
            
            # Count current conflicts
            conflicts = self._detect_conflicts()
            print(f"Current conflicts: {len(conflicts)}")
            
            # Show aircraft positions
            for agent_id, aircraft in self.aircraft.items():
                pos = aircraft.position
                vel = aircraft.velocity
                print(f"  {agent_id}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.0f}) "
                      f"vel=({vel[0]:.0f}°, {vel[1]:.0f}kt, {vel[2]:.0f}fpm)")
        
        return None
    
    def close(self):
        """Clean up environment resources"""
        self.aircraft.clear()
        self.message_buffer.clear()

## bluesky_gym_marl/utils.py

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Callable, Optional, Union
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import supersuit as ss
from pettingzoo.utils import parallel_to_aec


def make_sb3_env(env_id_or_env: Union[str, gym.Env], 
                 num_envs: int = 1,
                 monitor_wrapper: bool = True,
                 **env_kwargs) -> DummyVecEnv:
    """
    Create Stable-Baselines3 compatible vectorized environment from MARL env
    
    Args:
        env_id_or_env: Environment ID string or environment instance
        num_envs: Number of parallel environments
        monitor_wrapper: Whether to wrap with Monitor for logging
        **env_kwargs: Additional environment arguments
        
    Returns:
        Vectorized environment compatible with SB3
    """
    
    def make_env(rank: int = 0):
        def _init():
            if isinstance(env_id_or_env, str):
                env = gym.make(env_id_or_env, **env_kwargs)
            else:
                env = env_id_or_env
            
            # Convert PettingZoo parallel env to single-agent format
            # This wrapper treats all agents as a single "super-agent"
            env = ss.pettingzoo_env_to_vec_env_v1(env)
            env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
            
            if monitor_wrapper:
                env = Monitor(env, filename=None, allow_early_resets=True)
            
            return env
        return _init
    
    if num_envs == 1:
        return DummyVecEnv([make_env()])
    else:
        return SubprocVecEnv([make_env(i) for i in range(num_envs)])


def make_individual_agent_env(base_env, agent_id: str) -> gym.Env:
    """
    Extract single agent environment from multi-agent environment
    
    Args:
        base_env: Multi-agent PettingZoo environment
        agent_id: Specific agent to extract
        
    Returns:
        Single-agent Gymnasium environment
    """
    
    class SingleAgentWrapper(gym.Env):
        def __init__(self, marl_env, target_agent):
            self.marl_env = marl_env
            self.target_agent = target_agent
            
            # Use target agent's spaces
            self.observation_space = marl_env.observation_spaces[target_agent]
            self.action_space = marl_env.action_spaces[target_agent]
        
        def reset(self, **kwargs):
            obs_dict, info_dict = self.marl_env.reset(**kwargs)
            return obs_dict[self.target_agent], info_dict[self.target_agent]
        
        def step(self, action):
            # Create action dict with random actions for other agents
            action_dict = {}
            for agent in self.marl_env.agents:
                if agent == self.target_agent:
                    action_dict[agent] = action
                else:
                    action_dict[agent] = self.marl_env.action_space.sample()
            
            obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self.marl_env.step(action_dict)
            
            return (obs_dict[self.target_agent], 
                   reward_dict[self.target_agent],
                   term_dict[self.target_agent],
                   trunc_dict[self.target_agent],
                   info_dict[self.target_agent])
        
        def render(self, **kwargs):
            return self.marl_env.render(**kwargs)
        
        def close(self):
            return self.marl_env.close()
    
    return SingleAgentWrapper(base_env, agent_id)


class RewardNormalizer:
    """Normalize rewards across agents for stable multi-agent training"""
    
    def __init__(self, alpha: float = 0.99, epsilon: float = 1e-8):
        self.alpha = alpha
        self.epsilon = epsilon
        self.running_mean = {}
        self.running_var = {}
        self.count = {}
    
    def normalize(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Normalize rewards using running statistics"""
        normalized_rewards = {}
        
        for agent_id, reward in rewards.items():
            if agent_id not in self.running_mean:
                self.running_mean[agent_id] = 0.0
                self.running_var[agent_id] = 1.0
                self.count[agent_id] = 0
            
            # Update running statistics
            self.count[agent_id] += 1
            delta = reward - self.running_mean[agent_id]
            self.running_mean[agent_id] += delta / self.count[agent_id]
            
            if self.count[agent_id] > 1:
                delta2 = reward - self.running_mean[agent_id]
                self.running_var[agent_id] = (
                    (self.count[agent_id] - 2) * self.running_var[agent_id] + delta * delta2
                ) / (self.count[agent_id] - 1)
            
            # Normalize reward
            std = np.sqrt(self.running_var[agent_id] + self.epsilon)
            normalized_rewards[agent_id] = (reward - self.running_mean[agent_id]) / std
        
        return normalized_rewards


def evaluate_marl_performance(env, policy_dict: Dict[str, Any], 
                             episodes: int = 10, 
                             verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate multi-agent policy performance
    
    Args:
        env: Multi-agent environment
        policy_dict: Dictionary mapping agent IDs to policies
        episodes: Number of evaluation episodes
        verbose: Whether to print progress
        
    Returns:
        Performance metrics dictionary
    """
    
    episode_rewards = {agent: [] for agent in env.possible_agents}
    episode_conflicts = []
    episode_lengths = []
    coordination_events = []
    
    for episode in range(episodes):
        obs_dict, _ = env.reset()
        episode_reward = {agent: 0.0 for agent in env.agents}
        episode_length = 0
        total_conflicts = 0
        
        while env.agents:
            # Get actions from policies
            actions = {}
            for agent in env.agents:
                if agent in policy_dict:
                    action, _ = policy_dict[agent].predict(obs_dict[agent])
                    actions[agent] = action
                else:
                    actions[agent] = env.action_space.sample()
            
            # Execute step
            obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(actions)
            
            # Accumulate rewards
            for agent in reward_dict:
                episode_reward[agent] += reward_dict[agent]
            
            # Track conflicts
            if env.agents and 'global_conflicts' in info_dict.get(env.agents[0], {}):
                total_conflicts += info_dict[env.agents[0]]['global_conflicts']
            
            episode_length += 1
            
            # Check if all agents terminated/truncated
            if all(term_dict.values()) or all(trunc_dict.values()):
                break
        
        # Store episode metrics
        for agent, reward in episode_reward.items():
            episode_rewards[agent].append(reward)
        
        episode_conflicts.append(total_conflicts)
        episode_lengths.append(episode_length)
        
        if verbose and (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}/{episodes}: "
                  f"Avg reward: {np.mean([np.mean(rewards) for rewards in episode_rewards.values()]):.2f}, "
                  f"Conflicts: {total_conflicts}")
    
    # Calculate performance metrics
    results = {
        'mean_episode_reward': {agent: np.mean(rewards) for agent, rewards in episode_rewards.items()},
        'std_episode_reward': {agent: np.std(rewards) for agent, rewards in episode_rewards.items()},
        'mean_conflicts_per_episode': np.mean(episode_conflicts),
        'mean_episode_length': np.mean(episode_lengths),
        'total_episodes': episodes,
        'success_rate': np.mean([conflicts == 0 for conflicts in episode_conflicts])
    }
    
    return results


def group_agents_by_sector(env, agent_observations: Dict[str, np.ndarray]) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Group agent observations by sector for hierarchical training
    
    Args:
        env: Multi-agent environment with sector_based=True
        agent_observations: Dictionary of agent observations
        
    Returns:
        Dictionary mapping sector IDs to agent observation dictionaries
    """
    if not hasattr(env, 'sector_assignments'):
        return {0: agent_observations}
    
    sector_groups = {}
    
    for agent_id, obs in agent_observations.items():
        sector_id = env.sector_assignments.get(agent_id, 0)
        
        if sector_id not in sector_groups:
            sector_groups[sector_id] = {}
        
        sector_groups[sector_id][agent_id] = obs
    
    return sector_groups


class MALRLLogger:
    """Logging utility for multi-agent RL experiments"""
    
    def __init__(self, log_dir: str = "marl_logs"):
        self.log_dir = log_dir
        self.episode_data = []
        self.step_data = []
    
    def log_episode(self, episode: int, rewards: Dict[str, float], 
                   conflicts: int, episode_length: int, **kwargs):
        """Log episode-level metrics"""
        episode_info = {
            'episode': episode,
            'total_reward': sum(rewards.values()),
            'individual_rewards': rewards,
            'conflicts': conflicts,
            'episode_length': episode_length,
            **kwargs
        }
        self.episode_data.append(episode_info)
    
    def log_step(self, step: int, observations: Dict, actions: Dict, 
                 rewards: Dict, conflicts: int, **kwargs):
        """Log step-level metrics"""
        step_info = {
            'step': step,
            'num_agents': len(observations),
            'total_reward': sum(rewards.values()),
            'conflicts': conflicts,
            **kwargs
        }
        self.step_data.append(step_info)
    
    def save_logs(self, filename: Optional[str] = None):
        """Save logged data to file"""
        import json
        import os
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        if filename is None:
            filename = f"marl_experiment_{len(self.episode_data)}_episodes.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        data = {
            'episodes': self.episode_data,
            'steps': self.step_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved MARL logs to {filepath}")


# RLLib integration utilities
try:
    import ray
    from ray.rllib.env import PettingZooEnv
    
    class RLLibWrapper(PettingZooEnv):
        """Wrapper for RLLib compatibility"""
        
        def __init__(self, env):
            # Convert to AEC format for RLLib
            aec_env = parallel_to_aec(env)
            super().__init__(aec_env)
        
        @staticmethod
        def make_rllib_config(env_config: Dict[str, Any]) -> Dict[str, Any]:
            """Generate RLLib configuration for MARL training"""
            return {
                "env": "MARLConflictEnv-v0",
                "env_config": env_config,
                "multiagent": {
                    "policies": {
                        "shared_policy": (None, None, None, {})
                    },
                    "policy_mapping_fn": lambda agent_id, episode, **kwargs: "shared_policy",
                },
                "framework": "torch",
                "num_workers": 4,
                "num_envs_per_worker": 1,
                "rollout_fragment_length": 200,
                "train_batch_size": 4000,
                "lr": 3e-4,
                "gamma": 0.99,
                "lambda": 0.95,
                "clip_param": 0.2,
                "vf_clip_param": 10.0,
                "entropy_coeff": 0.01,
            }

except ImportError:
    # RLLib not available
    class RLLibWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("RLLib not installed. Install with: pip install ray[rllib]")
        
        @staticmethod
        def make_rllib_config(*args, **kwargs):
            raise ImportError("RLLib not installed. Install with: pip install ray[rllib]")
