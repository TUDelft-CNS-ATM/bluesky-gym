import gymnasium as gym
import numpy as np
import pygame
from typing import ClassVar
import bluesky as bs
from gymnasium import spaces
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
from bluesky_gym.envs.common.types import Waypoint

class BaseEnv(gym.Env):
    """ 
    Base environment class for BlueSky Gym environments.
    All BlueSky Gym environments should inherit from this class.
    """
    # Define class variables which are common to all environments
    action_frequency: ClassVar[int] = 10  # Frequency of actions for the bluesky simulator
    sim_speed: ClassVar[int] = 1  # Simulation speed for the bluesky simulator
    agent_speed: ClassVar[int] = 150  # Default aircraft speed in knots
    agent_id: ClassVar[str] = "KL001"  # Default aircraft ID for the own ship

    def __init__(self,window_size=(800, 600),render_mode=None,workdir=None):
        super().__init__()
        assert render_mode in (None, "human"), "Invalid render mode"
        self.under_lays = []
        self.over_lays = []
        self.window_size = window_size
        self.window_width, self.window_height = window_size
        self.canvas = None
        self.own_ship_idx = None
        self.episode_init = []
        self.total_reward = 0
        self.action_space,self.observation_space = self.init_action_observation_spaces()
        # initialize bluesky as non-networked simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True,workdir=workdir)
            
        # Initialize dummy screen and set correct simulation speed
        bs.src = ScreenDummy()
        bs.stack.stack(f"DT {self.sim_speed};FF")
        
        self.window = None
        self.clock = None
        
        
        
        


        # Common initialization code for all BlueSky Gym environments can go here
        
    def init_action_observation_spaces(self)->tuple[spaces.Box,spaces.Dict]:
        """Initialize the action and observation spaces for the environment.
        This method should be overridden by subclasses to provide specific action and observation spaces.
        """
        raise NotImplementedError("Subclasses must implement this method.")
        #Example:
        action_space = TODO
        observation_space = TODO
        return action_space,observation_space

    
    
        
    def _get_observation(self):
        """Get the current observation of the environment.
        This method should be overridden by subclasses to provide specific observation logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
        
    def init_own_ship(self):
        """Initialize the own ship in the environment.
        This method should be overridden by subclasses to provide specific own ship initialization logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _reset_instance_variables(self):
        """Reset instance variables for a new episode.
        This method should be overridden by subclasses to provide specific reset logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _get_info(self):
        """Get additional info about the current state of the environment.
        This method should be overridden by subclasses to provide specific info logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
        
    def reset(self,seed=None,options=None):
        """Reset the environment to an initial state and return an initial observation.

        Args:
            seed (int, optional): Seed for random number generator. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.
        """
        self._reset_instance_variables()
        super().reset(seed=seed,options=options)
        self._init_ownership()
        self._add_reset()
        
        observation = self._get_observation()
        info = self._get_info()
    
        
        
        return observation, info
    
    def _add_reset(self):
        """Additional reset logic specific to the environment.
        This method can be overridden by subclasses to provide specific additional reset logic.
        Do not call super() since this is the base model and end of the init chain.
        """
        pass
        
        
    def _get_reward(self):
        """Calculate the reward for the current step.
        This method should be overridden by subclasses to provide specific reward logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _is_done(self):
        """Determine if the current episode is done.
        This method should be overridden by subclasses to provide specific termination logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _init_ownership(self):
        """Initialize ownership of the environment.
        This method should be overridden by subclasses to provide specific ownership logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
        
    def _begin_frame(self):
        """ 
        
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        self.canvas = pygame.Surface(self.window_size)
        self.canvas.fill((135,206,235))

    
    def _render_world(self):
        """ 
        Render the world/environment.
        This method should be overridden by subclasses to provide specific world rendering logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _render_overlays(self):
        """ 
        Render overlays on top of the world/environment.
        This method should be overridden by subclasses to provide specific overlay rendering logic.
        """
        pass
    
    def _render_underlays(self):
        """ 
        Render underlays beneath the world/environment.
        This method should be overridden by subclasses to provide specific underlay rendering logic.
        """
        pass
        
    def render(self):
        """Renders the current image or returns a rgb_array needed for gymnasium wrapper ro funciton
        """
        self._begin_frame()
        self._render_underlays()
        self._render_world()
        self._render_overlays()
        
        if self.render_mode == "human":
            self.window.blit(self.canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.action_frequency)
        elif self.render_mode == "rgb_array":
            return np.array(pygame.surfarray.pixels3d(self.canvas).swapaxes(0, 1))
        
    
    def _projection_transform_to_x_y_distance(self,lat,lon,altitude)->tuple[int,int,float]:
        """Convert the position of the bluesky simulator to x and y cooridnates and return the distance from ownship

        Args:
            lat (float): latitude of the aircraft
            lon (float): longitude of the aircraft
            altitude (float): altitude of the aircraft
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    

class HasIntruders:
    """ 
    Mixin class for environments that have intruder aircraft.
    Provides common functionality for handling intruder aircraft.
    """
    
    num_intruders: ClassVar[int] = 5  # Default number of intruder aircraft
    intruder_speed: ClassVar[int] = 150  # Default speed of intruder aircraft in knots
    intrusion_distance: ClassVar[int] = 5  # Default intrusion distance in nautical miles
    
    
    def _init_intruders(self):
        for i in range(1,self.num_intruders+1):
            self._spawn_intruder(i,self.intruder_speed,self.intrusion_distance)
        
    def _spawn_intruder(self,index:int,intruder_speed:int,intrusion_distance:int):
        """Spawn an intruder aircraft in the environment.

        Args:
            index (int): Index of the intruder aircraft.
            intruder_speed (int): Speed of the intruder aircraft in knots.
            intrusion_distance (int): Intrusion distance in nautical miles.
        """
        raise NotImplementedError("Subclasses must implement this method.")
        
    
    def _add_reset(self):
        self._init_intruders()
        super()._add_reset()
        
class HasWaypoints:
    """ 
    Mixin class for environments that have waypoints.
    Provides common functionality for handling waypoints.
    Adds: 
    
        self.waypoints: list[Waypoint]
        class variable: number_of_waypoints int
    """
    num_of_waypoints: ClassVar[int] = 1 # Default number of waypoints
    def _init_(self,config):
        super()._init_(config)
        self.waypoints: list[Waypoint] = []
    
    def _init_waypoints(self):
        """Initialize waypoints in the environment.
        """
        for i in range(self.number_of_waypoints):
            self.waypoints.append(self._create_waypoint())
    
    def _create_waypoint(self)->Waypoint:
        """Create a waypoint in the environment.
        This method should be overridden by subclasses to provide specific waypoint creation logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _add_reset(self):
        self._init_waypoints()
        super()._add_reset()

class HasPolygons:
    """ 
    Mixin class for environments that have polygons.
    Provides common functionality for handling polygons.
    """
    
    number_of_polygons: ClassVar[int] = 1  # Default number of polygons
    
    def _init_polygons(self):
        """Initialize polygons in the environment.
        This method should be overridden by subclasses to provide specific polygon initialization logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _add_reset(self):
        self._init_polygons()
        super()._add_reset()
        
class HasAirport:
    """ 
    Mixin class for environments that have airports.
    Provides common functionality for handling airports.
    """
    
    def _init_airports(self):
        """Initialize airports in the environment.
        This method should be overridden by subclasses to provide specific airport initialization logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _add_reset(self):
        self._init_airports()
        super()._add_reset()


