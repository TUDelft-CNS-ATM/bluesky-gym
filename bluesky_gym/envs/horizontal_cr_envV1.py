from bluesky_gym.envs.baseEnv import BaseEnv,HasIntruders,HasWaypoints
import bluesky as bs
import bluesky_gym.envs.common.functions as fn
from bluesky_gym.envs.common.visualizer import Visualizer
from bluesky_gym.envs.common.types import Waypoint
from bluesky_gym.envs.common.constants import NM2KM
from gymnasium import spaces
import numpy as np






class HorizontalCrEnv(HasWaypoints,HasIntruders,BaseEnv):
    #CONFIG of the environment instead of global variables
    num_of_waypoints = 1
    waypoint_distance_min = 100
    waypoint_distance_max = 150
    intrusion_distance = 5  # km
    num_intruders = 5
    max_distance = 200 # width of screen in km
    drift_penalty = -0.1  # Penalty per radian of drift
    delta_heading = 45 # Max heading change in degrees
    distance_margin = 5 #km
    sim_speed = 5
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    
    
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        #override screen size for this env
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment
        
        self.visualizer = Visualizer(self._projection_transform_to_x_y_distance)
        
    def _get_action(self,action):
        action = self.ac_hdg + action * self.delta_heading

        bs.stack.stack(f"HDG KL001 {action[0]}")
    
    def _get_reward(self)->float:
        drift_penalty = self._check_drift()
        reach_reward = self._check_waypoint()
        intrusion_penalty = self._check_horizontal_intrusion()
        total_reward = drift_penalty + reach_reward + intrusion_penalty
        
        self.total_reward += total_reward
        return total_reward
    
    def _is_terminated(self):
        # for this its simple if waypoint 0 is reacted
        if self.waypoints[0].reach == 1:
            return True
    
    def _is_truncated(self):
        # for this env we dont have truncation
        return False        
    
    def _get_info(self):
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # for now just have 10, because it crashed if I gave none for some reason.
        return {
            'total_reward': self.total_reward,
            'total_intrusions': self.total_intrusions,
            'average_drift': self.average_drift.mean()
        }
        
    def _check_drift(self):
        drift = abs(np.deg2rad(self.drift[0]))
        self.average_drift = np.append(self.average_drift, drift)
        return drift * self.drift_penalty
    
    def _init_ownship(self):
        bs.traf.cre(self.agent_id,actype="A320",acspd=self.agent_speed)
    
    def _spawn_intruder(self, index, intruder_speed, intrusion_distance):
        target_idx = bs.traf.id2idx(self.agent_id)
        dpsi = self.np_random.integers(45,315)
        cpa = self.np_random.integers(0,intrusion_distance)
        tlosh = self.np_random.integers(100,1000)
        bs.traf.creconfs(acid=f'{index}',actype="A320",targetidx=target_idx,dpsi=dpsi,dcpa=cpa,tlosh=tlosh)
        
    
    def _create_waypoint(self)-> Waypoint:
        wpt_dis_init = self.np_random.integers(self.waypoint_distance_min,self.waypoint_distance_max)
        wpt_hdg_init = 0
        ac_idx = bs.traf.id2idx(self.agent_id)
        
        wpt_lat, wpt_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)   
        return Waypoint(lat=wpt_lat,lon=wpt_lon,reach=0)
            
    def _init_action_observation_spaces(self)->tuple[spaces.Box,spaces.Dict]:
        observation_space = spaces.Dict(
            {
                "intruder_distance": spaces.Box(-np.inf, np.inf, shape = (self.num_intruders,), dtype=np.float64),
                "cos_difference_pos": spaces.Box(-np.inf, np.inf, shape = (self.num_intruders,), dtype=np.float64),
                "sin_difference_pos": spaces.Box(-np.inf, np.inf, shape = (self.num_intruders,), dtype=np.float64),
                "x_difference_speed": spaces.Box(-np.inf, np.inf, shape = (self.num_intruders,), dtype=np.float64),
                "y_difference_speed": spaces.Box(-np.inf, np.inf, shape = (self.num_intruders,), dtype=np.float64),
                "waypoint_distance": spaces.Box(-np.inf, np.inf, shape = (self.num_intruders,), dtype=np.float64),
                "cos_drift": spaces.Box(-np.inf, np.inf, shape = (self.num_of_waypoints,), dtype=np.float64),
                "sin_drift": spaces.Box(-np.inf, np.inf, shape = (self.num_of_waypoints,), dtype=np.float64)
            }
        )
        action_space = self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)
        return action_space, observation_space
    
    def _get_observation(self):
        ac_idx = bs.traf.id2idx(self.agent_id)

        self.intruder_distance = []
        self.cos_bearing = []
        self.sin_bearing = []
        self.x_difference_speed = []
        self.y_difference_speed = []

        self.waypoint_distance = []
        self.wpt_qdr = []
        self.cos_drift = []
        self.sin_drift = []
        self.drift = []

        self.ac_hdg = bs.traf.hdg[ac_idx]

        for i in range(self.num_intruders):
            int_idx = i+1
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
        
            self.intruder_distance.append(int_dis * NM2KM)

            bearing = self.ac_hdg - int_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.sin_bearing.append(np.sin(np.deg2rad(bearing)))

            heading_difference = bs.traf.hdg[ac_idx] - bs.traf.hdg[int_idx]
            x_dif = - np.cos(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]
            y_dif = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]

            self.x_difference_speed.append(x_dif)
            self.y_difference_speed.append(y_dif)


        for waypoint in self.waypoints:
            
            self.ac_hdg = bs.traf.hdg[ac_idx]
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], waypoint.lat, waypoint.lon)
            waypoint.distance = wpt_dis * NM2KM
            self.waypoint_distance.append(wpt_dis * NM2KM)
            self.wpt_qdr.append(wpt_qdr)

            drift = self.ac_hdg - wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)

            self.drift.append(drift)
            self.cos_drift.append(np.cos(np.deg2rad(drift)))
            self.sin_drift.append(np.sin(np.deg2rad(drift)))

        observation = {
                "intruder_distance": np.array(self.intruder_distance)/self.waypoint_distance_max,
                "cos_difference_pos": np.array(self.cos_bearing),
                "sin_difference_pos": np.array(self.sin_bearing),
                "x_difference_speed": np.array(self.x_difference_speed)/self.agent_speed,
                "y_difference_speed": np.array(self.y_difference_speed)/self.agent_speed,
                "waypoint_distance": np.array(self.waypoint_distance)/self.waypoint_distance_max,
                "cos_drift": np.array(self.cos_drift),
                "sin_drift": np.array(self.sin_drift)
            }
        
        return observation
    
    def _reset_instance_variables(self):
        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])
    
    def _projection_transform_to_x_y_distance(self,lat,lon,alt=None)->tuple[int,int,float]:
        #For this env its agent centric projection
        
        ac_idx = bs.traf.id2idx(self.agent_id)
        int_qdr,int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat,lon)
        x_pos = (self.window_width/2)+(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/self.max_distance)*self.window_width
        y_pos = (self.window_height/2)-(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/self.max_distance)*self.window_height
        return x_pos,y_pos,int_dis
    
    def _render_world(self):
        agent_lat = bs.traf.lat[bs.traf.id2idx(self.agent_id)]
        agent_lon = bs.traf.lon[bs.traf.id2idx(self.agent_id)]
        agent_hdg = bs.traf.hdg[bs.traf.id2idx(self.agent_id)]
        self.visualizer.draw_horizontal_ownship(self.canvas,agent_lat,agent_lon,agent_hdg,aircraft_legth_km=8,heading_length_km=50,box_width=4,heading_width=4)
        self.visualizer.draw_all_horizontal_intruders(self.canvas,self.num_intruders,self.intrusion_distance,(self.intrusion_distance/self.max_distance)*self.window_width,heading_length_km=10,aircraft_length=3)
        self.visualizer.draw_horizontal_wayoints(self.canvas,self.waypoints,outer_radius=(self.distance_margin/self.max_distance)*self.window_width)
   