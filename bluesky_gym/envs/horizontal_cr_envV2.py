from baseEnv import BaseEnv,HasIntruders,HasWaypoints
import bluesky as bs
import bluesky_gym.envs.common.functions as fn
from bluesky_gym.envs.common.types import Waypoint
from gymnasium import spaces
import numpy as np





class HorizontalCrEnvV2(HasWaypoints,HasIntruders,BaseEnv):
    #CONFIG of the environment instead of global variables
    num_of_waypoints = 1
    waypoint_distance_min = 100
    waypoint_distance_max = 150
    intrusion_distance = 5  # km
    num_intruders = 5
    
    
    
    def _init_(self,config):
        super()._init_(config)
    
    def _init_ownership(self):
        bs.traf.cre(self.agent_id,actype="A320",acspd=self.agent_speed)
    
    def _spawn_intruder(self, index, intruder_speed, intrusion_distance):
        target_idx = bs.traf.id2idx(self.agent_id)
        dpsi = np.random.randint(45,315)
        cpa = np.random.randint(0,intrusion_distance)
        tlosh = np.random.randint(100,1000)
        bs.traf.creconfs(acid=f'{index}',actype="A320",targetidx=target_idx,dpsi=dpsi,dcpa=cpa,tlosh=tlosh)
        
    
    def _spawn_waypoint(self)-> Waypoint:
        wpt_dis_init = self.np_random.integers(self.waypoint_distance_min,self.waypoint_distance_max)
        wpt_hdg_init = 0
        ac_idx = bs.traf.id2idx(self.agent_id)
        
        wpt_lat,wpt_lon = fn.calculate_new_position(bs.traf.lat[ac_idx],bs.traf.lon[ac_idx],wpt_hdg_init,wpt_dis_init)
        return Waypoint(lat=wpt_lat,lon=wpt_lon,reach=0)
            
    def init_action_observation_spaces(self)->tuple[spaces.Box,spaces.Dict]:
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

        for i in range(NUM_INTRUDERS):
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
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon)
        
            self.waypoint_distance.append(wpt_dis * NM2KM)
            self.wpt_qdr.append(wpt_qdr)

            drift = self.ac_hdg - wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)

            self.drift.append(drift)
            self.cos_drift.append(np.cos(np.deg2rad(drift)))
            self.sin_drift.append(np.sin(np.deg2rad(drift)))

        observation = {
                "intruder_distance": np.array(self.intruder_distance)/WAYPOINT_DISTANCE_MAX,
                "cos_difference_pos": np.array(self.cos_bearing),
                "sin_difference_pos": np.array(self.sin_bearing),
                "x_difference_speed": np.array(self.x_difference_speed)/AC_SPD,
                "y_difference_speed": np.array(self.y_difference_speed)/AC_SPD,
                "waypoint_distance": np.array(self.waypoint_distance)/WAYPOINT_DISTANCE_MAX,
                "cos_drift": np.array(self.cos_drift),
                "sin_drift": np.array(self.sin_drift)
            }
        
        return observation