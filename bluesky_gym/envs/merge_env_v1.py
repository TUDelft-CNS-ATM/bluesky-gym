"""
This table enumerates the observation space for 3 neigbours in observation space,
if number of neighbours included is changed, indices will change aswell (of course):

| Index: [start, end) | Description                                                  |   Values    |
|:-----------------:|------------------------------------------------------------|:---------------:|
|          0          | cos(track deviation)                                         | [-inf, inf] |
|          1          | sin(track deviation)                                         | [-inf, inf] |
|          2          | airspeed                                                     | [-inf, inf] |
|          3          | final approach fix distance                                  | [-inf, inf] |
|          4          | faf reached boolean                                          |    (0,1)    |
|         5-7         | Relative X postions with 3 closest neighbours                | [-inf, inf] |
|         8-10        | Relative Y postions with 3 closest neighbours                | [-inf, inf] |
|        11-13        | Relative X velocity with 3 closest neighbours                | [-inf, inf] |
|        14-16        | Relative Y velocity with 3 closest neighbours                | [-inf, inf] |
|        17-19        | cos(heading difference) with 3 closest neighbours            | [-inf, inf] |
|        20-22        | sin(heading difference) with 3 closest neighbours            | [-inf, inf] |
|        23-25        | distance with 3 closest neighbours                           | [-inf, inf] |

"""

import functools
from pettingzoo import ParallelEnv
import gymnasium as gym
import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy

import bluesky_gym.envs.common.functions as fn
import random

DISTANCE_MARGIN = 20 # km
REACH_REWARD = 0#1

DRIFT_PENALTY = -0.02 #0.1
INTRUSION_PENALTY = -0.2#-0.2 #1

INTRUSION_DISTANCE = 4 # NM

SPAWN_DISTANCE_MIN = 50
SPAWN_DISTANCE_MAX = 200

INTRUDER_DISTANCE_MIN = 500
INTRUDER_DISTANCE_MAX = 700

D_HEADING = 3 #15
D_SPEED = 4 #20 

AC_SPD = 150

NM2KM = 1.852
MpS2Kt = 1.94384

ACTION_FREQUENCY = 10

NUM_AC = 20
NUM_AC_STATE = 5
NUM_WAYPOINTS = 1


TARGET_LAT = 52.36239301495972
TARGET_LON = 4.713195734579777

class MergeEnvMulti(ParallelEnv):
    """ 
    Multi-agent arrival manager environment - all aircraft are required to merge into a single traffic stream.
    """
    metadata = {
        "name": "merge_v1",
        "render_modes": ["rgb_array","human"],
         "render_fps": 120}

    def __init__(self, render_mode=None, n_agents=10, time_limit=600):
        self.window_width = 1500
        self.window_height = 1000
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        self.n_agents = n_agents
        self.agents = self._get_agents(self.n_agents)
        self.possible_agents = self.agents[:]

        self.time_limit = time_limit
        self.steps = 0

        self.observation_spaces = {agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4+7*NUM_AC_STATE,), dtype=np.float64) for agent in self.agents}
        self.action_spaces = {agent: gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float64) for agent in self.agents}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        bs.init(mode='sim', detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1;FF')

        # initialize values used for logging -> input in _get_info
        self.total_reward = 0
        self.reward_array = np.array([])
        self.num_episodes = 0
        self.average_drift = np.array([])
        self.total_intrusions = 0

        self.window = None
        self.clock = None
        self.nac = NUM_AC
        self.target_lat = TARGET_LAT
        self.target_lon = TARGET_LON

    def reset(self, seed=None, options=None):
        
        bs.traf.reset()
        self.steps = 0
        self.agents = self.possible_agents[:]

        self.num_episodes += 1
        if self.num_episodes > 1:
            self.reward_array = np.append(self.reward_array, self.total_reward)
            print(self.num_episodes)
            print(self.reward_array[-100:].mean())

        self.total_reward = 0
        self.average_drift = np.array([])
        self.total_intrusions = 0

        self._gen_aircraft()

        observations = self._get_observation()
        infos = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observations, infos
    
    def step(self, actions):

        self._get_action(actions)

        for i in range(ACTION_FREQUENCY):
            bs.sim.step()
            if self.render_mode == "human":
                observations = self._get_observation()
                self._render_frame()


        rewards, dones = self._get_reward()
        observations = self._get_observation()

        if self.time_limit < self.steps:
            time_exceeded = True
        else:
            time_exceeded = False
        trunc = [time_exceeded] * len(self.agents)
        truncates = {
            a: d
            for a,d in zip(self.agents,trunc)
        }

        infos = self._get_info()

        self.steps += 1

        if any(dones.values()) or all(truncates.values()):
            dones = {a: True for a in self.agents}
            self.agents = []

        return observations, rewards, dones, truncates, infos
    
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent] # have to define observation_spaces & action_spaces, probably in init

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_agents(self, n_agents):
        return [f'kl00{i+1}'.upper() for i in range(n_agents)]
    
    def _gen_aircraft(self):
        # Agenten spawnen am rechten Rand (gleiche x-Position), aber y (Breitengrad) ist zufällig
        for agent, idx in zip(self.agents, np.arange(self.n_agents)):
            spawn_dist = random.uniform(INTRUDER_DISTANCE_MIN, INTRUDER_DISTANCE_MAX)
            spawn_angle = 0  # Richtung Osten (rechter Rand)
            # Basisposition am rechten Rand
            base_lat, base_lon = fn.get_point_at_distance(self.target_lat, self.target_lon, spawn_dist, spawn_angle)
            lat_ac = base_lat 
            lon_ac = base_lon + random.uniform(-8, 8)
            heading = 180.0 
            speed = random.uniform(100, 300)
            bs.traf.cre(agent, actype="A320", acspd=speed, aclat=lat_ac, aclon=lon_ac, achdg=heading, acalt=10000)
            bs.stack.stack(f"{agent} addwpt {self.target_lat} {self.target_lon}")
            bs.stack.stack(f"{agent} dest {self.target_lat} {self.target_lon}")
        bs.stack.stack('reso off')
        return

    def _get_observation(self):
        obs = []
        self.target_dist = {a: 0 for a in self.agents}

        for agent in self.agents:
            ac_idx = bs.traf.id2idx(agent)

            cos_drift = np.array([])
            sin_drift = np.array([])
            airspeed = np.array([])
            x_r = np.array([])
            y_r = np.array([])
            vx_r = np.array([])
            vy_r = np.array([])
            cos_track = np.array([])
            sin_track = np.array([])
            distances = np.array([])

            ac_hdg = bs.traf.hdg[ac_idx]
            
            # Get and decompose agent aircaft drift
            wpt_qdr, wpt_dist  = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.target_lat, self.target_lon)

            drift = ac_hdg - wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)

            cos_drift = np.append(cos_drift, np.cos(np.deg2rad(drift)))
            sin_drift = np.append(sin_drift, np.sin(np.deg2rad(drift)))

            self.target_dist[agent] = wpt_dist

            airspeed = np.append(airspeed, bs.traf.tas[ac_idx])
            vx = np.cos(np.deg2rad(ac_hdg)) * bs.traf.tas[ac_idx]
            vy = np.sin(np.deg2rad(ac_hdg)) * bs.traf.tas[ac_idx]
            
            distances = bs.tools.geo.kwikdist_matrix(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat,bs.traf.lon)
            ac_idx_by_dist = np.argsort(distances) # sort aircraft by distance to ownship

            for i in range(self.n_agents):
                int_idx = ac_idx_by_dist[i]
                if int_idx == ac_idx:
                    continue
                int_hdg = bs.traf.hdg[int_idx]
                
                # Intruder AC relative position, m
                brg, dist = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx],bs.traf.lon[int_idx]) 
                x_r = np.append(x_r, (dist * NM2KM * 1000) * np.cos(np.deg2rad(brg)))
                y_r = np.append(y_r, (dist * NM2KM * 1000) * np.sin(np.deg2rad(brg)))
                
                # Intruder AC relative velocity, m/s
                vx_int = np.cos(np.deg2rad(int_hdg)) * bs.traf.tas[int_idx]
                vy_int = np.sin(np.deg2rad(int_hdg)) * bs.traf.tas[int_idx]
                vx_r = np.append(vx_r, vx_int - vx)
                vy_r = np.append(vy_r, vy_int - vy)

                # Intruder AC relative track, rad
                track = np.arctan2(vy_int - vy, vx_int - vx)
                cos_track = np.append(cos_track, np.cos(track))
                sin_track = np.append(sin_track, np.sin(track))

                distances = np.append(distances, distances[ac_idx-1])

            # very crude normalization for the observation vectors
            observation = {
                "cos(drift)": np.array(cos_drift),
                "sin(drift)": np.array(sin_drift),
                "airspeed": np.array(airspeed-150)/6,
                "waypoint_dist": np.array([wpt_dist/250]),
                "x_r": np.array(x_r[:NUM_AC_STATE]/1000000),
                "y_r": np.array(y_r[:NUM_AC_STATE]/1000000),
                "vx_r": np.array(vx_r[:NUM_AC_STATE]/150),
                "vy_r": np.array(vy_r[:NUM_AC_STATE]/150),
                "cos(track)": np.array(cos_track[:NUM_AC_STATE]),
                "sin(track)": np.array(sin_track[:NUM_AC_STATE]),
                "distances": np.array(distances[:NUM_AC_STATE]/250)
            }

            obs.append(np.concatenate(list(observation.values())))
        
        observations = {
            a: o
            for a, o in zip(self.agents, obs)
        }
        
        return observations
    
    def _get_info(self):
        # for now just multiply the global agent info, could not be bothered
        return {
            a: {'total_reward': self.total_reward,
            'total_intrusions': self.total_intrusions,
            'average_drift': self.average_drift.mean()}
            for a in self.agents
        }

    def _get_reward(self):
        rew = []
        dones = []
        for agent in self.agents:
            ac_idx = bs.traf.id2idx(agent)
            reach_reward, done = self._check_waypoint(agent)
            drift_reward = self._check_drift(ac_idx, agent)
            intrusion_reward = self._check_intrusion(ac_idx)

            reward = reach_reward + drift_reward + intrusion_reward

            rew.append(reward)
            dones.append(done)
            self.total_reward += reward

        rewards = {
            a: r
            for a, r in zip(self.agents, rew)
        }

        done = {
            a: d
            for a,d in zip(self.agents,dones)
        }

        return rewards, done
        
    def _check_waypoint(self, agent):
        reward = 0
        done = 0
        # Episode endet, wenn Target erreicht ist
        if self.target_dist[agent] < DISTANCE_MARGIN:
            done = 1
        return reward, done

    def _check_drift(self, ac_idx, agent):
        ac_hdg = bs.traf.hdg[ac_idx]
        # Drift immer zum Target
        target_qdr, target_dist  = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.target_lat, self.target_lon)
        drift = ac_hdg - target_qdr
        drift = fn.bound_angle_positive_negative_180(drift)
        drift = abs(np.deg2rad(drift))
        self.average_drift = np.append(self.average_drift, drift)
        return drift * DRIFT_PENALTY

    def _check_intrusion(self, ac_idx):
        reward = 0
        for i in range(self.n_agents):
            int_idx = i
            if i == ac_idx:
                continue
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if int_dis < INTRUSION_DISTANCE:
                self.total_intrusions += 1
                reward += INTRUSION_PENALTY
        
        return reward

    def _get_action(self,actions):
        for agent in self.agents:
            action = actions[agent]
            dh = action[0] * D_HEADING
            heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx(agent)] + dh)
            bs.stack.stack(f"HDG {agent} {heading_new}")

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 1000 # width of screen in km

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235)) 

        # Zielpunkt: Target-Position im ersten linken Viertel
        # Mapping: Lat/Lon aus der Simulation werden auf Fensterkoordinaten gemappt
        # Referenzpunkt ist das Ziel, das im linken Viertel angezeigt wird
        target_x = self.window_width * 0.25
        target_y = self.window_height * 0.5
        circle_x = target_x
        circle_y = target_y

        # Hilfsfunktion: Wandelt Lat/Lon in Fensterkoordinaten um
        def latlon_to_screen(lat, lon):
            # Berechne Distanz und Richtung vom Ziel
            qdr, dist = bs.tools.geo.kwikqdrdist(self.target_lat, self.target_lon, lat, lon)
            # Umrechnung in km
            rel_x = np.cos(np.deg2rad(qdr)) * dist * NM2KM
            rel_y = np.sin(np.deg2rad(qdr)) * dist * NM2KM
            # Mapping: max_distance km nach rechts/oben
            x = circle_x + (rel_x / max_distance) * self.window_width * 0.75
            y = circle_y - (rel_y / max_distance) * self.window_height * 0.75
            return x, y

        pygame.draw.circle(
            canvas, 
            (255,255,255),
            (circle_x,circle_y),
            radius = 4,
            width = 0
        )
        pygame.draw.circle(
            canvas, 
            (255,255,255),
            (circle_x,circle_y),
            radius = (DISTANCE_MARGIN/max_distance)*self.window_width,
            width = 2
        )

        # draw ownship
        for agent in self.agents:
            ac_idx = bs.traf.id2idx(agent)
            ac_length = 8
            # Fensterkoordinaten aus Lat/Lon
            x_pos, y_pos = latlon_to_screen(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])

            separation = bs.tools.geo.kwikdist(np.concatenate((bs.traf.lat[:ac_idx], bs.traf.lat[ac_idx+1:])), 
                                               np.concatenate((bs.traf.lon[:ac_idx], bs.traf.lon[ac_idx+1:])), 
                                               bs.traf.lat[ac_idx], 
                                               bs.traf.lon[ac_idx])

            # Determine color
            if np.any(separation < INTRUSION_DISTANCE):
                color = (220,20,60)
            else: 
                color = (80,80,80)

            # Heading line
            # Berechne Endpunkt der Heading-Linie
            heading_x, heading_y = latlon_to_screen(
                bs.traf.lat[ac_idx] + np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length / NM2KM / 60,
                bs.traf.lon[ac_idx] + np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length / NM2KM / 60
            )
            pygame.draw.line(canvas,
                (0,0,0),
                (x_pos,y_pos),
                (heading_x,heading_y),
                width = 4
            )

            # draw heading line (kleiner)
            heading_length = 10
            heading_x2, heading_y2 = latlon_to_screen(
                bs.traf.lat[ac_idx] + np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length / NM2KM / 60,
                bs.traf.lon[ac_idx] + np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length / NM2KM / 60
            )
            pygame.draw.line(canvas,
                (0,0,0),
                (x_pos,y_pos),
                (heading_x2,heading_y2),
                width = 1
            )

            pygame.draw.circle(
                canvas, 
                color,
                (x_pos,y_pos),
                radius = (INTRUSION_DISTANCE*NM2KM/max_distance)*self.window_width,
                width = 2
            )

        # # draw intruders
        # ac_length = 3

        # for i in range(1,NUM_AC):
        #     int_idx = i
        #     int_hdg = bs.traf.hdg[int_idx]
        #     heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width
        #     heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width

        #     int_qdr, int_dis = bs.tools.geo.kwikqdrdist(self.wpt_lat, self.wpt_lon, bs.traf.lat[int_idx], bs.traf.lon[int_idx])

        #     # determine color
        #     if int_dis < INTRUSION_DISTANCE:
        #         color = (220,20,60)
        #     else: 
        #         color = (80,80,80)
        #     if i==0:
        #         color = (252, 43, 28)

        #     x_pos = (circle_x)+(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_width
        #     y_pos = (circle_y)-(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_height

        #     pygame.draw.line(canvas,
        #         color,
        #         (x_pos,y_pos),
        #         ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
        #         width = 4
        #     )

        #     # draw heading line
        #     heading_length = 10
        #     heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.window_width
        #     heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.window_width

        #     pygame.draw.line(canvas,
        #         color,
        #         (x_pos,y_pos),
        #         ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
        #         width = 1
        #     )

        #     pygame.draw.circle(
        #         canvas, 
        #         color,
        #         (x_pos,y_pos),
        #         radius = (INTRUSION_DISTANCE*NM2KM/max_distance)*self.window_width,
        #         width = 2
        #     )

        # PyGame update
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

        # Event-loop for MacOS to recognise window as active
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        pygame.quit()