"""This file creates a visualizer class to draw common objects in the render method.
It gets a call back function to the environment specific projection
"""
import pygame
import bluesky_gym.envs.common.functions as fn
from bluesky_gym.envs.common.types import Waypoint
from bluesky_gym.envs.common.constants import NM2KM
import bluesky as bs

class Visualizer:
    def __init__(self, projection_callback):
        """
        Args:
            projection_callback: Callback function that returns (x, y) coordinates
             def _projection_transform_to_x_y(self,lat,lon,altitude): the projection gets lat,lon,altitude passed and returns x,y coordinats
        """
        self.projection_callback = projection_callback
        
    def draw_horizontal_line(self,canvas, y, color=(255, 0, 0), width=2):
        """Draws a horizontal line across the canvas at the given y coordinate.

        Args:
            canvas: Pygame surface to draw on.
            y: Y coordinate in world space.
            color: Color of the line.
            width: Width of the line.
        """
        start_pos = self.projection_callback((-1000, y))
        end_pos = self.projection_callback((1000, y))
        pygame.draw.line(canvas, color, start_pos, end_pos, width)
        
    def draw_horizontal_ownship(self,canvas,lat,long,hdg,color = (0,0,0),heading_width=1,box_width=4,heading_length_km=1,aircraft_legth_km=3):
        x_pos, y_pos, _ = self.projection_callback(lat, long)
    
        # 1. Draw Aircraft Body (Thick Line) - 3km length matching old env
        # Old env drew from Center -> Outward (Forward), not centered.
        lat_body, lon_body = fn.get_point_at_distance(lat, long, aircraft_legth_km/NM2KM, hdg)
        body_end_x, body_end_y, _ = self.projection_callback(lat_body, lon_body)
        
        pygame.draw.line(canvas,
            color,
            (x_pos, y_pos),
            (body_end_x, body_end_y),
            width=4 # Match old env width
        )

        # 2. Draw Heading Line (Thin Line)
        lat_end, lon_end = fn.get_point_at_distance(lat, long, heading_length_km/NM2KM, hdg) # Ensure conversion if fn expects NM
        heading_end_x, heading_end_y, _ = self.projection_callback(lat_end, lon_end)
        
        pygame.draw.line(canvas,
            color,
            (x_pos, y_pos),
            (heading_end_x, heading_end_y),
            width=1
        )

    def draw_all_horizontal_intruders(self,canvas,number_intruders,intrusion_distance,radius=5,heading_length_km=10,heading_width=2,aircraft_length=3):
        """Adds all intruders to the canvas

        Args:
            canvas (_type_): _description_
            number_intruders (_type_): _description_
            intrusion_distance (_type_): _description_
            agent_idx (_type_): _description_
        """
        for intruder_idx in range(1,number_intruders+1):
            int_lat = bs.traf.lat[intruder_idx]
            int_lon = bs.traf.lon[intruder_idx]
            int_hdg = bs.traf.hdg[intruder_idx]
            x_pos,y_pos,int_dis = self.projection_callback(int_lat,int_lon)
            
            
            if int_dis < intrusion_distance:
                color = (220,20,60)  # Red for intruders within intrusion distance
            else:
                color = (80,80,80)
            
            self.draw_horizontal_single_intruder(canvas,int_lat,int_lon,int_hdg, color=color, radius=radius,heading_length_km=heading_length_km,heading_width=heading_width,aircraft_length=aircraft_length)
    
    def draw_horizontal_single_intruder(self,canvas,lat,long,hdg, color=(0, 0, 255), radius=5,heading_length_km=10,heading_width=2,aircraft_length=3):
        """Adds a single intruder to the canvas

        Args:
            canvas (_type_): _description_
            lat (_type_): _description_
            long (_type_): _description_
            hdg (_type_): _description_
            color (tuple, optional): _description_. Defaults to (0, 0, 255).
            radius (int, optional): _description_. Defaults to 5.
            heading_length (int, optional): _description_. Defaults to 10.
            heading_width (int, optional): _description_. Defaults to 2.
        """
        x_pos, y_pos,_ = self.projection_callback(lat,long)
        lat_end, lon_end = fn.get_point_at_distance(lat, long, heading_length_km, hdg)
        heading_end_x, heading_end_y,_ = self.projection_callback(lat_end, lon_end)
        
        lat_body, lon_body = fn.get_point_at_distance(lat, long, aircraft_length/NM2KM, hdg)
        body_end_x, body_end_y, _ = self.projection_callback(lat_body, lon_body)
        
        pygame.draw.line(canvas,
            color,
            (x_pos, y_pos),
            (body_end_x, body_end_y),
            width=4 # Match old env width
        )
        
        pygame.draw.line(canvas,
            color,
            (x_pos,y_pos),
            (heading_end_x,heading_end_y),
            width = 1
        )

        pygame.draw.circle(
            canvas, 
            color,
            (x_pos,y_pos),
            radius = radius,
            width = 2
        )

    def draw_horizontal_wayoints(self,canvas,waypoints:list[Waypoint],inner_radius=4,outer_radius=6):
        for waypoint in waypoints:
            lat = waypoint.lat
            lon = waypoint.lon
            x_pos, y_pos,_ = self.projection_callback(lat,lon)
            
            if waypoint.reach > 0:
                color = (155,155,155)
            else:
                color = (255,255,255)
                
            pygame.draw.circle(
                canvas, 
                color,
                (x_pos,y_pos),
                radius = 4,
                width = 0
            )
            
            pygame.draw.circle(
                canvas, 
                color,
                (x_pos,y_pos),
                radius = 6,
                width = 2
            )