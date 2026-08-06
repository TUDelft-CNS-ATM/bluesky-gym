import numpy as np
from typing import List
import numpy as np


def bound_angle_positive_negative_180(angle_deg: float) -> float:
    """ maps any angle in degrees to the [-180,180] interval 
    Parameters
    __________
    angle_deg: float
        angle that needs to be mapped (in degrees)
    
    Returns
    __________
    angle_deg: float
        input angle mapped to the interval [-180,180] (in degrees)
    """

    if angle_deg > 180:
        return -(360 - angle_deg)
    elif angle_deg < -180:
        return (360 + angle_deg)
    else:
        return angle_deg

def get_point_at_distance(lat1, lon1, d, bearing, R=6371):
    """
    lat: latitude of the reference point, in degrees
    lon: longitude of the referemce point, in degrees
    d: target distance from the reference point, in km
    bearing: (true) heading, in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from the reference point, in degrees
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    a = np.radians(bearing)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d/R) + np.cos(lat1) * np.sin(d/R) * np.cos(a))
    lon2 = lon1 + np.arctan2(
        np.sin(a) * np.sin(d/R) * np.cos(lat1),
        np.cos(d/R) - np.sin(lat1) * np.sin(lat2)
    )
    return np.degrees(lat2), np.degrees(lon2)

def random_point_on_circle(radius: float,generator: np.random.Generator) -> np.array:
    """ Get a random point on a circle circumference with given radius
    Parameters
    __________
    radius: float
        radius for the circle
    
    Returns
    __________
    point: np.array
        randomly sampled point
    """
    alpha = 2 * np.pi * generator.uniform(0., 1.)
    x = radius * np.cos(alpha)
    y = radius * np.sin(alpha)
    return np.array([x, y])

def sort_points_clockwise(vertices: np.array) -> np.array:
    """ Sort the points in clockwise order
    Parameters
    __________
    vertices: np.array
        array of points
    
    Returns
    __________
    sorted_vertices: np.array
        sorted array of points
    """
    sorted_vertices = [vertices[i] for i in np.argsort([np.arctan2(v[1], v[0]) for v in vertices])]

    return sorted_vertices   

def polygon_area(vertices: np.array) -> float:
    """ Calculate the area of a polygon given the vertices
    Parameters
    __________
    vertices: np.array
        array of vertices of the polygon
    
    Returns
    __________
    area: float
        area of the polygon
    """
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]  # Wrap around to the first vertex
        area += x1 * y2 - y1 * x2
    area = np.abs(area) / 2.0
    return area

def nm_to_latlong(center: np.array, point: np.array) -> np.array:
    """ Convert a point in nm to lat/long coordinates
    Parameters
    __________
    center: np.array
        center point of the conversion
    point: np.array
        point to be converted
    
    Returns
    __________
    latlong: np.array
        converted point in lat/long coordinates
    """
    lat = center[0] + (point[0] / 60)
    lon = center[1] + (point[1] / (60 * np.cos(np.radians(center[0]))))
    return np.array([lat, lon])

def latlong_to_nm(center: np.array, point: np.array) -> np.array:
    """ Convert a point in lat/long coordinates to nm
    Parameters
    __________
    center: np.array
        center point of the conversion
    point: np.array
        point to be converted
    
    Returns
    __________
    nm: np.array
        converted point in nm
    """
    x = (point[0] - center[0]) * 60
    y = (point[1] - center[1]) * 60 * np.cos(np.radians(center[0]))
    return np.array([x, y])

def euclidean_distance(point1: np.array, point2: np.array) -> float:
    """ Calculate the euclidean distance between two points
    Parameters
    __________
    point1: np.array
        [x, y] of the first point
    point2: np.array
        [x, y] of the second point
        
    Returns
    __________
    distance: float
        euclidean distance between the two points
    """
    return np.sqrt(np.sum((point2 - point1)**2))

def get_hdg(point1: np.array, point2: np.array) -> float:
    """ Calculate the heading from point1 to point2
    Parameters
    __________
    point1: np.array
        [lat, lon] of the first point 
    point2: np.array
        [lat, lon] of the second point
    
    Returns
    __________
    hdg: float
        heading from point1 to point2
    """
    
    lat1, lon1 = np.radians(point1)
    lat2, lon2 = np.radians(point2)
    
    delta_lon = lon2 - lon1
    
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    
    hdg = np.degrees(np.arctan2(x, y))
    
    hdg = (hdg + 360) % 360 # Convert back to [0, 360] interval
    
    return hdg

