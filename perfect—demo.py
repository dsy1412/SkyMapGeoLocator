import math
import numpy as np
from scipy.optimize import fsolve
from sklearn.cluster import DBSCAN

def hour_angle_to_degrees(hour_angle):
    parts = hour_angle.replace('h', ' ').replace('m', ' ').replace('s', ' ').split()
    hours, minutes, seconds = map(float, parts)
    return (hours + minutes / 60 + seconds / 3600) * 15

def declination_to_degrees(declination):
    parts = declination.replace('°', ' ').replace("'", " ").replace('"', ' ').split()
    degrees, minutes, seconds = map(float, parts)
    sign = -1 if degrees < 0 else 1
    return degrees + sign * (minutes / 60 + seconds / 3600)

def calculate_gp(stars_data):
    gp_coordinates = []
    for hour_angle, declination in stars_data:
        hour_angle_deg = hour_angle_to_degrees(hour_angle)
        print(hour_angle_deg)
        declination_deg = declination_to_degrees(declination)
        longitude = 360 - hour_angle_deg
        longitude = (longitude + 180) % 360 - 180
        gp_coordinates.append((longitude, declination_deg))
    return gp_coordinates

'''

def calculate_gp(stars_data):
    gp_coordinates = []
    for hour_angle, declination in stars_data:
     
        declination_deg = declination_to_degrees(declination)
        longitude = 360 - hour_angle_deg
        longitude = (longitude + 180) % 360 - 180
        gp_coordinates.append((longitude, declination_deg))
    return gp_coordinates


'''

def calculate_3d_angle(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    cosine_angle = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
def spherical_to_cartesian(longitude, latitude):
    lon_rad = math.radians(longitude)
    lat_rad = math.radians(latitude)
    x = math.cos(lat_rad) * math.cos(lon_rad)
    y = math.cos(lat_rad) * math.sin(lon_rad)
    z = math.sin(lat_rad)
    return np.array([x, y, z])
def angle_between_stars(star1, star2):
    vector1 = spherical_to_cartesian(*star1)
    vector2 = spherical_to_cartesian(*star2)
    dot_product = np.dot(vector1, vector2)
    angle_rad = math.acos(dot_product / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
    return math.degrees(angle_rad)

stars_data = [("16h6m48.35s", "13°38'1.9''"),("15h37m30.47s", "4°11'4.50''"),("14h52m0.01s", "24°10'46.60''"),("14h41m55.45s", "-13°26'19.2''"),  ("15h56m28.72s", "3°20'16.60''")]

gp_coordinates = calculate_gp(stars_data)
print(gp_coordinates)

angles_between_stars = {}
for i in range(len(gp_coordinates)):
    for j in range(i + 1, len(gp_coordinates)):
        angle = angle_between_stars(gp_coordinates[i], gp_coordinates[j])
        angles_between_stars[f"{i+1}&{j+1}"] = angle
print(angles_between_stars)

'''(44, -131),
    (-97, 97.5),
    (-384, -365),
    (-412, 581),
    (13, 106.5)'''

photo_coordinates = [ (44, -131),(-97, 97.5),(-384, -365), (-412, 581), (13, 106.5)]

def find_best_z(photo_coordinates, theoretical_angles):
    def calculate_difference(z):
        stars_3d_coordinates = [np.array([x, y, z]) for x, y in photo_coordinates]
        observed_angles = {}
        for i in range(len(stars_3d_coordinates)):
            for j in range(i + 1, len(stars_3d_coordinates)):
                angle = calculate_3d_angle(stars_3d_coordinates[i], stars_3d_coordinates[j])
                observed_angles[f"{i + 1}&{j + 1}"] = angle
        difference = sum((theoretical_angles[pair] - observed_angles[pair]) ** 2 for pair in theoretical_angles)
        return difference
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(calculate_difference)

    return res.x if res.success else None
best_z = find_best_z(photo_coordinates, angles_between_stars)
print(best_z)

def calculate_observed_angles(photo_coordinates, z):
    stars_3d_coordinates = [np.array([x, y, z]) for x, y in photo_coordinates]
    observed_angles = {}
    for i in range(len(stars_3d_coordinates)):
        for j in range(i + 1, len(stars_3d_coordinates)):
            angle = calculate_3d_angle(stars_3d_coordinates[i], stars_3d_coordinates[j])
            observed_angles[f"{i + 1}&{j + 1}"] = angle
    return observed_angles
observed_angles_with_best_z = calculate_observed_angles(photo_coordinates, best_z)
print(observed_angles_with_best_z)
z=observed_angles_with_best_z


#得到图片的天顶

def calculate_elevation_angles(photo_coordinates, zenith_coordinates, z):
    zenith_3d = np.array([zenith_coordinates[0], zenith_coordinates[1], z])
    elevation_angles = {}
    for i, (x, y) in enumerate(photo_coordinates, start=1):
        star_3d = np.array([x, y, z])
        angle_to_zenith = calculate_3d_angle(star_3d, zenith_3d)
        elevation_angle = 90 - angle_to_zenith
        elevation_angles[f"Star {i}"] = elevation_angle
    return elevation_angles

zenith_coordinates = (-117.5, -852.5)

elevation_angles = calculate_elevation_angles(photo_coordinates, zenith_coordinates, best_z)
print(elevation_angles)


def calculate_plane_equations(gp_coordinates, elevation_angles):
    plane_equations = {}
    for i, (gp_long, gp_lat) in enumerate(gp_coordinates, start=1):
        elevation_angle = elevation_angles[f"Star {i}"]
        gp_long_rad = np.radians(gp_long)
        gp_lat_rad = np.radians(gp_lat)
        zenith_distance_rad = np.radians(90) - np.radians(elevation_angle)
        A = np.cos(gp_lat_rad) * np.cos(gp_long_rad)
        B = np.cos(gp_lat_rad) * np.sin(gp_long_rad)
        C = np.sin(gp_lat_rad)
        D = np.cos(zenith_distance_rad)
        plane_equations[f"Star {i}"] = (A, B, C, D)
    return plane_equations

plane_equations = calculate_plane_equations(gp_coordinates, elevation_angles)
print(plane_equations)




def intersection_of_two_planes_and_sphere(plane1, plane2):
    def equations(vars):
        x, y, z = vars
        eq1 = plane1[0]*x + plane1[1]*y + plane1[2]*z - plane1[3]
        eq2 = plane2[0]*x + plane2[1]*y + plane2[2]*z - plane2[3]
        eq3 = x**2 + y**2 + z**2 - 1  # Unit sphere equation
        return (eq1, eq2, eq3)

    solution1 = fsolve(equations, (1, 0, 0))
    solution2 = fsolve(equations, (-1, 0, 0))
    return solution1, solution2

intersection_points = []
for i, key1 in enumerate(plane_equations):
    for key2 in list(plane_equations)[i+1:]:
        plane1 = plane_equations[key1]
        plane2 = plane_equations[key2]
        points = intersection_of_two_planes_and_sphere(plane1, plane2)
        intersection_points.extend(points)

def cartesian_to_geographic(cartesian_coords):
    x, y, z = cartesian_coords
    # Avoid division by zero in longitude calculation
    longitude = np.degrees(np.arctan2(y, x)) % 360
    latitude = np.degrees(np.arcsin(z))
    return longitude, latitude

geographic_coordinates = [cartesian_to_geographic(point) for point in intersection_points if np.linalg.norm(point) - 1 < 1e-6]  # We check if the point is on the unit sphere

print(geographic_coordinates)
print(len(geographic_coordinates))

filtered_points = [coord for coord in geographic_coordinates if 0 <= coord[1] <= 90]

def calculate_centroid(points):
    x, y, z = zip(*[spherical_to_cartesian(lon, lat) for lon, lat in points])
    centroid = np.array([np.mean(x), np.mean(y), np.mean(z)])
    return cartesian_to_geographic(centroid / np.linalg.norm(centroid))


def points_in_northern_hemisphere(geographic_coordinates):
    return [coord for coord in geographic_coordinates if coord[1] >= 0]

northern_points = points_in_northern_hemisphere(geographic_coordinates)
cartesian_coords = [spherical_to_cartesian(*coord) for coord in geographic_coordinates]

db = DBSCAN(eps=0.1, min_samples=1)
clusters = db.fit_predict(cartesian_coords)
largest_cluster_index = max(range(len(set(clusters))), key=list(clusters).count)
largest_cluster_mask = clusters == largest_cluster_index
largest_cluster_coords = np.array(geographic_coordinates)[largest_cluster_mask]
centroid = calculate_centroid(largest_cluster_coords) if largest_cluster_coords.size else None

print(centroid)