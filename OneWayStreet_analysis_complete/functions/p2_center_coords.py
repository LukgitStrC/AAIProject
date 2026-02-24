from PIL import Image, ImageDraw
import numpy as np
import math
import os


# ------------------------------------
#          Image Center Point
# ------------------------------------

def center_point_visualize(image_path, label_path, save_result=True):

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size
    with open(label_path, "r") as f:
        line = f.readline().strip()
    

    parts = line.split()
    coords = [float(x) for x in parts[1:]]  # ignore class_id 
    polygon = np.array(coords).reshape(-1, 2)


    # Get center point
    centerPoint = polygon.mean(axis=0)
    cx, cy = int(centerPoint[0]*width), int(centerPoint[1]*height)


    # Draw center point in Segmentation
    r = 8
    draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill="red")

    # Save Image as Result
    if save_result:
        folder = os.path.dirname(label_path)
        basename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{folder}/{basename}_centerPoint.png"
        img.save(output_path)


    return centerPoint




# ------------------------------------
#          Geo Coordinates
# ------------------------------------

def get_meters_per_pixel(lat, zoom):
    earth_circumference = 40075017  # in meters
    lat_rad = math.radians(lat)
    meters_per_pixel = (earth_circumference * math.cos(lat_rad)) / (256 * 2**zoom)

    return meters_per_pixel


def yolo_to_geo(cp_x, cp_y, img_width, img_height, lat_center, lon_center, meters_per_pixel):

    # Position of Center Point Pixel
    x_pixel = cp_x * img_width
    y_pixel = cp_y * img_height

    # Position of Image center Pixel
    center_x = img_width / 2
    center_y = img_height / 2

    # Difference in px
    dx_pixels = x_pixel - center_x
    dy_pixels = center_y - y_pixel

    # Difference in m
    dx_m = dx_pixels * meters_per_pixel
    dy_m = dy_pixels * meters_per_pixel

    # Calculate lat/lon from Difference from image center
    lat = lat_center + (dy_m / 111320)
    lon = lon_center + (dx_m / (111320 * math.cos(math.radians(lat_center))))

    return lat, lon