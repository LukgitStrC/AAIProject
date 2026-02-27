from ultralytics import YOLO
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import math
import os
import shutil
import json

from functions.p1_satImg import *
from functions.p2_center_coords import *
from functions.p3_svp import *
from functions.p4_streetImages import *
from functions.p5_streetSigns import *

from functions.drawings import create_marked_svp


# Pipeline:
#     1. Get Satellite Image at give geo-coordinates
#     2. Get Center point of intersection in satellite Image
#     3. Download nearest Street View Panorama at intersection center
#     4. Split Street View Panorama into normal street images contianing only single street entrances
#     5. detect street signs and classify them

# INPUT
# coordinates of intersection (with available 360 streetview on panoramax instance)
lat_sat, lon_sat = 49.18278, -0.35821
identifier = "test1" 

# OUTPUT
summary = {}
summary['coords'] = (lat_sat,lon_sat)

# Folder structure
res_dir = 'results'
model_dir = 'models'

abs_script_dir = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(res_dir):
    shutil.rmtree(res_dir)
os.makedirs(res_dir, exist_ok=True)

sat_img_dir = os.path.join(res_dir,'sat')
intersection_dir = os.path.join(res_dir,'intersection')
svp_images_dir = os.path.join(res_dir,'svp')
crops_dir = os.path.join(res_dir,'crops')
summary_dir = os.path.join(res_dir,'summary')

os.makedirs(sat_img_dir, exist_ok=True)
os.makedirs(intersection_dir, exist_ok=True)
os.makedirs(svp_images_dir, exist_ok=True)
os.makedirs(crops_dir, exist_ok=True)
os.makedirs(summary_dir, exist_ok=True)

# distzinct suffixes for files which otherway would have the same identifier
SAT_IMG_SUFFIX = '_SAT'
INTERS_SUFFIX = '_INTERS'
SVP_SUFFIX = '_SVP'
CROP_SUFFIX = '_CROP'

# paths
sat_img_path = os.path.join(sat_img_dir,f'{identifier}{SAT_IMG_SUFFIX}.png')
intersection_img_path = os.path.join(intersection_dir,'predict',f'{identifier}{SAT_IMG_SUFFIX}.jpg')
intersection_label_path = os.path.join(intersection_dir,'predict','labels',f'{identifier}{SAT_IMG_SUFFIX}.txt')
svp_img_path = os.path.join(svp_images_dir,f'{identifier}{SVP_SUFFIX}.png')
# number of crops is dynamic, so their paths are also dynamic
summary_path = os.path.join(summary_dir,f'{identifier}.json')

seg_model_path = os.path.join(model_dir,'segmentation','best.pt') 

# ------------------------------------
#         1 Get Satellite Image
# ------------------------------------

tiles_radius = 2    # number of tiles around center-coords
zoom = 21       # Std = 21
api_key = "qR5z0FXl7vgm9kk146HC"
map_img = build_map(lat_sat, lon_sat, zoom, tiles_radius, api_key)

map_img.save(sat_img_path) 
if os.path.exists(sat_img_path):
    print("Download of satellite image successful!")

# Segmentation/Prediction
# absolute path to intersection dir is needed because of yolos habit of saving
# per default to the venv root/runs folder
abs_intersection_dir = os.path.join(abs_script_dir, intersection_dir)

results = predict_intersection(
    save_dir=abs_intersection_dir,
    image_path= sat_img_path,
    model_path=seg_model_path
)

print("Segmentation of intersection successful!")

# ------------------------------------
#      2 Get intersection center
# ------------------------------------

# prediction image/labels to get center point

# Get and Visualize center point
center_point = center_point_visualize(intersection_img_path, intersection_label_path)

# Get geo Coords of Center Point
cx, cy = center_point[0], center_point[1]
mpp = get_meters_per_pixel(lat_sat, zoom)
cp_lat, cp_lon = yolo_to_geo(cx, cy, 1280, 1280, lat_sat, lon_sat, mpp)

print("Center Point lat:", cp_lat, "Center Point lon:", cp_lon)
print(f"{identifier}: [{lat_sat},{lon_sat}],[{cp_lat},{cp_lon}]")
summary['center'] = (cp_lat,cp_lon)
# ------------------------------------
#      3 Download adjacent SVP
# ------------------------------------

# Configuration 
BASE_URL = "https://api.panoramax.xyz"
SEARCH_URL = f"{BASE_URL}/api/search"

summary['panoramax_instance'] = BASE_URL

RADIUS = 3.0    # Search Radius around coordinates
summary['svp_search_radius'] = 3.0

found_svp = get_images_at(cp_lat, cp_lon, RADIUS, SEARCH_URL)

svp_dist = []
for x in found_svp:
    svp_dist.append(x.get('distance_from_location'))

min_dist_idx = svp_dist.index(min(svp_dist))
svp = found_svp[min_dist_idx]
svp_id = svp.get('id')
found_svp = download_image_from_feature(svp, svp_img_path)

if not os.path.exists(svp_img_path):
    print("No Street View Panorama downloaded!")

summary['panoramax_picture'] = found_svp # contains all information about the panoramax image found and used

# ------------------------------------------
#      4 Get relevant image crops
# ------------------------------------------
img = cv2.imread(found_svp['path'])
depth_model = DepthEstimationModel()
# use ML model to estimate depth of image as heuristic for streetviews in cities
depth_map = depth_model.predict(img)
# from this depth map the direction of streets can be estimated using peak detection
angles = find_street_angles(depth_map)
# then the crops can be extracted
image_crops = extract_street_views(img, angles)

# save crops to file system
crop_paths = []
for crop, yaw in zip(image_crops, angles):
    crop_path = os.path.join(crops_dir,f'{identifier}{yaw}{CROP_SUFFIX}.png')
    cv2.imwrite(crop_path,crop)
    crop_paths.append(crop_path)

# ------------------------------------
#        5 classify street
# ------------------------------------
roads = []
for crop_path, yaw in zip(crop_paths,angles):
    crop_label = classify_street(crop_path)
    print(f'The street at angle {yaw} is a {crop_label.value}')
    roads.append({'yaw':yaw, 'label':crop_label.value})

marked_svp_path = os.path.join(svp_images_dir,f'{identifier}{SVP_SUFFIX}_marked.png')
create_marked_svp(svp_img_path,marked_svp_path,roads)

summary['roads'] = roads
# save summary to file system
with open(summary_path,'w') as f:
    json.dump(summary,f, indent=4)