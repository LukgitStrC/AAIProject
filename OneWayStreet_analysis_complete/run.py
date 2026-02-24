from ultralytics import YOLO
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import math
import os
import shutil


from functions.p1_satImg import *
from functions.p2_center_coords import *
from functions.p3_svp import *
from functions.p4_streetImages import *
from functions.p5_streetSigns import *



# Pipeline:
#     1. Get Satellite Image at give geo-coordinates
#     2. Get Center point of intersection in satellite Image
#     3. Download nearest Street View Panorama at intersection center
#     4. Split Street View Panorama into normal street images
#     5. Get relevant street images
#     6. Search for relevant street signs
#     7. Classify as One-Way Street or not





# ------------------------------------
#         1 Get Satellite Image
# ------------------------------------

lat_sat, lon_sat = 49.177560, -0.380128
img_name = "IMG 1 TEST"     # Name of Satellite Image


tiles_radius = 2    # number of tiles around center-coords
zoom = 21       # Std = 21
api_key = "qR5z0FXl7vgm9kk146HC"
map_img = build_map(lat_sat, lon_sat, zoom, tiles_radius, api_key)

img_input_fp = "results/input_satImg"
os.makedirs(img_input_fp, exist_ok=True)
map_img.save(f"{img_input_fp}/{img_name}.png") 
if os.path.exists(f"{img_input_fp}/{img_name}.png"):
    print("Download of satellite image successful!")


os.makedirs("results/intersections_pred", exist_ok=True)       # Create/Check for predictionsfolder
save_dir = f"results/intersections_pred/{img_name}"            # save in predictions-folder for each image processed
os.makedirs(save_dir, exist_ok=True)
if save_dir:
    shutil.rmtree(save_dir)                     # deletes previous results


# Segmentation/Prediction
results = predict_intersection(
    save_dir=save_dir,
    image_path= f"{img_input_fp}/{img_name}.png",   # image save-path
    model_path="models/segmentation/best.pt",       # Path to YOLO-model
)


if not save_dir:
    print(f"No Predictions for image {img_name}")
else: print("Segmentation of intersection successful!")



# ------------------------------------
#      2 Get intersection center
# ------------------------------------


# prediction image/labels to get center point
image_path = f"{save_dir}/predict/{img_name}.jpg"
label_path = f"{save_dir}/predict/labels/{img_name}.txt"

# Get and Visualize center point
center_point = center_point_visualize(image_path, label_path)

# Get geo Coords of Center Point
cx, cy = center_point[0], center_point[1]
mpp = get_meters_per_pixel(lat_sat, zoom)
cp_lat, cp_lon = yolo_to_geo(cx, cy, 1280, 1280, lat_sat, lon_sat, mpp)


print("Center Point lat:", cp_lat, "Center Point lon:", cp_lon)
print(f"{img_name}: [{lat_sat},{lon_sat}],[{cp_lat},{cp_lon}]")





# ------------------------------------
#      3 Download adjacent SVP
# ------------------------------------

# Configuration 
OUT_DIR = "results/svp_images"
os.makedirs(OUT_DIR, exist_ok=True)       # Create/Check for svp_image
BASE_URL = "https://api.panoramax.xyz"
SEARCH_URL = f"{BASE_URL}/api/search"

RADIUS = 3.0    # Search Radius around coordinates

found_svp = get_images_at(cp_lat, cp_lon, RADIUS, SEARCH_URL)

svp_dist = []
for x in found_svp:
    svp_dist.append(x.get('distance_from_location'))

min_dist_idx = svp_dist.index(min(svp_dist))
svp = found_svp[min_dist_idx]
svp_id = svp.get('id')
found_svp = download_images_from_features([svp], OUT_DIR)


svp_path = f"{OUT_DIR}/{svp_id}.jpg"
if not os.path.exists(svp_path):
    print("No Street View Panorama downloaded!")




# ------------------------------------
#           4 Split SVP
# ------------------------------------

streetImg_output_dir = f"results/svp_split/{svp_id}"
if os.path.exists(streetImg_output_dir):
    shutil.rmtree(streetImg_output_dir)    

os.makedirs(streetImg_output_dir, exist_ok=True) 


panorama = Image.open(svp_path)
pano_width, pano_height = panorama.size


for deg in tqdm(np.arange(0, 360, 20)):
    output_image = panorama_to_plane(svp_path, 80, (600, 600), deg, 90)
    filename = f"img_{svp_id}_{int(deg)}.png"
    filepath = os.path.join(streetImg_output_dir, filename)
    output_image.save(filepath)

if not os.path.exists(filepath):
    print("No Street View Panorama downloaded!")

# ------------------------------------
#     5 get relevant street images
# ------------------------------------





# ------------------------------------
#        6 detect street signs
# ------------------------------------





# ------------------------------------
#         7 classify street
# ------------------------------------