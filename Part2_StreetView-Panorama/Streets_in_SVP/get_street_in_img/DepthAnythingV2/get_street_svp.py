from transformers import pipeline
from PIL import Image
import numpy as np
import time
import os
import shutil


import warnings
warnings.filterwarnings("ignore")


def get_depth_map(image_path):
    # ----- Create Depth-Map -----
    image = Image.open(image_path)
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

    output = pipe(image)
    depth_map = output["depth"]

    depth_map_np = np.array(depth_map)
    orig_img_np = np.array(image.convert("RGB"))

    return orig_img_np, depth_map_np



def get_center_region_mean_depth(depth_map_np, region_ratio_to_img=0.5):

    # image center point
    H, W = depth_map_np.shape
    center_y = H // 2
    center_x = W // 2

    # dimensions and coords of rectangle
    region_h = int(H * region_ratio_to_img)
    region_w = int(W * region_ratio_to_img)
    y_start = max(0, center_y - region_h // 2)
    y_end = min(H, center_y + region_h // 2)
    x_start = max(0, center_x - region_w // 2)
    x_end = min(W, center_x + region_w // 2)

    # center region of depth-map
    center_region = depth_map_np[y_start:y_end, x_start:x_end]
    mean_depth_raw = np.mean(center_region)

    return mean_depth_raw




def get_dist_to_center(orig_img_np, depth_map_np):

    # ----- Create strip in depth-map -----
    H, W = depth_map_np.shape
    orig_h, orig_w = orig_img_np.shape[:2]

    # strip dimensions
    y_ratio = 1           # vertical position (0: upper end, 1: lower end)
    strip_height_ratio = 1.2  # height of strip (default: 1.2)

    # strip in depth-Map
    y_center = int(H * y_ratio)
    strip_h = max(5, int(H * strip_height_ratio))
    y_start = max(0, y_center - strip_h // 2)
    y_end = min(H, y_center + strip_h // 2)
    depth_strip = depth_map_np[y_start:y_end, :]

    # Strip in original image
    y_start_orig = int(orig_h * y_ratio - orig_h * strip_height_ratio / 2)
    y_end_orig = int(orig_h * y_ratio + orig_h * strip_height_ratio / 2)


    # ----- get point of maximal depth in strip -----
    y_strip, x_strip = np.unravel_index(np.argmin(depth_strip, axis=None), depth_strip.shape)
    x_depth = x_strip
    x_orig = int(x_depth * orig_w / W)  # point of max depth on original image


    # ----- get area of maximal depth in strip -----
    tolerance = 1e-5
    ys, xs = np.where(np.abs(depth_strip - depth_strip[y_strip, x_strip]) < tolerance)

    for y_pt, x_pt in zip(ys, xs):  # all points in area
        x_orig = int(x_pt * orig_w / W)
        y_orig = int((y_pt / strip_h) * (y_end_orig - y_start_orig)) + y_start_orig


    # ----- get mean of maximal depth area -----
    center_x_img = orig_w // 2
    center_y_img = orig_h // 2

    max_depth_point = np.array([x_orig, y_orig])
    image_center = np.array([center_x_img, center_y_img])



    # ----- calculate distance to image center point -----
    distance = np.linalg.norm(max_depth_point - image_center)

    return distance


def save_max_depth_images(input_path, img_names, dist_list, outdir):

    for f in dist_list:
        img_fp = f"{input_path}/{img_names[f]}"
        shutil.copy(img_fp, outdir)








if __name__ == "__main__":
    start_t = time.time()


    # ---- Params ----
    img_name = "cc6b8656-d52e-42d1-ab10-542588c9f9f0"   # Image ID of Panoramax SVP
    # img_name = "6f34a62d-c8be-429d-b266-b1aca354ae1d"
    # img_name = "out_61efb4ce-cdf2-4160-aaf6-f5bca9c7e247"
    # img_name = "61efb4ce-cdf2-4160-aaf6-f5bca9c7e247"
    img_name = "2ad5a6fe-14bb-4f07-8ee5-780c7e068fee"

    input_path = f"examples/out_{img_name}"             # fp to folder
    outdir_name = "results"                             # Name of general output directory
    outdir = f"{outdir_name}/{img_name}"                # fp to output directory of folder


    if os.path.exists(outdir):
        shutil.rmtree(outdir)    
    os.makedirs(outdir, exist_ok=True)

    


    # ---- depth analysis ----
    dist_list = np.array([])
    mean_center_depth  = np.array([])


    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
        region_ratio = 0.4      # Area around image center for mean depth anaylsis
        
        # Filter for relevant images (street) 
        for img_fp in files:                                                                    # for each image in folder
            orig_img, depth_map = get_depth_map(f"{input_path}/{img_fp}")                       # get depth-map
            mean_center_depth = get_center_region_mean_depth(depth_map, region_ratio)           # get mean depth in image center
            mcd_list = np.append(mcd_list, mean_center_depth)

        mean_img_depth = np.mean(mcd_list)                                                      # get mean of all image center depths
        mcd_list = sorted(mcd_list)

        for d in mcd_list:                          # for each image 
            if d <= mean_img_depth:                 # lower bound
                continue
            elif:
                for img_fp in files:
                #     Wurde ein Bild mit benachbartem Winkel eingefügt?
                #         JA: Skip
                #     Nein: Einfügen als potentielle Straße

                # Pro Bild potentielle Straße
                #     benachbarte Bilder nehmen (Gruppieren)
                #         Pro Bild in Gruppe:
                #             Distanz Tiefster Punkt zum Bildmittelpunkt
                #             Auswahl des Bildes mit geringster Distanz

                #             => Dieses Bild speichern als Straßenbild zur Klassifizierung
                



        # save_max_depth_images(input_path, files, topMax, outdir)

    else:
        print("Input is not a folder. Stopping..")
        

    end_time = time.time()
    print(f"runtime: {end_time - start_t:.3f} seconds")
