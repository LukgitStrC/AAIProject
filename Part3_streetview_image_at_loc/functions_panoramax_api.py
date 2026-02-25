import os
import requests
import math

# --- Configuration ---
OUT_DIR = "streetview_images"
BASE_URL = "https://api.panoramax.xyz"
SEARCH_URL = f"{BASE_URL}/api/search"

def download_image(picture_id):
    """
    downloads image behind given picture id into OUT_DIR
    """
    try:
        path = f"api/pictures/{picture_id}"
        metadata_url = f"{BASE_URL}/{path}"
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"Directory verified: {OUT_DIR}")

        # Get the picture metadata (JSON response)
        print(f"Fetching metadata from: {metadata_url}")
        response = requests.get(metadata_url)
        response.raise_for_status() # Check for HTTP errors
        data = response.json()

        # extract azimuth
        properties = data.get('properties', {})

        # Extract the image URL from the response
        # Panoramax/GeoVisio returns a STAC Item. The image links are in the 'assets' dictionary.
        # We prioritize 'hd' (High Definition) but can fall back to 'sd' (Standard Definition).
        assets = data.get('assets', {})
        
        if 'hd' in assets:
            image_url = assets['hd']['href']
            print("Found HD image URL.")
        elif 'sd' in assets:
            image_url = assets['sd']['href']
            print("HD not available, found SD image URL.")
        else:
            raise ValueError("No suitable image asset ('hd' or 'sd') found in the response.")

        print(f"Image URL extracted: {image_url}")
        print("Downloading image...")
        image_response = requests.get(image_url)
        image_response.raise_for_status()

        output_path = f"{OUT_DIR}/{picture_id}.png"
        
        with open(output_path, 'wb') as file:
            file.write(image_response.content)
        print(f"Success! Image saved to: {output_path}")
        return data

    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_bbox_from_point(lat, lon, dist_meters):
    """
    converts given long, lat, dist into  lon1,lat1,lon2,lat2 - BBox
    """
    earth_radius = 6378137
    d_lat = dist_meters / earth_radius
    d_lon = dist_meters / (earth_radius * math.cos(math.pi * lat / 180))
    
    # Offsets in degrees
    d_lat_deg = d_lat * 180 / math.pi
    d_lon_deg = d_lon * 180 / math.pi
    
    return f"{lon - d_lon_deg},{lat - d_lat_deg},{lon + d_lon_deg},{lat + d_lat_deg}"

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the distance between two points in meters.
    """
    R = 6378137  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def is_360_panorama(feature):
    """
    Determines if a STAC feature is a 360 degree panorama using multiple checks.
    """
    props = feature.get("properties", {})
    assets = feature.get("assets", {})
    
    # Check 1: Explicit Projection Type (The Standard)
    if props.get("GPano:ProjectionType") == "equirectangular":
        return True

    # Check 2: Explicit Field of View
    # Some uploads might lack GPano tags but have STAC view properties
    
    fov = props.get('pers:interior_orientation').get("field_of_view")
    if fov and fov >= 360:
        return True

    # Check 3: Aspect Ratio Heuristic (The Fallback)
    # If metadata is missing, we check if the image dimensions are 2:1
    # We look for dimensions in 'hd' or 'sd' assets, or properties
    width = props.get("exif:pixelXDimension")
    height = props.get("exif:pixelYDimension")

    # If not in properties, try to find it in assets (sometimes hidden there)
    if not width or not height:
        for key in ['hd', 'sd', 'visual']:
            asset = assets.get(key, {})
            # Some STAC items put dims in the asset metadata
            if 'proj:shape' in asset: # [height, width]
                h, w = asset['proj:shape']
                if w > 0 and h > 0:
                    ratio = w / h
                    # Allow small margin of error for compression artifacts (1.9 to 2.1)
                    if 1.9 < ratio < 2.1:
                        return True

    return False

# function that returns all images in search radius
# the images are are here the features
# 'properties' contains metadata
# 'assets' contains the image url
# 'geometry' contains infromation about the location where the image was taken
def get_images_at(lat,lon,rad):
  """
  returns 
  """
  bbox_string = get_bbox_from_point(lat,lon,rad)
  print(f"Searching area: {bbox_string}")

  params = {
    "bbox": bbox_string,
    "limit": 100
  }

  try:
    # request
    response = requests.get(SEARCH_URL, params=params)
    response.raise_for_status()
    features = response.json().get("features", [])
    
    print(f"Total images found: {len(features)}")
    
    def get_distance(x):
       geo = x.get('geometry',{}).get("coordinates", [])
       if len(geo) > 2:
          # images where distance can not be calculated should be at the end
          return math.inf
       img_lon, img_lat = geo[0], geo[1]
       dist = haversine_distance(lat, lon, img_lat, img_lon)
       # save calculated distance for later uses
       x["distance_from_location"] = dist
       return dist

    filtered_features = sorted(
      filter(is_360_panorama,features),
      key=get_distance)
    print(f"360° images found: {len(filtered_features)}")
    return filtered_features

  except Exception as e:
    print(f"Error: {e}")
    return []

def down_load_images_from_features(features,out_dir="streetview_images"):
    """
    Loads images from image_url property and saves them in the out_dir
    path is now saved in feature.path"""
    for feature in features:
        picture_id = feature.get('id')
        output_filename = f"{picture_id}.jpg"
        output_path = os.path.join(OUT_DIR, output_filename)

        assets = feature.get('assets', {})
        if 'hd' in assets:
            image_url = assets['hd']['href']
            print("Found HD image URL.")
        elif 'sd' in assets:
            image_url = assets['sd']['href']
            print("HD not available, found SD image URL.")
        else:
            raise ValueError("No suitable image asset ('hd' or 'sd') found in the response.")

        print(f"Image URL extracted: {image_url}")
        print("Downloading image...")
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        with open(output_path, 'wb') as file:
            file.write(image_response.content)
        print(f"Success! Image saved to: {output_path}")
        # output_path saved to features
        feature['path'] = output_path
    return features