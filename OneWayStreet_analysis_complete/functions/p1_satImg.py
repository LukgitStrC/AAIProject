from ultralytics import YOLO
import requests
from io import BytesIO
from PIL import Image
import math



# ------------------------------------
#         MapTiler API Download
# ------------------------------------

def deg2num(lat, lon, zoom):
    """
    Calculate XYZ-Coords from Lat/Lon
    """
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) +
                (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x, y


def fetch_tile(map_id, zoom, x, y, api_key):
    """
    Loads single MapTiler Tile
    """
    url = f"https://api.maptiler.com/maps/{map_id}/256/{zoom}/{x}/{y}.png?key={api_key}"
    r = requests.get(url)
    r.raise_for_status()
    return Image.open(BytesIO(r.content))


def build_map(lat, lon, zoom, tiles_radius, api_key, map_id="satellite"):
    """
    Creates Satellite-Image from multiple tiles around center point
    """
    center_x, center_y = deg2num(lat, lon, zoom)

    size = 256 * (tiles_radius * 2 + 1)
    canvas = Image.new("RGB", (size, size))

    for dx in range(-tiles_radius, tiles_radius + 1):
        for dy in range(-tiles_radius, tiles_radius + 1):
            x = center_x + dx
            y = center_y + dy
            tile = fetch_tile(map_id, zoom, x, y, api_key)
            px = (dx + tiles_radius) * 256
            py = (dy + tiles_radius) * 256
            canvas.paste(tile, (px, py))

    return canvas





# ------------------------------------
#         YOLO Prediction
# ------------------------------------

def predict_intersection(save_dir, image_path, model_path="yolov11n-seg.pt"):    

    # load model
    model = YOLO(model_path)
    model.to('cuda')

    # segment image (prediction)
    results = model.predict(
        source=image_path,
        imgsz=512,     
        conf=0.4,      # Confidence
        max_det=1,     # only 1 Detection per Image
        save=True,
        project=f"{save_dir}",
        save_txt=True,
    )

    # get resulting image
    # result_img = results[0].plot(pil=True)  # returns rgb image of np.array from result object

    return results





