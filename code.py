import math
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime

def deg2num(lat, lon, zoom):
    """
    Umrechnung Lat/Lon → XYZ Tile‑Koordinaten
    """
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) +
                (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x, y

def fetch_tile(map_id, zoom, x, y, api_key):
    """
    Lädt eine einzelne Tile‑Kachel
    """
    url = f"https://api.maptiler.com/maps/{map_id}/256/{zoom}/{x}/{y}.png?key={api_key}"
    r = requests.get(url)
    r.raise_for_status()
    return Image.open(BytesIO(r.content))

def build_map(lat, lon, zoom, tiles_radius, api_key, map_id="satellite"):
    """
    Erstellt aus mehreren Tiles ein großes Bild
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

if __name__ == "__main__":
    start_time = datetime.now()

    # Beispiel: Paris
    # lat, lon = 48.867606, 2.325756
    # lat, lon = 48.868157, 2.327764
    # lat, lon = 48.865904, 2.332157
    # lat, lon = 48.837053, 2.415571
    lat, lon = 48.861511, 2.299869
    zoom = 18
    api_key = "qR5z0FXl7vgm9kk146HC"        # HIER DEINEN APIKEY AUS DEM FREE PLAN
    img_name = "hehe"

    # Anzahl Tiles um Zentrum (z.B. 1 = 3×3, 2 = 5×5)
    tiles_radius = 2

    map_img = build_map(lat, lon, zoom, tiles_radius, api_key)
    map_img.save(f"{img_name}.png")
    print("Karte gespeichert!")

    end_time = datetime.now()
    print("time:",end_time-start_time)
