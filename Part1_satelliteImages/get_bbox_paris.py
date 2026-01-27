import math
import requests
from PIL import Image
from io import BytesIO
import os
import sys
from datetime import datetime

def deg2num(lat, lon, zoom):
    """
    Umrechnung Lat/Lon → XYZ Tile-Koordinaten
    """
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) +
                (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x, y

def fetch_tile(map_id, zoom, x, y, api_key):
    """
    Lädt eine einzelne Tile-Kachel
    """
    url = f"https://api.maptiler.com/maps/{map_id}/256/{zoom}/{x}/{y}.png?key={api_key}"
    r = requests.get(url)
    r.raise_for_status()
    return Image.open(BytesIO(r.content))

def build_map_from_center(center_x, center_y, zoom, tiles_radius, api_key, map_id="satellite"):
    """
    Erstellt aus mehreren Tiles ein großes Bild (wie im Originalcode),
    aber mit expliziten Tile-Koordinaten statt Lat/Lon.
    """
    size = 256 * (tiles_radius * 2 + 1)
    canvas = Image.new("RGB", (size, size))

    total_tiles = (tiles_radius * 2 + 1) ** 2
    tile_counter = 0

    for dx in range(-tiles_radius, tiles_radius + 1):
        for dy in range(-tiles_radius, tiles_radius + 1):
            tile_counter += 1
            percent = (tile_counter / total_tiles) * 100
            print(f"\r   Tile-Fortschritt: {tile_counter}/{total_tiles} ({percent:.1f}%)", end="", flush=True)

            x = center_x + dx
            y = center_y + dy
            tile = fetch_tile(map_id, zoom, x, y, api_key)
            px = (dx + tiles_radius) * 256
            py = (dy + tiles_radius) * 256
            canvas.paste(tile, (px, py))

    print()  # neue Zeile nach Tiles
    return canvas



def build_maps_from_bbox(lat_min, lon_min, lat_max, lon_max, zoom, tiles_radius, api_key,
                         map_id="satellite", out_dir="bbox_maps"):
    """
    Erstellt eine Reihe an zusammengesetzten Bildern, die zusammen die Bounding Box abdecken,
    mit Fortschrittsanzeige.
    """

    os.makedirs(out_dir, exist_ok=True)

    # Tile-Koordinaten der Bounding-Box-Ecken
    x_min, y_max = deg2num(lat_min, lon_min, zoom)  # unten links
    x_max, y_min = deg2num(lat_max, lon_max, zoom)  # oben rechts

    # Sortieren zur Sicherheit
    x_start, x_end = sorted([x_min, x_max])
    y_start, y_end = sorted([y_min, y_max])

    block_size = 2 * tiles_radius + 1  # z.B. 5 bei radius=2

    total_blocks_x = math.ceil((x_end - x_start + 1) / block_size)
    total_blocks_y = math.ceil((y_end - y_start + 1) / block_size)
    total_images = total_blocks_x * total_blocks_y
    image_counter = 0

    print(f"Erzeuge {total_images} Bilder ({block_size}×{block_size} Tiles je Bild)...")

    for cy in range(y_start, y_end + 1, block_size):
        for cx in range(x_start, x_end + 1, block_size):
            image_counter += 1
            percent = (image_counter / total_images) * 100
            print(f"\nBild {image_counter}/{total_images} ({percent:.1f}%)")

            # Mittelpunkt-Tile für diesen Block
            center_x = cx + tiles_radius
            center_y = cy + tiles_radius

            map_img = build_map_from_center(center_x, center_y, zoom, tiles_radius, api_key, map_id)

            filename = f"{out_dir}/bbox_map_{image_counter:03d}_z{zoom}_x{center_x}_y{center_y}.png"
            map_img.save(filename)
            print(f"   Gespeichert: {filename}")

    print("\nAlle Kartenbilder erstellt!")




if __name__ == "__main__":
    start_time = datetime.now()

    # Beispiel-Bounding-Box
    # lat_min = 48.8450  
    # lon_min = 2.3100
    lat_min, lon_min = 48.857345, 2.349261  # unten links
    # lat_max = 48.8750   
    # lon_max = 2.3600
    lat_max, lon_max = 48.859422, 2.356257 # oben rechts

    zoom = 21
    api_key = "qR5z0FXl7vgm9kk146HC"  # DEIN API-KEY
    tiles_radius = 2

    build_maps_from_bbox(lat_min, lon_min, lat_max, lon_max, zoom, tiles_radius, api_key)

    end_time = datetime.now()
    print("time:",end_time-start_time)