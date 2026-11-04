"""Light-gray OSM tile backdrop using existing dependencies and a disk cache."""
from __future__ import annotations

import io
import math
import time

import numpy as np
import requests
from PIL import Image, ImageOps
from pyproj import Transformer

from .common import REF, write_json
from .review_maps import tile_position

WEB = Transformer.from_crs(4326, 3857, always_xy=True)
HALF_WORLD = 20037508.342789244


def gray_backdrop(bounds, zoom=13):
    west, south, east, north = bounds
    xmin, ymin = tile_position(north, west, zoom)
    xmax, ymax = tile_position(south, east, zoom)
    first_x, last_x = math.floor(xmin), math.floor(xmax)
    first_y, last_y = math.floor(ymin), math.floor(ymax)
    mosaic = Image.new("RGB", ((last_x - first_x + 1) * 256, (last_y - first_y + 1) * 256), "white")
    session = requests.Session()
    session.headers["User-Agent"] = "YuseongParkingV1/1.0 (academic figure rendering; disk cache)"
    for tx in range(first_x, last_x + 1):
        for ty in range(first_y, last_y + 1):
            path = REF / "osm_tiles" / str(zoom) / str(tx) / f"{ty}.png"
            if path.exists():
                data = path.read_bytes()
            else:
                response = session.get(f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png", timeout=20)
                response.raise_for_status()
                data = response.content
                Image.open(io.BytesIO(data)).verify()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                time.sleep(.15)
            mosaic.paste(Image.open(io.BytesIO(data)).convert("RGB"), ((tx - first_x) * 256, (ty - first_y) * 256))
    image = ImageOps.grayscale(mosaic).convert("RGB")
    image = Image.blend(image, Image.new("RGB", image.size, "white"), .22)
    tile_m = 2 * HALF_WORLD / (2 ** zoom)
    extent = (first_x * tile_m - HALF_WORLD, (last_x + 1) * tile_m - HALF_WORLD,
              HALF_WORLD - (last_y + 1) * tile_m, HALF_WORLD - first_y * tile_m)
    write_json({"source": "https://tile.openstreetmap.org/{z}/{x}/{y}.png", "zoom": zoom,
                "license": "© OpenStreetMap contributors (ODbL)",
                "style": "grayscale; 22% white blend", "crs": "EPSG:3857", "new_dependencies": []}, REF / "analysis_basemap_source.json")
    return np.asarray(image), extent
