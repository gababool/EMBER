# GIS module — data setup

Local Swedish geodata that feeds operational context (roads, water, forest,
settlements, terrain) to the LLM stage via OSM and SRTM elevation data.

## Region

50 × 50 km box over the 2014 Hälleskogsbrännan wildfire area (Västmanland).

| | min | max |
|---|---|---|
| WGS84 lon | 15.5 | 16.8 |
| WGS84 lat | 59.5 | 60.2 |

## Setup

```bash
uv sync                            # Python deps
brew install osmium-tool gdal      # OSM clipping + raster tools
mkdir -p data/gis/osm data/gis/elevation
```

### OSM vector data

Download `sweden-latest.osm.pbf` from
<https://download.geofabrik.de/europe/sweden.html> into `data/gis/osm/`,
then clip to the operational region:

```bash
cd data/gis/osm
osmium extract -b 15.5,59.5,16.8,60.2 \
  sweden-*.osm.pbf -o vastmanland-50km.osm.pbf
```

The clipped file is ~50–100 MB. The full Sweden download (~2 GB) can be
deleted after clipping.

### SRTM elevation raster

Terrain context (elevation, slope, aspect) is derived from SRTM 1 arc-second
(~30m resolution) data downloaded via the `elevation` Python package.

```bash
uv add elevation                   # if not already installed
mkdir -p data/gis/elevation
eio clip -o data/gis/elevation/vastmanland.tif \
    --bounds 15.5 59.5 16.8 60.2
```

This downloads four SRTM tiles covering the region, merges them, and clips
to the bounding box. Output is a GeoTIFF (~10 MB) in WGS84 (EPSG:4326).

If `eio` fails with a cache error on the first attempt, clean and retry:

```bash
eio clean
eio clip -o data/gis/elevation/vastmanland.tif \
    --bounds 15.5 59.5 16.8 60.2
```

**Verify the raster loaded correctly:**

```bash
uv run python -c "
import rasterio
with rasterio.open('data/gis/elevation/vastmanland.tif') as src:
    print('CRS:', src.crs)
    print('Resolution:', src.res)
    print('Bounds:', src.bounds)
    print('Size:', src.width, 'x', src.height)
"
```

Expected output:
```
CRS: EPSG:4326
Resolution: (0.0002777..., 0.0002777...)
Bounds: BoundingBox(left=15.499..., bottom=59.499..., right=16.800..., top=60.200...)
Size: 4681 x 2521
```

### Verify OSM data

```bash
uv run python -m pipeline.gis.verify
```

## Layout

```
data/gis/
├── osm/
│   ├── sweden-*.osm.pbf               # full country download (optional, can delete)
│   └── vastmanland-50km.osm.pbf       # clipped to region — used by extraction
└── elevation/
    └── vastmanland.tif                # SRTM 30m raster — used by terrain queries
```

Both files are gitignored. Regenerate them locally using the steps above.

## Generating scenario context files

Once the data is in place, generate scenario JSONs with the extraction script:

```bash
uv run python -m pipeline.context.extract \
    --lat 59.8 --lon 16.1 \
    --output data/contexts/scenario_01/scenario_01.json
```

Each scenario folder should contain the JSON and a corresponding image:

```
data/contexts/
    scenario_01/
        scenario_01.json
        scenario_01.jpg
    scenario_02/
        scenario_02.json
        scenario_02.png
```

See `pipeline/context/README.md` for the full scenario list and coordinate rationale.

## Attribution

- **OSM**: ODbL — required attribution **"© OpenStreetMap contributors"**
  (<https://www.openstreetmap.org/copyright>)
- **SRTM**: NASA/USGS public domain — no attribution required but
  "SRTM data courtesy of the U.S. Geological Survey" is recommended