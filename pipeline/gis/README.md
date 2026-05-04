# GIS module — data setup

Local Swedish geodata that feeds operational context (roads, water, forest,
settlements) to the LLM stage via OSM. Elevation and terrain analysis are
deferred until Lantmäteriet elevation tiles arrive.

## Region

50 × 50 km box over the 2014 Hälleskogsbrännan wildfire area (Västmanland).

| | min | max |
|---|---|---|
| WGS84 lon | 15.75 | 16.65 |
| WGS84 lat | 59.70 | 60.15 |

## Setup

```bash
uv sync                            # Python deps
brew install osmium-tool           # for OSM clipping
mkdir -p data/gis/osm
```

**OSM** — download `sweden-latest.osm.pbf` from
<https://download.geofabrik.de/europe/sweden.html> into `data/gis/osm/`,
then clip:

```bash
cd data/gis/osm
osmium extract -b 15.75,59.70,16.65,60.15 \
  sweden-*.osm.pbf -o vastmanland-50km.osm.pbf
```

**Verify**:

```bash
uv run python -m pipeline.gis.verify
```

## Layout

```
data/gis/osm/
├── sweden-*.osm.pbf               # full country (optional)
└── vastmanland-50km.osm.pbf       # clipped to region
```

## Attribution

- OSM: ODbL — required attribution **"© OpenStreetMap contributors"**
  (<https://www.openstreetmap.org/copyright>)
