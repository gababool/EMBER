"""Smoke test for local GIS data — verifies OSM .pbf is readable."""

from pathlib import Path

import geopandas as gpd
import pyogrio

REPO_ROOT = Path(__file__).resolve().parents[2]
OSM_FILE = REPO_ROOT / "data" / "gis" / "osm" / "vastmanland-50km.osm.pbf"


def verify_osm() -> bool:
    if not OSM_FILE.exists():
        print(f"[OSM]   file not found: {OSM_FILE}")
        return False

    pbf = OSM_FILE
    print(f"[OSM]   file:   {pbf.name} ({pbf.stat().st_size / 1_000_000:.0f} MB)")
    print(f"[OSM]   layers: {[name for name, _ in pyogrio.list_layers(pbf)]}")

    queries = {
        "roads":       ("lines",          "highway IS NOT NULL"),
        "water":       ("multipolygons",  "natural = 'water'"),
        "forest":      ("multipolygons",  "landuse = 'forest' OR natural = 'wood'"),
        "settlements": ("multipolygons",  "landuse = 'residential'"),
        "admin":       ("multipolygons",  "boundary = 'administrative'"),
    }
    for label, (layer, where) in queries.items():
        gdf = gpd.read_file(pbf, layer=layer, where=where)
        print(f"[OSM]   {label:<12} {len(gdf):>6}  ({layer})")
    return True


if __name__ == "__main__":
    osm_ok = verify_osm()
    print()
    print(f"OSM: {'OK' if osm_ok else 'MISSING'}")
