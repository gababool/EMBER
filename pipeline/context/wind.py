"""Wind helpers shared by extraction, UI, and tests."""

from __future__ import annotations

import hashlib

from pipeline.context.schemas import CompassBearing, Wind


MOCK_WIND_DIRECTIONS: tuple[tuple[int, CompassBearing], ...] = (
    (0, "N"),
    (45, "NE"),
    (90, "E"),
    (135, "SE"),
    (180, "S"),
    (225, "SW"),
    (270, "W"),
    (315, "NW"),
)
MOCK_WIND_DEGREES_BY_COMPASS: dict[CompassBearing, int] = {
    compass: degrees for degrees, compass in MOCK_WIND_DIRECTIONS
}
MOCK_WIND_MIN_SPEED_MPS = 1.0
MOCK_WIND_MAX_SPEED_MPS = 10.0
BENCHMARK_WIND_DIRECTION_COMPASS: CompassBearing = "SW"
BENCHMARK_WIND_SPEED_MPS = 6.5


def build_manual_wind(direction_compass: CompassBearing, speed_mps: float) -> Wind:
    return Wind(
        direction_degrees=MOCK_WIND_DEGREES_BY_COMPASS[direction_compass],
        direction_compass=direction_compass,
        speed_mps=speed_mps,
        source="manual",
    )


def build_mock_wind(
    key: str,
    direction_compass: CompassBearing | None = None,
    speed_mps: float | None = None,
) -> Wind:
    """Return deterministic semi-random regional wind for scenario/UI demos."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()

    if direction_compass is None:
        direction_degrees, direction_compass = MOCK_WIND_DIRECTIONS[
            digest[0] % len(MOCK_WIND_DIRECTIONS)
        ]
    else:
        direction_degrees = MOCK_WIND_DEGREES_BY_COMPASS[direction_compass]

    if speed_mps is None:
        speed_range_tenths = int(
            (MOCK_WIND_MAX_SPEED_MPS - MOCK_WIND_MIN_SPEED_MPS) * 10
        )
        speed_offset_tenths = (
            int.from_bytes(digest[1:3], "big") % (speed_range_tenths + 1)
        )
        speed_mps = MOCK_WIND_MIN_SPEED_MPS + speed_offset_tenths / 10

    return Wind(
        direction_degrees=direction_degrees,
        direction_compass=direction_compass,
        speed_mps=round(speed_mps, 1),
        source="mocked",
    )
