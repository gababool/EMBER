from dataclasses import dataclass

from PySide6.QtGui import QFont


# -----------------------
# Fonts
# -----------------------
FONT_FAMILY = "JetBrains Mono"

FONT_SIZE_XS = 9
FONT_SIZE_SM = 10
FONT_SIZE_MD = 12
FONT_SIZE_LG = 14
FONT_SIZE_XL = 22
FONT_SIZE_XXL = 28


def app_font(size: int, bold: bool = False) -> QFont:
    font = QFont(FONT_FAMILY, size)
    font.setBold(bold)
    return font


@dataclass(frozen=True)
class Theme:
    name: str
    bg_topbar: str
    bg_main: str
    bg_panel: str
    bg_panel_alt: str
    border: str
    text_primary: str
    text_muted: str
    accent_orange: str
    accent_blue: str
    accent_cyan: str
    accent_green: str
    accent_purple: str
    danger: str
    shadow: str
    map_bg: str
    map_water_fill: str
    map_water_stroke: str
    map_protected_fill: str
    map_protected_stroke: str
    map_road_major: str
    map_road_minor: str
    map_track: str
    map_power: str
    map_waterway: str
    map_settlement: str
    map_building: str


DARK_THEME = Theme(
    name="dark",
    bg_topbar="#17191d",
    bg_main="#1a1d22",
    bg_panel="#1d2025",
    bg_panel_alt="#242930",
    border="#272d35",
    text_primary="#edf2f6",
    text_muted="#aeb7c1",
    accent_orange="#ff8c2b",
    accent_blue="#00a6ff",
    accent_cyan="#00d7ff",
    accent_green="#49d86f",
    accent_purple="#d34dff",
    danger="#ff5a66",
    shadow="#111317",
    map_bg="#193620",
    map_water_fill="#245f73",
    map_water_stroke="#35a8c7",
    map_protected_fill="#774545",
    map_protected_stroke="#a05454",
    map_road_major="#926835",
    map_road_minor="#787266",
    map_track="#5d5648",
    map_power="#cd50ff7a",
    map_waterway="#00d7ff",
    map_settlement="#9eab09",
    map_building="#8a725c",
)

LIGHT_THEME = Theme(
    name="light",
    bg_topbar="#ffffff",
    bg_main="#f3f5f7",
    bg_panel="#ffffff",
    bg_panel_alt="#eef1f4",
    border="#d8dee6",
    text_primary="#303338",
    text_muted="#626b75",
    accent_orange="#ff7a1a",
    accent_blue="#0079c8",
    accent_cyan="#008dad",
    accent_green="#168a45",
    accent_purple="#8b37d6",
    danger="#cf2f43",
    shadow="#c6ccd4",
    map_bg="#7caf85",
    map_water_fill="#78c7db",
    map_water_stroke="#2f9fbd",
    map_protected_fill="#a88585",
    map_protected_stroke="#4f9960",
    map_road_major="#ff9f35",
    map_road_minor="#716755",
    map_track="#8d7d64",
    map_power="#8b37d6",
    map_waterway="#008dad",
    map_settlement="#168a45",
    map_building="#8d725a",
)

DEFAULT_THEME = DARK_THEME

# Backwards-compatible aliases for older components.
BG_TOPBAR = DEFAULT_THEME.bg_topbar
BG_MAIN = DEFAULT_THEME.bg_main

TEXT_PRIMARY = DEFAULT_THEME.text_primary
TEXT_MUTED = DEFAULT_THEME.text_muted

ACCENT_ORANGE = DEFAULT_THEME.accent_orange
ACCENT_BLUE = DEFAULT_THEME.accent_blue
ACCENT_CYAN = DEFAULT_THEME.accent_cyan
ACCENT_GREEN = DEFAULT_THEME.accent_green
ACCENT_PURPLE = DEFAULT_THEME.accent_purple
