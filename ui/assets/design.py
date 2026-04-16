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


def app_font(size: int, bold: bool = False) -> QFont:
    font = QFont(FONT_FAMILY, size)
    font.setBold(bold)
    return font


# -----------------------
# Colors
# -----------------------
BG_TOPBAR = "#FFFFFF"#"#1b1d21"
BG_MAIN = "#FFFFFF"#"#23262b"

TEXT_PRIMARY = "#353637"
TEXT_MUTED = "#343639"

ACCENT_ORANGE = "#ff8c2b"
ACCENT_BLUE = "#00a6ff"
ACCENT_CYAN = "#00d7ff"
ACCENT_GREEN = "#49d86f"
ACCENT_PURPLE = "#d34dff"