from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from ui.assets.design import (
        DEFAULT_THEME,
        FONT_SIZE_LG,
        FONT_SIZE_SM,
        FONT_SIZE_XL,
        Theme,
        app_font,
    )
    from ui.components.status_item import StatusItem
    from ui.components.stat_meter import StatMeter
    from ui.components.icon_utils import load_svg_asset
except ImportError:
    from assets.design import (
        DEFAULT_THEME,
        FONT_SIZE_LG,
        FONT_SIZE_SM,
        FONT_SIZE_XL,
        Theme,
        app_font,
    )
    from components.status_item import StatusItem
    from components.stat_meter import StatMeter
    from components.icon_utils import load_svg_asset


class TopBar(QWidget):
    def __init__(self, theme: Theme = DEFAULT_THEME, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setFixedHeight(72)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.main_row = QFrame()
        self.main_row.setFixedHeight(68)

        row_layout = QHBoxLayout(self.main_row)
        row_layout.setContentsMargins(20, 8, 20, 8)
        row_layout.setSpacing(16)

        logo_widget = QWidget()
        logo_layout = QVBoxLayout(logo_widget)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.setSpacing(2)

        logo_row = QWidget()
        logo_row_layout = QHBoxLayout(logo_row)
        logo_row_layout.setContentsMargins(0, 0, 0, 0)
        logo_row_layout.setSpacing(8)

        self.logo_icon_label = QLabel()
        self.logo_icon_label.setFixedSize(30, 30)
        self.logo_icon_label.setScaledContents(True)

        self.logo_label = QLabel("EMBER")
        self.logo_label.setFont(app_font(FONT_SIZE_XL, bold=True))

        logo_row_layout.addWidget(self.logo_icon_label)
        logo_row_layout.addWidget(self.logo_label)

        logo_layout.addWidget(logo_row)

        row_layout.addWidget(logo_widget)
        row_layout.addStretch()

        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(18)

        self.wind_item = StatusItem("⇄", "5m/s nw", icon_color=theme.accent_cyan)
        self.cpu_meter = StatMeter("CPU", "cpu.svg", theme.accent_cyan, theme)
        self.ram_meter = StatMeter("RAM", "memory-stick.svg", theme.accent_green, theme)
        self.gpu_meter = StatMeter("GPU", "gpu.svg", theme.accent_purple, theme)

        self.theme_button = QPushButton()
        self.theme_button.setFont(app_font(FONT_SIZE_SM, bold=True))
        self.theme_button.setCursor(Qt.PointingHandCursor)
        self.theme_button.setFixedHeight(32)

        self.signal_label = QLabel("⌁")
        self.signal_label.setFont(app_font(FONT_SIZE_LG, bold=True))

        status_layout.addWidget(self.wind_item)
        status_layout.addWidget(self.cpu_meter)
        status_layout.addWidget(self.ram_meter)
        status_layout.addWidget(self.gpu_meter)
        status_layout.addWidget(self.theme_button)
        status_layout.addWidget(self.signal_label)

        row_layout.addWidget(status_widget)

        self.accent_line = QFrame()
        self.accent_line.setFixedHeight(4)

        outer_layout.addWidget(self.main_row)
        outer_layout.addWidget(self.accent_line)

        self.apply_theme(theme)

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.main_row.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.bg_topbar};
                border: none;
            }}
        """)
        self.logo_icon_label.setPixmap(load_svg_asset("EMBER_LOGO.svg", 30))
        self.logo_label.setStyleSheet(self._label_style(theme.accent_orange))
        self.wind_item.setIconColor(theme.accent_cyan)
        self.wind_item.apply_theme(theme)
        self.cpu_meter.accent_color = theme.accent_cyan
        self.cpu_meter.apply_theme(theme)
        self.ram_meter.accent_color = theme.accent_green
        self.ram_meter.apply_theme(theme)
        self.gpu_meter.accent_color = theme.accent_purple
        self.gpu_meter.apply_theme(theme)
        self.signal_label.setStyleSheet(self._label_style(theme.accent_orange))
        self.theme_button.setText("LIGHT" if theme.name == "dark" else "DARK")
        self.theme_button.setStyleSheet(f"""
            QPushButton {{
                color: {theme.text_primary};
                background-color: {theme.bg_panel_alt};
                border: 1px solid {theme.border};
                border-radius: 4px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                border-color: {theme.accent_orange};
            }}
            QPushButton:pressed {{
                background-color: {theme.bg_panel};
            }}
        """)
        self.accent_line.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.accent_orange};
                border: none;
            }}
        """)

    def set_logo_text(self, text: str):
        self.logo_label.setText(text)

    def set_wind(self, text: str):
        self.wind_item.setText(text)

    def set_system_stats(self, cpu_percent, ram_percent, gpu_percent):
        self.cpu_meter.set_value(cpu_percent)
        self.ram_meter.set_value(ram_percent)
        self.gpu_meter.set_value(gpu_percent)

    @staticmethod
    def _label_style(color: str) -> str:
        return f"""
            QLabel {{
                color: {color};
                background: transparent;
                border: none;
            }}
        """
