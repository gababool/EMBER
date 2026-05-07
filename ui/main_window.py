from pathlib import Path
import sys

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ui.assets.design import DARK_THEME, LIGHT_THEME, Theme
    from ui.components.pipeline_dashboard import PipelineDashboard
    from ui.components.top_bar import TopBar
    from ui.pipeline_worker import PipelineWorker
    from ui.system_monitor import SystemMonitor
except ImportError:
    from assets.design import DARK_THEME, LIGHT_THEME, Theme
    from components.pipeline_dashboard import PipelineDashboard
    from components.top_bar import TopBar
    from pipeline_worker import PipelineWorker
    from system_monitor import SystemMonitor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.theme = DARK_THEME

        self.setWindowTitle("EMBER")
        self.resize(1400, 900)

        self.central = QWidget()
        self.central.setObjectName("MainWindowCentral")
        self.setCentralWidget(self.central)

        self.main_layout = QVBoxLayout(self.central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.top_bar = TopBar(self.theme)
        self.top_bar.theme_button.clicked.connect(self.toggle_theme)
        self.main_layout.addWidget(self.top_bar)

        self.dashboard = PipelineDashboard(self.theme)
        self.dashboard.start_button.clicked.connect(self.start_demo)
        self.dashboard.wind_mode_select.currentIndexChanged.connect(
            self.update_wind_status
        )
        self.dashboard.wind_direction_select.currentIndexChanged.connect(
            self.update_wind_status
        )
        self.dashboard.wind_speed_input.valueChanged.connect(self.update_wind_status)
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("DashboardScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setWidget(self.dashboard)
        self.main_layout.addWidget(self.scroll_area, 1)

        # Example updates
        self.top_bar.set_version("V.0.3.1")
        self.update_wind_status()
        self.top_bar.set_mode("OFFLINE MODE")
        self.apply_theme(self.theme)

        self.pipeline_thread = None
        self.pipeline_worker = None
        self.system_monitor = SystemMonitor(parent=self)
        self.system_monitor.stats_updated.connect(self.update_system_stats)
        self.system_monitor.start()

    def start_demo(self):
        if self.pipeline_thread is not None:
            return

        self.dashboard.reset_demo()
        self.top_bar.set_mode("RUNNING DEMO")
        wind_config = self.dashboard.selected_wind_config()

        self.pipeline_thread = QThread(self)
        self.pipeline_worker = PipelineWorker(
            llm_model=self.dashboard.selected_llm_model(),
            wind_mode=wind_config["mode"],
            wind_direction=wind_config["direction"],
            wind_speed_mps=wind_config["speed_mps"],
        )
        self.pipeline_worker.moveToThread(self.pipeline_thread)

        self.pipeline_thread.started.connect(self.pipeline_worker.run)
        self.pipeline_worker.image_selected.connect(self.dashboard.set_selected_image)
        self.pipeline_worker.wind_updated.connect(self.update_generated_wind)
        self.pipeline_worker.event_received.connect(self.dashboard.handle_pipeline_event)
        self.pipeline_worker.failed.connect(self.dashboard.set_pipeline_error)
        self.pipeline_worker.finished.connect(self.pipeline_thread.quit)
        self.pipeline_worker.finished.connect(self.pipeline_worker.deleteLater)
        self.pipeline_thread.finished.connect(self.pipeline_thread.deleteLater)
        self.pipeline_thread.finished.connect(self._demo_finished)
        self.pipeline_thread.start()

    def _demo_finished(self):
        self.dashboard.set_demo_finished()
        self.top_bar.set_mode("OFFLINE MODE")
        self.pipeline_thread = None
        self.pipeline_worker = None

    def update_system_stats(self, stats):
        self.top_bar.set_system_stats(
            stats.cpu_percent,
            stats.ram_percent,
            stats.gpu_percent,
        )

    def update_wind_status(self):
        self.top_bar.set_wind(self.dashboard.wind_status_text())

    def update_generated_wind(self, wind_status):
        self.top_bar.set_wind(wind_status.get("text", "NO WIND"))
        self.dashboard.set_generated_wind(
            wind_status.get("direction"),
            wind_status.get("speed_mps"),
        )

    def resizeEvent(self, event):
        self.dashboard.close_open_popups()
        super().resizeEvent(event)

    def toggle_theme(self):
        next_theme = LIGHT_THEME if self.theme.name == "dark" else DARK_THEME
        self.apply_theme(next_theme)

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.top_bar.apply_theme(theme)
        self.dashboard.apply_theme(theme)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea#DashboardScrollArea {{
                background-color: {theme.bg_main};
                border: none;
            }}
            QScrollBar:vertical, QScrollBar:horizontal {{
                background-color: {theme.bg_main};
                border: none;
                width: 10px;
                height: 10px;
            }}
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
                background-color: {theme.bg_panel_alt};
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
                background-color: {theme.border};
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{
                width: 0px;
                height: 0px;
            }}
            QScrollBar::add-page, QScrollBar::sub-page {{
                background: none;
            }}
        """)
        self.central.setStyleSheet(f"""
            QWidget#MainWindowCentral {{
                background-color: {theme.bg_main};
            }}
        """)


if __name__ == "__main__":
    app = QApplication()
    window = MainWindow()
    window.show()
    app.exec()
