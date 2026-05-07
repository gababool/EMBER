from pathlib import Path
import sys

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

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
        screen = QApplication.primaryScreen().availableGeometry()
        if screen.width() < 1440 or screen.height() < 960:
            self.showMaximized()
        else:
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
        self.dashboard.rerun_button.clicked.connect(self.rerun_demo)
        self.dashboard.image_picked.connect(self.run_picked_image)
        self.main_layout.addWidget(self.dashboard, 1)

        # Example updates
        self.top_bar.set_version("V.0.3.1")
        self.top_bar.set_wind("5m/s nw")
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

        self.pipeline_thread = QThread(self)
        self.pipeline_worker = PipelineWorker(
            llm_model=self.dashboard.selected_llm_model(),
            prompt_file=self.dashboard.selected_prompt_file(),
            context_file=self.dashboard.selected_context_file(),
            skip_quality_screening=self.dashboard.skip_quality_screening(),
            use_annotation=self.dashboard.use_annotation(),
            label_filter=self.dashboard.selected_image_filter(),
        )
        self.pipeline_worker.moveToThread(self.pipeline_thread)

        self.pipeline_thread.started.connect(self.pipeline_worker.run)
        self.pipeline_worker.image_selected.connect(self.dashboard.set_selected_image)
        self.pipeline_worker.event_received.connect(self.dashboard.handle_pipeline_event)
        self.pipeline_worker.failed.connect(self.dashboard.set_pipeline_error)
        self.pipeline_worker.finished.connect(self.pipeline_thread.quit)
        self.pipeline_worker.finished.connect(self.pipeline_worker.deleteLater)
        self.pipeline_thread.finished.connect(self.pipeline_thread.deleteLater)
        self.pipeline_thread.finished.connect(self._demo_finished)
        self.pipeline_thread.start()

    def run_picked_image(self, image_path: str):
        if self.pipeline_thread is not None:
            return

        self.dashboard.reset_demo()
        self.top_bar.set_mode("RUNNING DEMO")

        self.pipeline_thread = QThread(self)
        self.pipeline_worker = PipelineWorker(
            llm_model=self.dashboard.selected_llm_model(),
            prompt_file=self.dashboard.selected_prompt_file(),
            context_file=self.dashboard.selected_context_file(),
            skip_quality_screening=self.dashboard.skip_quality_screening(),
            use_annotation=self.dashboard.use_annotation(),
            fixed_image_path=image_path,
        )
        self.pipeline_worker.moveToThread(self.pipeline_thread)

        self.pipeline_thread.started.connect(self.pipeline_worker.run)
        self.pipeline_worker.image_selected.connect(self.dashboard.set_selected_image)
        self.pipeline_worker.event_received.connect(self.dashboard.handle_pipeline_event)
        self.pipeline_worker.failed.connect(self.dashboard.set_pipeline_error)
        self.pipeline_worker.finished.connect(self.pipeline_thread.quit)
        self.pipeline_worker.finished.connect(self.pipeline_worker.deleteLater)
        self.pipeline_thread.finished.connect(self.pipeline_thread.deleteLater)
        self.pipeline_thread.finished.connect(self._demo_finished)
        self.pipeline_thread.start()

    def rerun_demo(self):
        image_path = self.dashboard.selected_rerun_image()
        if self.pipeline_thread is not None or image_path is None:
            return

        self.dashboard.reset_demo()
        self.top_bar.set_mode("RUNNING DEMO")

        self.pipeline_thread = QThread(self)
        self.pipeline_worker = PipelineWorker(
            llm_model=self.dashboard.selected_llm_model(),
            prompt_file=self.dashboard.selected_prompt_file(),
            context_file=self.dashboard.selected_context_file(),
            skip_quality_screening=self.dashboard.skip_quality_screening(),
            use_annotation=self.dashboard.use_annotation(),
            fixed_image_path=image_path,
            label_filter=self.dashboard.selected_image_filter(),
        )
        self.pipeline_worker.moveToThread(self.pipeline_thread)

        self.pipeline_thread.started.connect(self.pipeline_worker.run)
        self.pipeline_worker.image_selected.connect(self.dashboard.set_selected_image)
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

    def toggle_theme(self):
        next_theme = LIGHT_THEME if self.theme.name == "dark" else DARK_THEME
        self.apply_theme(next_theme)

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.top_bar.apply_theme(theme)
        self.dashboard.apply_theme(theme)
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
