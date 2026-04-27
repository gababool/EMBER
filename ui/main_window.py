from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

try:
    from ui.assets.design import DARK_THEME, LIGHT_THEME, Theme
    from ui.components.pipeline_dashboard import PipelineDashboard
    from ui.components.top_bar import TopBar
except ImportError:
    from assets.design import DARK_THEME, LIGHT_THEME, Theme
    from components.pipeline_dashboard import PipelineDashboard
    from components.top_bar import TopBar

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
        self.main_layout.addWidget(self.dashboard, 1)

        # Example updates
        self.top_bar.set_version("V.0.3.1")
        self.top_bar.set_wind("5m/s nw")
        self.top_bar.set_gpu("GPU 43%")
        self.top_bar.set_mode("OFFLINE MODE")
        self.apply_theme(self.theme)

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
