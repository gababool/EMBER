from PySide6.QtWidgets import QApplication, QSizePolicy, QVBoxLayout, QWidget, QMainWindow, QLabel
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from components.top_bar import TopBar

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMBER")
        self.resize(1400, 800)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.top_bar = TopBar()
        layout.addWidget(self.top_bar)

        # Example updates
        self.top_bar.set_version("V.0.3.1")
        self.top_bar.set_wind("5m/s nw")
        self.top_bar.set_gpu("GPU 43%")
        self.top_bar.set_mode("OFFLINE MODE")

app = QApplication()
window = MainWindow()
window.show()

app.exec() 