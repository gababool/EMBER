from PySide6.QtWidgets import QApplication, QMainWindow, QLabel
from PySide6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('Hello World Applicaiton')

        label = QLabel('Hello World')
        label.setAlignment(Qt.AlignCenter)

        self.setCentralWidget(label)

app = QApplication()
window = MainWindow()
window.show()

app.exec()