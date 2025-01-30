import sys
from PyQt5.QtWidgets import QApplication
from src.gui.gui import GUI


class App:
    def __init__(self, gui_path):
        self.gui = GUI(gui_path)
        self.gui.showMaximized()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    try:
        import qdarktheme
        print("pyqtdarktheme is installed.")
        qdarktheme.setup_theme()
    except ImportError:
        print("pyqtdarktheme is NOT installed.")

    gui = App("src/gui/main.ui")
    app.exec_()
