import sys
from PyQt5.QtWidgets import QApplication
from src.gui.gui import GUI
import qdarktheme


class App:
    def __init__(self, gui_path):
        self.gui = GUI(gui_path)
        self.gui.showMaximized()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    gui = App("src/gui/main.ui")
    app.exec_()
