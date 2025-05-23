import datetime
import enum
import os

import cv2
import numpy as np
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QPushButton, QLabel, QTabWidget, QCheckBox, QSlider, QComboBox, \
    QSpinBox

from src.ancuti2018 import Ancuti2018, ANCUTI2018_STEPS
from src.mohansimon2020 import MohanSimon2020, MOHANSIMON2020_STEPS
from src.yang2011 import Yang2011, YANG2011_STEPS
from src.methods.metrics import get_uciqe


class Frameworks(enum.Enum):
    ANCUTI2018 = 0
    MOHANSIMON2020 = 1
    YANG2011 = 2


class GUI(QMainWindow):
    def __init__(self, gui_path):
        super(GUI, self).__init__()
        uic.loadUi(gui_path, self)
        self.setWindowIcon(QIcon("src/gui/icon.png"))

        self.path = None
        self.filename = None
        self.extension = None
        self.img_original = None
        self.result_set = None
        self.last_framework_run = None
        self.previous_uciqe = None

        # GENERAL WIDGETS
        self.btn_load = self.findChild(QPushButton, "btn_load")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_process = self.findChild(QPushButton, "btn_process")
        self.btn_process.clicked.connect(self.process_image)
        self.btn_default = self.findChild(QPushButton, "btn_default")
        self.btn_default.clicked.connect(self.reset_defaults)
        self.btn_save = self.findChild(QPushButton, "btn_save")
        self.btn_save.clicked.connect(self.save_results)

        self.disp_original = self.findChild(QLabel, "disp_original")
        self.disp_processed = self.findChild(QLabel, "disp_processed")
        self.tabs_settings = self.findChild(QTabWidget, "tabs_settings")
        self.lbl_metrics = self.findChild(QLabel, "lbl_metrics")

        # ANCUTI2018
        self.ancuti_combo_wb_method = self.findChild(QComboBox, "ancuti_combo_wb_method")
        self.ancuti_combo_fusion_method = self.findChild(QComboBox, "ancuti_combo_fusion_method")
        self.ancuti_check_do_precomp = self.findChild(QCheckBox, "ancuti_check_precomp")
        self.ancuti_slider_precomp_red = self.findChild(QSlider, "ancuti_slider_prec_red")
        self.ancuti_slider_precomp_blue = self.findChild(QSlider, "ancuti_slider_prec_blue")
        self.ancuti_slider_gamma = self.findChild(QSlider, "ancuti_slider_gamma")
        self.ancuti_spinner_msf_levels = self.findChild(QSpinBox, "ancuti_spinner_msf_levels")
        self.ancuti_lbl_prec_red = self.findChild(QLabel, "ancuti_lbl_prec_red")
        self.ancuti_lbl_prec_blue = self.findChild(QLabel, "ancuti_lbl_prec_blue")
        self.ancuti_lbl_gamma = self.findChild(QLabel, "ancuti_lbl_gamma")
        self.ancuti_combo_see_step = self.findChild(QComboBox, "ancuti_combo_see_step")

        # MOHANSIMON2020
        self.mohan_check_do_precomp = self.findChild(QCheckBox, "mohan_check_precomp")
        self.mohan_slider_precomp_red = self.findChild(QSlider, "mohan_slider_prec_red")
        self.mohan_slider_precomp_blue = self.findChild(QSlider, "mohan_slider_prec_blue")
        self.mohan_slider_gamma = self.findChild(QSlider, "mohan_slider_gamma")
        self.mohan_slider_clahe_clip_limit = self.findChild(QSlider, "mohan_slider_clahe_clip_limit")
        self.mohan_combo_wb_method = self.findChild(QComboBox, "mohan_combo_wb_method")
        self.mohan_spinner_tile_grid_size = self.findChild(QSpinBox, "mohan_spinner_tile_grid_size")
        self.mohan_combo_fusion_method = self.findChild(QComboBox, "mohan_combo_fusion_method")
        self.mohan_spinner_msf_levels = self.findChild(QSpinBox, "mohan_spinner_msf_levels")
        self.mohan_combo_see_step = self.findChild(QComboBox, "mohan_combo_see_step")
        self.mohan_lbl_prec_red = self.findChild(QLabel, "mohan_lbl_prec_red")
        self.mohan_lbl_prec_blue = self.findChild(QLabel, "mohan_lbl_prec_blue")
        self.mohan_lbl_gamma = self.findChild(QLabel, "mohan_lbl_gamma")
        self.mohan_lbl_clahe_clip_limit = self.findChild(QLabel, "mohan_lbl_clahe_clip_limit")

        # YANG2011
        self.yang_combo_pixels_considered = self.findChild(QComboBox, "yang_combo_pixels_considered")
        self.yang_combo_trans_smoothing_method = self.findChild(QComboBox, "yang_combo_trans_smoothing_method")
        self.yang_combo_wb_method = self.findChild(QComboBox, "yang_combo_wb_method")
        self.yang_combo_see_step = self.findChild(QComboBox, "yang_combo_see_step")
        self.yang_spinner_no_of_pixels = self.findChild(QSpinBox, "yang_spinner_no_of_pixels")
        self.yang_slider_perc_of_pixels = self.findChild(QSlider, "yang_slider_perc_of_pixels")
        self.yang_lbl_perc_of_pixels = self.findChild(QLabel, "yang_lbl_perc_of_pixels")
        self.yang_spinner_dcp_patch_size = self.findChild(QSpinBox, "yang_spinner_dcp_patch_size")
        self.yang_slider_perc_of_pixels = self.findChild(QSlider, "yang_slider_perc_of_pixels")
        self.yang_spinner_median_ksize = self.findChild(QSpinBox, "yang_spinner_median_ksize")

        self.init_combo_boxes()

        # CONNECTS
        self.ancuti_check_do_precomp.stateChanged.connect(self.toggle_precomp_sliders)
        self.ancuti_combo_fusion_method.currentIndexChanged.connect(self.toggle_msf_levels_spinner)
        self.ancuti_slider_precomp_red.valueChanged.connect(self.update_slider_label)
        self.ancuti_slider_precomp_blue.valueChanged.connect(self.update_slider_label)
        self.ancuti_slider_gamma.valueChanged.connect(self.update_slider_label)
        self.mohan_check_do_precomp.stateChanged.connect(self.toggle_precomp_sliders)
        self.mohan_combo_fusion_method.currentIndexChanged.connect(self.toggle_msf_levels_spinner)
        self.mohan_slider_precomp_red.valueChanged.connect(self.update_slider_label)
        self.mohan_slider_precomp_blue.valueChanged.connect(self.update_slider_label)
        self.mohan_slider_gamma.valueChanged.connect(self.update_slider_label)
        self.mohan_slider_clahe_clip_limit.valueChanged.connect(self.update_slider_label)
        self.yang_slider_perc_of_pixels.valueChanged.connect(self.update_slider_label)

        self.ancuti_combo_see_step.setCurrentIndex(len(ANCUTI2018_STEPS) - 1)
        self.mohan_combo_see_step.setCurrentIndex(len(MOHANSIMON2020_STEPS) - 1)
        self.yang_combo_see_step.setCurrentIndex(len(YANG2011_STEPS) - 1)

        self.ancuti_combo_see_step.currentIndexChanged.connect(self.change_displayed_step)
        self.mohan_combo_see_step.currentIndexChanged.connect(self.change_displayed_step)
        self.yang_combo_see_step.currentIndexChanged.connect(self.change_displayed_step)
        self.yang_combo_pixels_considered.currentIndexChanged.connect(self.toggle_pixels_considered)

    def load_image(self):
        try:
            self.path = QFileDialog.getOpenFileName(filter="Képfájlok (*.jpg *.jpeg *.png *.bmp)")[0]
            if not self.path:
                print("No file selected.")
                return

            self.filename, self.extension = os.path.splitext(os.path.basename(self.path))
            print(f"Selected file: {self.path}")

            self.img_original = cv2.imread(self.path)
            if self.img_original is None:
                print(f"Failed to load image: {self.path}")
                return

            self.display_image(self.img_original, 'top')

        except Exception as e:
            print(f"Error: {e}")

    def display_image(self, img: np.ndarray, position):
        try:
            if img is None:
                print("Invalid image.")
                return

            if np.max(img) <= 1:
                img = (img * 255).astype(np.uint8)

            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            if img.ndim != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            old_height, old_width = img.shape[:2]
            aspect_ratio = old_width / old_height
            new_height = self.disp_original.height()
            new_width = int(new_height * aspect_ratio)

            frame = cv2.resize(img, (new_width, new_height))
            image_displayed = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)

            if position == 'top':
                self.disp_original.setPixmap(QPixmap.fromImage(image_displayed))
            elif position == 'bottom':
                self.disp_processed.setPixmap(QPixmap.fromImage(image_displayed))

        except Exception as e:
            print(f"Error in display_image: {e}")

    def process_image(self):
        if self.img_original is None:
            print("No image selected.")
            return

        framework = self.tabs_settings.currentIndex()

        if framework == Frameworks.ANCUTI2018.value:
            self.result_set = Ancuti2018(
                self.img_original,
                self.ancuti_check_do_precomp.isChecked(),
                self.ancuti_slider_precomp_red.value() / 10.0,
                self.ancuti_slider_precomp_blue.value() / 10.0,
                self.ancuti_slider_gamma.value() / 10.0,
                self.ancuti_combo_wb_method.currentIndex(),
                self.ancuti_combo_fusion_method.currentIndex(),
                self.ancuti_spinner_msf_levels.value()
            )
            self.last_framework_run = Frameworks.ANCUTI2018
        elif framework == Frameworks.MOHANSIMON2020.value:
            clahe_clip_limit = self.findChild(QSlider, "mohan_slider_clahe_clip_limit").value() / 10.0
            clahe_tile_grid_size = self.findChild(QSpinBox, "mohan_spinner_tile_grid_size").value()

            self.result_set = MohanSimon2020(
                self.img_original,
                self.mohan_check_do_precomp.isChecked(),
                self.mohan_slider_precomp_red.value() / 10.0,
                self.mohan_slider_precomp_blue.value() / 10.0,
                self.mohan_slider_gamma.value() / 10.0,
                0,
                float(clahe_clip_limit),
                clahe_tile_grid_size,
                self.mohan_combo_fusion_method.currentIndex(),
                self.mohan_spinner_msf_levels.value()
            )
            self.last_framework_run = Frameworks.MOHANSIMON2020
        elif framework == Frameworks.YANG2011.value:
            # median_ksize = self.findChild(QSpinBox, "yang_spinner_median_ksize").value()

            self.result_set = Yang2011(
                self.img_original,
                self.yang_spinner_dcp_patch_size.value(),
                not bool(self.yang_combo_pixels_considered.currentIndex()),
                self.yang_spinner_no_of_pixels.value(),
                self.yang_slider_perc_of_pixels.value() / 10.0,
                self.yang_combo_trans_smoothing_method.currentIndex(),
                self.yang_spinner_median_ksize.value(),
                self.yang_combo_wb_method.currentIndex()
            )
            self.last_framework_run = Frameworks.YANG2011

        self.display_image(self.result_set[-1], 'bottom')
        self.update_metrics()
        self.btn_save.setEnabled(True)

    def save_results(self):
        try:
            save_path = QFileDialog.getExistingDirectory(self, "Select a directory", os.getcwd(), QFileDialog.ShowDirsOnly)

            print(f"Selected folder: {save_path}")

            save_path = os.path.join(
                save_path,
                f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.last_framework_run.name}_{self.filename}"
            )

            os.makedirs(save_path, exist_ok=False)
            i = 0

            print(self.result_set)
            for result in self.result_set:
                if result.dtype != np.uint8:
                    result = (result * 255).astype(np.uint8)

                cv2.imwrite(os.path.join(save_path, f"{'{:02d}'.format(i)}.jpg"), result)
                i += 1

        except Exception as e:
            print(f"Error: {e}")

    def update_metrics(self):
        original_uciqe = round(get_uciqe(self.img_original), 4)
        result_uciqe = round(get_uciqe(self.result_set[-1]), 4)
        self.lbl_metrics.setText(
            f"Original UCIQE: {original_uciqe}\t"
            f"Result UCIQE: {result_uciqe}\t"
            f"Previous result UCIQE: {self.previous_uciqe if self.previous_uciqe is not None else '-'}"
        )
        self.previous_uciqe = result_uciqe

    def reset_defaults(self):
        self.ancuti_check_do_precomp.setChecked(True)
        self.ancuti_slider_precomp_red.setValue(10)
        self.ancuti_slider_precomp_blue.setValue(0)
        self.ancuti_combo_wb_method.setCurrentIndex(0)
        self.ancuti_slider_gamma.setValue(20)
        self.ancuti_combo_fusion_method.setCurrentIndex(1)
        self.ancuti_spinner_msf_levels.setValue(3)

        self.mohan_check_do_precomp.setChecked(True)
        self.mohan_slider_precomp_red.setValue(10)
        self.mohan_slider_precomp_blue.setValue(0)
        self.mohan_combo_wb_method.setCurrentIndex(0)
        self.mohan_slider_gamma.setValue(20)
        self.mohan_slider_clahe_clip_limit.setValue(40)
        self.mohan_spinner_tile_grid_size.setValue(8)
        self.mohan_combo_fusion_method.setCurrentIndex(0)
        self.mohan_spinner_msf_levels.setValue(3)

        self.yang_spinner_dcp_patch_size.setValue(15)
        self.yang_spinner_no_of_pixels.setValue(1000)
        self.yang_combo_pixels_considered.setCurrentIndex(1)
        self.yang_slider_perc_of_pixels.setValue(1)
        self.yang_combo_trans_smoothing_method.setCurrentIndex(0)
        self.yang_spinner_median_ksize.setValue(5)
        self.yang_combo_wb_method.setCurrentIndex(0)

    def init_combo_boxes(self):
        wb_items = ["Gray World", "Iqbal Gray World"]
        fusion_items = ["Naive Fusion", "Multiscale Fusion"]
        yang_pixels_considered = ["Fixed amount", "Percentage"]
        yang_trans_smoothing = ["Median Filter (Yang et al.)", "Guided Filter (He et al.)"]

        self.ancuti_combo_wb_method.addItems(wb_items)
        self.ancuti_combo_fusion_method.addItems(fusion_items)
        self.mohan_combo_wb_method.addItems(wb_items)
        self.mohan_combo_fusion_method.addItems(fusion_items)
        self.yang_combo_pixels_considered.addItems(yang_pixels_considered)
        self.yang_combo_trans_smoothing_method.addItems(yang_trans_smoothing)
        self.yang_combo_wb_method.addItems(wb_items)

        self.ancuti_combo_see_step.addItems(ANCUTI2018_STEPS)
        self.mohan_combo_see_step.addItems(MOHANSIMON2020_STEPS)
        self.yang_combo_see_step.addItems(YANG2011_STEPS)

        # DEFAULT INDICES
        self.ancuti_combo_wb_method.setCurrentIndex(0)
        self.ancuti_combo_fusion_method.setCurrentIndex(1)

        self.mohan_combo_wb_method.setCurrentIndex(0)
        self.mohan_combo_fusion_method.setCurrentIndex(1)

        self.yang_combo_pixels_considered.setCurrentIndex(1)
        self.yang_spinner_no_of_pixels.setEnabled(False)
        self.yang_combo_trans_smoothing_method.setCurrentIndex(0)
        self.yang_combo_wb_method.setCurrentIndex(0)

    def toggle_precomp_sliders(self):
        self.ancuti_slider_precomp_red.setEnabled(self.ancuti_check_do_precomp.isChecked())
        self.ancuti_slider_precomp_blue.setEnabled(self.ancuti_check_do_precomp.isChecked())
        self.mohan_slider_precomp_red.setEnabled(self.mohan_check_do_precomp.isChecked())
        self.mohan_slider_precomp_blue.setEnabled(self.mohan_check_do_precomp.isChecked())

    def toggle_msf_levels_spinner(self):
        self.ancuti_spinner_msf_levels.setEnabled(self.ancuti_combo_fusion_method.currentIndex() == 1)
        self.mohan_spinner_msf_levels.setEnabled(self.mohan_combo_fusion_method.currentIndex() == 1)

    def update_slider_label(self):
        slider = self.sender()

        if slider.objectName() == "ancuti_slider_prec_red":
            value = self.ancuti_slider_precomp_red.value()
            self.ancuti_lbl_prec_red.setText(str(value / 10.0))
        elif slider.objectName() == "ancuti_slider_prec_blue":
            value = self.ancuti_slider_precomp_blue.value()
            self.ancuti_lbl_prec_blue.setText(str(value / 10.0))
        elif slider.objectName() == "ancuti_slider_gamma":
            value = self.ancuti_slider_gamma.value()
            self.ancuti_lbl_gamma.setText(str(value / 10.0))
        elif slider.objectName() == "mohan_slider_prec_red":
            value = self.mohan_slider_precomp_red.value()
            self.mohan_lbl_prec_red.setText(str(value / 10.0))
        elif slider.objectName() == "mohan_slider_prec_blue":
            value = self.mohan_slider_precomp_blue.value()
            self.mohan_lbl_prec_blue.setText(str(value / 10.0))
        elif slider.objectName() == "mohan_slider_gamma":
            value = self.mohan_slider_gamma.value()
            self.mohan_lbl_gamma.setText(str(value / 10.0))
        elif slider.objectName() == "mohan_slider_clahe_clip_limit":
            value = self.mohan_slider_clahe_clip_limit.value()
            self.mohan_lbl_clahe_clip_limit.setText(str(value / 10.0))
        elif slider.objectName() == "yang_slider_perc_of_pixels":
            value = self.yang_slider_perc_of_pixels.value()
            self.yang_lbl_perc_of_pixels.setText(str(value / 10) + "%")

    def change_displayed_step(self, value):
        if (self.result_set is not None and
                ((self.last_framework_run == Frameworks.ANCUTI2018 and self.tabs_settings.currentIndex() == 0) or
                 (self.last_framework_run == Frameworks.MOHANSIMON2020 and self.tabs_settings.currentIndex() == 1) or
                 (self.last_framework_run == Frameworks.YANG2011 and self.tabs_settings.currentIndex() == 2))
        ):
            self.display_image(self.result_set[value], "bottom")

    def toggle_pixels_considered(self):
        if self.yang_combo_pixels_considered.currentIndex() == 0:
            self.yang_slider_perc_of_pixels.setEnabled(False)
            self.yang_spinner_no_of_pixels.setEnabled(True)
        else:
            self.yang_slider_perc_of_pixels.setEnabled(True)
            self.yang_spinner_no_of_pixels.setEnabled(False)
