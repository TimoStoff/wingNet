from PySide2 import QtCore, QtWidgets, QtGui
from PySide2.QtCore import QThread, SIGNAL
import ui.main_window as main_window
import ui.gallery_window as gallery_window
import sys, os
import time

import data_loader.data_loaders as module_data

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
import cv2 as cv


class WingNet(QtWidgets.QMainWindow, main_window.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.btn_label_wings.clicked.connect(self.browse_folders)
        self.listWidget.currentItemChanged.connect(self.selection_changed)
        self.splitter_image_list.dragMoveEvent.connect(self.splitter_moved)
        # self.dialog = WingNetGallery()
        self.folder_list = []

    def browse_folders(self):
        self.listWidget.clear()  # In case there are any existing elements in the list

        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.DirectoryOnly)
        file_dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        file_view = file_dialog.findChild(QtWidgets.QListView, 'listView')

        # to make it possible to select multiple directories:
        if file_view:
            file_view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        f_tree_view = file_dialog.findChild(QtWidgets.QTreeView)
        if f_tree_view:
            f_tree_view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        if file_dialog.exec():
            self.folder_list = file_dialog.selectedFiles()

        if self.folder_list:  # if user didn't pick a directory don't continue
            image_paths = module_data.get_image_paths(self.folder_list)
            for image_path in image_paths:
                print(image_path)
                self.listWidget.addItem(image_path)  # add file to the listWidget

        self.btn_edit_tps.setEnabled(False)
        self.btn_label_wings.setText("Start")
        self.btn_label_wings.clicked.disconnect()
        self.btn_label_wings.clicked.connect(self.display_gallery())

    def selection_changed(self):
        selected = self.listWidget.currentItem().text()
        print(selected)
        image = cv.imread(selected)
        image = cv.resize(image, (256, 256))
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(q_img)

        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)
        self.image_label.setSizePolicy(QtGui.QSizePolicy.horizontalPolicy())
        self.image_label.show()


def main():
    app = QtWidgets.QApplication(sys.argv)
    form = WingNet()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
