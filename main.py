from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2 import QtCore, QtWidgets, QtGui

import ui.main_window as main_window
import deploy_network as wing_net
import sys, os
import time

import data_loader.data_loaders as module_data

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
import cv2 as cv

img_default_size = (512, 512)

class WingSceneWidget(QtWidgets.QGraphicsScene):
    def __init__(self, parent):
        super(self.__class__, self).__init__(parent)
        self.keypoints = []
        self.annotating = False
        self.p_item = None
        self.circ_radius = 5
        self.circs = []

    def mouseDoubleClickEvent(self, event):
        print("New Annotation")
        self.annotating = True
        self.keypoints = []
        for circ in self.circs:
            self.removeItem(circ)

    def mousePressEvent(self, event):
        if self.annotating:
            x = event.scenePos().x()
            y = event.scenePos().y()
            self.keypoints.append([x, y])
            circ = QGraphicsEllipseItem(x- self.circ_radius, y- self.circ_radius,
                                        self.circ_radius*2,  self.circ_radius*2, self.p_item)
            circ.setPen(QPen(Qt.red, 2))
            circ.setFlag(QGraphicsItem.ItemIsMovable, True)
            circ.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.circs.append(circ)
            # self.parent().fitInView(self.sceneRect())

            print(self.keypoints)
            if len(self.keypoints) >= 8:
                self.annotating = False

        for circ in self.circs:
            if circ.rect().contains(event.scenePos()):
                circ.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        for circ in self.circs:
            if circ.rect().contains(event.scenePos()) or circ.isSelected():
                circ.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        for circ in self.circs:
            if circ.rect().contains(event.scenePos()) or circ.isSelected():
                circ.mouseReleaseEvent(event)

    def give_pitem(self, p_item):
        self.p_item = p_item


    def draw_keypoints(self, keypoints, image_size):
        for circ in self.circs:
            self.removeItem(circ)
        circs = []
        for x, y in zip(keypoints[0::2], keypoints[1::2]):
            x *= image_size[0]
            y *= image_size[1]
            circ = QGraphicsEllipseItem(x - self.circ_radius, y - self.circ_radius,
                                        self.circ_radius * 2, self.circ_radius * 2, self.p_item)
            circ.setPen(QPen(Qt.red, 2))
            circ.setFlag(QGraphicsItem.ItemIsMovable, True)
            circ.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.circs.append(circ)


class WingNet(QtWidgets.QMainWindow, main_window.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.folder_list = []
        self.image_paths = []
        self.wing_result = []
        self.image_current_size = img_default_size

        self.scene = WingSceneWidget(self.gv_wing_image)
        self.gv_wing_image.setScene(self.scene)
        self.gv_wing_image.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        self.btn_label_wings.setEnabled(False)
        self.btn_label_wings.clicked.connect(self.process_wings)
        self.actionAdd_Wings.triggered.connect(self.browse_folders)
        self.tableWidget.itemSelectionChanged.connect(self.selection_changed)
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)

    def browse_folders(self):
        self.tableWidget.clear()

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
            self.image_paths = module_data.get_image_paths(self.folder_list)
            for image_path in self.image_paths:
                row_position = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row_position)
                self.wing_result.append([image_path, [], "-"])
                self.tableWidget.setItem(row_position, 0, QTableWidgetItem(image_path))
                self.tableWidget.setItem(row_position, 1, QTableWidgetItem("-"))
                self.tableWidget.setItem(row_position, 2, QTableWidgetItem("-"))

        self.btn_label_wings.setEnabled(True)


    def selection_changed(self):
        if self.tableWidget.currentColumn() is 0:
            selected = self.tableWidget.selectedItems()[0].text()
            row = self.tableWidget.currentIndex().row()
            print(selected)
            image = cv.imread(selected)
            image = cv.resize(image, img_default_size)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_img = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
            pixmap = QtGui.QPixmap.fromImage(q_img)

            p_item = self.scene.addPixmap(pixmap)
            self.scene.give_pitem(p_item)

            kpts = self.wing_result[row][1]
            if len(kpts) == 16:
                self.scene.draw_keypoints(kpts, self.image_current_size)


    def update_table(self):
        for result, index in zip(self.wing_result, range(0, len(self.wing_result), 1)):
            print("{}: {} = {}".format(result[0], result[1], result[2]))
            self.tableWidget.setItem(index, 0, QTableWidgetItem(str(result[0])))
            self.tableWidget.setItem(index, 1, QTableWidgetItem(str(result[2])))
            self.tableWidget.setItem(index, 2, QTableWidgetItem(str(1)))

    def process_wings(self):
        print("Process wings")
        keypoint_generator = wing_net.WingKeypointsGenerator(self.image_paths)
        self.wing_result = keypoint_generator.process_images()
        self.update_table()



def main():
    app = QtWidgets.QApplication(sys.argv)
    form = WingNet()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
