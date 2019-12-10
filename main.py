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
import pandas as pd

img_default_size = (512, 512)
NORM_FACTOR = [10, 10, 10, 10, 10, 10, 10, 10,
               10, 10, 10, 10, 10, 10, 10, 10]

class WingSceneWidget(QtWidgets.QGraphicsScene):
    def __init__(self, parent):
        super(self.__class__, self).__init__(parent)
        self.keypoints = []
        self.annotating = False
        self.p_item = None
        self.circ_radius = 5

    def mouseDoubleClickEvent(self, event):
        print("New Annotation")
        self.annotating = True
        self.keypoints = []
        self.remove_all_circles()

    def mousePressEvent(self, event):
        items = self.items(event.scenePos())
        if items and isinstance(items[0], QGraphicsEllipseItem):
            QtWidgets.QGraphicsScene.mousePressEvent(self, event)
            return
        if self.annotating:
            x = event.scenePos().x()
            y = event.scenePos().y()
            self.keypoints.append([x, y])
            circ = QGraphicsEllipseItem(0, 0, self.circ_radius * 2, self.circ_radius * 2, self.p_item)
            circ.setPen(QPen(Qt.red, 2))
            circ.setFlags(circ.flags() | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
            circ.setPos(x - self.circ_radius, y - self.circ_radius)
            if len(self.keypoints) >= 8:
                self.annotating = False
                self.update_keypoints()

    def update_keypoints(self):
        kpts = []
        for item in self.items():
            if isinstance(item, QGraphicsEllipseItem):
                c_pt = item.pos()
                kpts.extend((c_pt.x(), c_pt.y()))
        if len(kpts) == 16:
            self.parent().update_keypoints(kpts)

    def mouseReleaseEvent(self, event):
        items = self.items(event.scenePos())
        if items and isinstance(items[0], QGraphicsEllipseItem):
            QtWidgets.QGraphicsScene.mouseReleaseEvent(self, event)
            self.update_keypoints()
            return

    def give_pitem(self, p_item):
        self.p_item = p_item

    def draw_keypoints(self, keypoints, image_size):
        self.remove_all_circles()
        for x, y in zip(keypoints[0::2], keypoints[1::2]):
            x *= image_size[0]
            y *= image_size[1]
            circ = QGraphicsEllipseItem(0, 0, self.circ_radius * 2, self.circ_radius * 2, self.p_item)
            circ.setPen(QPen(Qt.red, 2))
            circ.setFlag(QGraphicsItem.ItemIsMovable, True)
            circ.setFlag(QGraphicsItem.ItemIsSelectable, True)
            circ.setPos(x - self.circ_radius, y - self.circ_radius)

    def remove_all_circles(self):
        for item in self.items():
            if isinstance(item, QGraphicsEllipseItem):
                self.removeItem(item)

class WingNet(QtWidgets.QMainWindow, main_window.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.folder_list = []
        self.image_paths = []
        # wing_result contains 0: image path, 1: keypoints array, 2: scale 3: area and 4: image size
        # self.wing_result = []
        # self."path" = 0
        # self."keypoints" = 1
        # self."scale" = 2
        # self."area" = 3
        # self."image_size" = 4
        self.wing_result = pd.DataFrame(columns=['path', 'keypoints', 'scale', 'area', 'image_size'])
        self.image_current_size = img_default_size
        self.slider_image_size.setValue(img_default_size[0])
        self.scale = 1.0

        self.scene = WingSceneWidget(self)
        self.gv_wing_image.setScene(self.scene)
        self.gv_wing_image.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        self.actionSet_Scale.triggered.connect(self.add_scale)
        self.actionExport_CSV.triggered.connect(self.save_csv)
        self.btn_label_wings.setEnabled(False)
        self.btn_label_wings.clicked.connect(self.process_wings)
        self.tableWidget.itemSelectionChanged.connect(self.selection_changed)
        self.slider_image_size.valueChanged.connect(self.resize_image)
        self.actionAdd_Wings.triggered.connect(self.browse_folders)

        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)

    def browse_folders(self):
        print("browse")
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
                image = cv.imread(image_path)
                self.wing_result.loc[len(self.wing_result)] = [image_path, [], self.scale, 0, (image.shape[0], image.shape[1])]
                # self.wing_result.append([image_path, [], self.scale, 0, (image.shape[0], image.shape[1])])
                self.tableWidget.setItem(row_position, 0, QTableWidgetItem(image_path))
                self.tableWidget.setItem(row_position, 1, QTableWidgetItem("-"))
                self.tableWidget.setItem(row_position, 2, QTableWidgetItem(str(1.0/self.scale)))

        self.btn_label_wings.setEnabled(True)

    def selection_changed(self):
        # if self.tableWidget.currentColumn() is 0:
        row = self.tableWidget.currentIndex().row()
        selected = self.tableWidget.item(row, 0).text()
        print(selected)
        image = cv.imread(selected)
        image = cv.resize(image, self.image_current_size)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(q_img)

        self.scene.clear()
        p_item = self.scene.addPixmap(pixmap)
        self.scene.give_pitem(p_item)

        kpts = self.wing_result.at[row, "keypoints"]
        if len(kpts) == 16:
            self.scene.draw_keypoints(kpts, self.image_current_size)

    def update_table(self):
        for index, result in self.wing_result.iterrows():
            # print("{}: {} = {}".format(result[0], result[1], result[2]))
            self.tableWidget.setItem(index, 0, QTableWidgetItem(str(result["path"])))
            self.tableWidget.setItem(index, 1, QTableWidgetItem(str(result["area"])))
            self.tableWidget.setItem(index, 2, QTableWidgetItem(str(1.0/result["scale"])))

    def process_wings(self):
        print("Process wings")
        keypoint_generator = wing_net.WingKeypointsGenerator(self.image_paths)
        output = keypoint_generator.process_images()
        for i in range(len(output)):
            # print(self.wing_result.at[i, "keypoints"])
            # print(output[i][1])
            self.wing_result.at[i, "keypoints"] = output[i][1]
            area = self.compute_area(i)
            self.wing_result.at[i, "area"] = area
        self.update_table()

    def resize_image(self):
        self.image_current_size = (self.slider_image_size.value(), self.slider_image_size.value())
        self.selection_changed()

    def compute_area(self, idx):
        norm_factor = (self.wing_result.at[idx, "image_size"][0] * self.wing_result.at[idx, "scale"],
                       self.wing_result.at[idx, "image_size"][1] * self.wing_result.at[idx, "scale"])
        # print(norm_factor)
        return self.shoelace_polygon_area(self.wing_result.at[idx, "keypoints"], norm_factor)

    @staticmethod
    def shoelace_polygon_area(points, norm_factor):
        x = np.array(points[0::2])*norm_factor[0]
        y = np.array(points[1::2])*norm_factor[1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def update_keypoints(self, kpts):
        row = self.tableWidget.currentIndex().row()

        kpts[0::2] = [x/self.image_current_size[0] for x in kpts[0::2]]
        kpts[1::2] = [x/self.image_current_size[1] for x in kpts[1::2]]

        area = self.compute_area(row)
        self.wing_result.at[row, "area"] = area
        self.tableWidget.setItem(row, 1, QTableWidgetItem(str(area)))

    def add_scale(self):
        text, ok = QInputDialog.getText(self, 'Set Scale', 'Set Scale (pixels/mm):')
        if ok:
            self.scale = 1.0/float(text)
            print(self.scale)
            rows = self.tableWidget.selectedIndexes()
            if rows:
                for row in rows:
                    self.wing_result.at[row.row(), "scale"] = self.scale
                    self.wing_result.at[row.row(), "area"] = self.compute_area(row.row())
            else:
                for i in range(0, len(self.wing_result), 1):
                    self.wing_result.at[i, "scale"] = self.scale
                    self.wing_result.at[i, "area"] = self.compute_area(i)
            self.update_table()

    def save_csv(self):
        filename = QFileDialog.getSaveFileName(self, "Save file", "", ".csv")

        print(filename)
        path = filename[0]+filename[1]
        self.wing_result.to_csv(path, mode='w', columns=['path','keypoints','scale','area'], index=False)
        # df = pd.DataFrame(self.wing_result["path"], self.wing_result["keypoints"].values.tolist(), columns=["name", "kp1_x", "kp1_y", "kp2_x",
        #                                                                           "kp2_y", "kp3_x", "kp3_y", "kp4_x",
        #                                                                           "kp4_y", "kp5_x", "kp5_y", "kp6_x",
        #                                                                           "kp6_y", "kp7_x", "kp7_y", "kp8_x",
        #                                                                           "kp8_y", "scale_mm/pixel", "area"])


def main():
    app = QtWidgets.QApplication(sys.argv)
    form = WingNet()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
