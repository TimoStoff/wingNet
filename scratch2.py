from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2 import QtWidgets, QtGui

import ui.main_window as main_window
import sys
import data_loader.data_loaders as module_data

import cv2 as cv

img_default_size = (512, 512)


class WingSceneWidget(QtWidgets.QGraphicsScene):
    def __init__(self, parent):
        super(self.__class__, self).__init__(parent)
        self.keypoints = []
        self.annotating = False
        self.p_item = None
        self.circs = []

    def mouseDoubleClickEvent(self, event):
        print("New Annotation")
        self.annotating = True
        self.keypoints = []
        for circ in self.circs:
            self.removeItem(circ)

    def mousePressEvent(self, event):
        circ_radius = 5
        if self.annotating:
            x = event.scenePos().x()
            y = event.scenePos().y()
            self.keypoints.append([x, y])
            circ = QGraphicsEllipseItem(x-circ_radius, y-circ_radius, circ_radius*2, circ_radius*2, self.p_item)
            circ.setPen(QPen(Qt.red, 2))
            circ.setFlag(QGraphicsItem.ItemIsMovable, True)
            circ.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.circs.append(circ)

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


class WingNet(QtWidgets.QMainWindow, main_window.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.folder_list = []
        self.scene = WingSceneWidget(self.gv_wing_image)
        self.gv_wing_image.setScene(self.scene)
        self.gv_wing_image.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        self.btn_label_wings.clicked.connect(self.browse_folders)
        self.listWidget.currentItemChanged.connect(self.selection_changed)

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

        if self.folder_list:
            image_paths = module_data.get_image_paths(self.folder_list)
            for image_path in image_paths:
                print(image_path)
                self.listWidget.addItem(image_path)

        self.btn_edit_tps.setEnabled(False)
        self.btn_label_wings.setText("Start")
        self.btn_label_wings.clicked.disconnect()

    def selection_changed(self):
        selected = self.listWidget.currentItem().text()
        print(selected)
        image = cv.imread(selected)
        image = cv.resize(image, img_default_size)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(q_img)

        p_item = self.scene.addPixmap(pixmap)
        self.scene.give_pitem(p_item)


def main():
    app = QtWidgets.QApplication(sys.argv)
    form = WingNet()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
