from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2 import QtCore, QtWidgets, QtGui

import ui.main_window as main_window
import ui.drag as drag_pt
import deploy_network as wing_net
import sys, os
import math

import data_loader.data_loaders as module_data
import ui.wing_view as wing_view

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
import cv2 as cv
from PIL import Image
import pandas as pd

img_default_size = (512, 512)
NORM_FACTOR = [10, 10, 10, 10, 10, 10, 10, 10,
               10, 10, 10, 10, 10, 10, 10, 10]

class WingNet(QtWidgets.QMainWindow, main_window.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.folder_list = []
        self.image_paths = []
        self.autosave_path = ".tmp_project.wings"

        self.wing_result = pd.DataFrame(columns=['path', 'keypoints', 'scale', 'area', 'image_size'])
        self.image_current_size = img_default_size
        self.slider_image_size.setValue(img_default_size[0])
        self.scale = 1.0
        self.model_path = "/home/timo/Data2/wingNet/wingNet_models/wings_resnet34_weights"
        self.model_path = self.check_if_file_exists(self.model_path, "", message="Please Load Model")
        print("Loading model from {}".format(self.model_path))

        self.scene = wing_view.WingView(callback=self.update_kpts_callback)
        self.wingview_layout.addWidget(self.scene)
        self.update_feature_size()

        self.menuBar.setNativeMenuBar(False)
        self.actionSet_Scale.triggered.connect(self.add_scale)
        self.actionLoad_Model.triggered.connect(self.load_model_dialog)
        self.actionExport_CSV.triggered.connect(self.save_csv)
        self.btn_label_wings.setEnabled(False)
        self.btn_label_wings.clicked.connect(self.process_wings)
        self.tableWidget.itemSelectionChanged.connect(self.selection_changed)
        self.slider_image_size.valueChanged.connect(self.resize_image)
        self.actionAdd_Wings.triggered.connect(self.browse_folders)
        self.actionSave_Project.triggered.connect(self.save_project_dialog)
        self.actionOpen_Existing_Project.triggered.connect(self.load_project_dialog)
        self.slider_feature_size.valueChanged.connect(self.update_feature_size)

        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

        # self.load_project(self.autosave_path)

    def update_kpts_callback(self, kpts):
        if len(kpts) == 16:
            self.update_keypoints(kpts)


    def browse_folders(self):
        self.tableWidget.clear()

        file_dialog = QtWidgets.QFileDialog(filter="*")
        file_dialog.setFileMode(QtWidgets.QFileDialog.DirectoryOnly)
        file_dialog.ShowDirsOnly = False
        file_dialog.setNameFilter("JPG (*.jpg)")
        file_dialog.setOption(QFileDialog.DontUseNativeDialog, True);
        file_dialog.setOption(QFileDialog.ShowDirsOnly, False)
        file_view = file_dialog.findChild(QtWidgets.QListView, 'listView')

        # to make it possible to select multiple directories:
        if file_view:
            file_view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        f_tree_view = file_dialog.findChild(QtWidgets.QTreeView)
        if f_tree_view:
            f_tree_view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        all_paths = []
        if file_dialog.exec():
            all_paths = file_dialog.selectedFiles()

        print("Currently containing {}".format(all_paths))
        if all_paths:  # if user didn't pick a directory don't continue
            for f_path in all_paths:
                if os.path.isfile(f_path):
                    self.image_paths.append(f_path)
                elif os.path.isdir(f_path):
                    self.folder_list.append(f_path)

            self.image_paths = module_data.get_image_paths(self.folder_list)
            for image_path, table_idx in zip(self.image_paths, range(0, len(self.image_paths), 1)):
                self.add_image_to_table(image_path, table_idx)
        self.btn_label_wings.setEnabled(True)

    def add_image_to_table(self, image_path, table_index):
        num_rows = self.tableWidget.rowCount()
        if table_index >= num_rows:
            self.tableWidget.insertRow(num_rows)
            table_index = num_rows
        image = Image.open(image_path)
        self.wing_result.loc[len(self.wing_result)] = [image_path, [], self.scale, 0,
                                                       (image.size[0], image.size[1])]
        self.tableWidget.setItem(table_index, 0, QTableWidgetItem(image_path))
        self.tableWidget.setItem(table_index, 1, QTableWidgetItem("-"))
        self.tableWidget.setItem(table_index, 2, QTableWidgetItem(str(1.0 / self.scale)))
        return table_index

    def selection_changed(self):
        row = self.tableWidget.currentIndex().row()
        selected = self.tableWidget.item(row, 0).text()

        kpts = self.wing_result.at[row, "keypoints"]
        if len(kpts) > 16:
            kpts = []
        self.scene.updateImage(image_path=selected, keypoints=kpts)

    def update_table(self):
        for index, result in self.wing_result.iterrows():
            self.tableWidget.setItem(index, 0, QTableWidgetItem(str(result["path"])))
            self.tableWidget.setItem(index, 1, QTableWidgetItem(str(result["area"])))
            self.tableWidget.setItem(index, 2, QTableWidgetItem(str(1.0/result["scale"])))
        self.save_project(self.autosave_path)

    def process_wings(self):
        print("Process wings")
        keypoint_generator = wing_net.WingKeypointsGenerator(self.image_paths, model_path=self.model_path)
        output = keypoint_generator.process_images()
        for i in range(len(output)):
            image_size = self.wing_result.at[i, 'image_size']
            self.wing_result.at[i, "keypoints"] = self.normed_to_pixel_coords(output[i][1], image_size)
            area = self.compute_area(i)
            self.wing_result.at[i, "area"] = area
        self.update_table()

    def resize_image(self):
        pass
       # self.image_current_size = (self.slider_image_size.value(), self.slider_image_size.value())
       # self.selection_changed()

    def image_selected_key_event(self, e):
        self.tableWidget.keyPressEvent(e)

    def square_distance_centroid(self, points, norm_factor):
        points[0::2] = np.array(points[0::2]) * norm_factor[0]
        points[1::2] = np.array(points[1::2]) * norm_factor[1]
        pts = [np.array([x, y]) for x, y in zip(points[0::2], points[1::2])]
        centroid = np.array([sum(points[0::2])/len(pts), sum(points[1::2])/len(pts)])

        distances = np.array([math.sqrt(sum(x)) for x in ((pts[:]-centroid)**2)[:]])
        distances = distances**2
        metric = math.sqrt(sum(distances))
        return metric


    def compute_area(self, idx):
        use_polygon_area = False
        norm_factor = (self.wing_result.at[idx, "scale"],
                       self.wing_result.at[idx, "scale"])
        kpts = list(self.wing_result.at[idx, "keypoints"])
        if len(kpts) != 16:
            return
        if use_polygon_area:
            return self.shoelace_polygon_area(kpts, norm_factor)
        else:
            return self.square_distance_centroid(kpts, norm_factor)


    @staticmethod
    def shoelace_polygon_area(points, norm_factor):
        x = np.array(points[0::2])*norm_factor[0]
        y = np.array(points[1::2])*norm_factor[1]
        # print("area calc x={}, y={}".format(x,y))
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


    def update_keypoints(self, kpts):
        row = self.tableWidget.currentIndex().row()
        self.wing_result.at[row, "keypoints"] = kpts
        area = self.compute_area(row)
        self.wing_result.at[row, "area"] = area
        self.update_table()

    def add_scale(self):
        text, ok = QInputDialog.getText(self, 'Set Scale', 'Set Scale (pixels/mm):')
        if ok:
            self.scale = 1.0/float(text)
            rows = self.tableWidget.selectedIndexes()
            if rows:
                for row in rows:
                    self.wing_result.at[row.row(), "scale"] = self.scale
                    if len(self.wing_result.at[row.row(), "keypoints"])==16:
                        self.wing_result.at[row.row(), "area"] = self.compute_area(row.row())
            else:
                for i in range(0, len(self.wing_result), 1):
                    self.wing_result.at[i, "scale"] = self.scale
                    self.wing_result.at[i, "area"] = self.compute_area(i)
            self.update_table()

    def load_model_dialog(self):
        filename = QFileDialog.getOpenFileName(parent=self, caption='Load Model', dir='.')

        print("model loaded from {}".format(filename))
        if filename:
            self.model_path = filename

    def save_project_dialog(self):
        filename = QFileDialog.getSaveFileName(self, "Save file", "", ".wings")
        path = filename[0] + filename[1]
        self.save_project(path)

    def save_project(self, path):
        if path:
            print("save wings to {}".format(path))
            self.wing_result.to_pickle(path)

    def load_project_dialog(self):
        filename, ext = QFileDialog.getOpenFileName(parent=self, caption='Load Project', dir='.',
                                               filter="Wing File (*.wings)")
        self.load_project(filename)

    def load_project(self, path):
        print("loading from {}".format(path))
        if path and os.path.exists(path):
            project = pd.read_pickle(path)
            for img_path in project["path"]:
                if not os.path.exists(img_path):
                    if path is not self.autosave_path:
                        error_dialog = QErrorMessage(self)
                        error_dialog.setWindowTitle('Error Loading {}'.format(path))
                        error_dialog.showMessage(
                            "Unable to load project file {}, since it contains the image {}, which no longer exists.".format(path, img_path)
                        )
                    print("Image {} does not exist!".format(img_path))
                    return
            self.wing_result = project
            self.tableWidget.clear()
            for i in range(len(self.wing_result)):
                self.tableWidget.insertRow(i)
            self.update_table()
        if self.tableWidget.rowCount() > 0:
            self.btn_label_wings.setEnabled(True)

    def save_csv(self):
        filename = QFileDialog.getSaveFileName(self, "Save file", "", ".csv")
        if filename:
            print("save cvs to {}".format(filename))
            path = filename[0]+filename[1]
            df_expanded = pd.DataFrame(self.wing_result['keypoints'].values.tolist(),
                columns=['kp1_x', 'kp1_y', 'kp2_x', 'kp2_y', 'kp3_x', 'kp3_y', 'kp4_x', 'kp4_y',
                         'kp5_x', 'kp5_y', 'kp6_x', 'kp6_y', 'kp7_x', 'kp7_y', 'kp8_x', 'kp8_y'])
            dims = pd.DataFrame({'w':[x[0] for x in self.wing_result['image_size'][:]], 
                'h':[x[1] for x in self.wing_result['image_size'][:]]})
            df_expanded = pd.concat(
                    [self.wing_result['path'], dims['w'], dims['h'],
                    df_expanded['kp1_x'], df_expanded['kp1_y'],
                    df_expanded['kp2_x'], df_expanded['kp2_y'], df_expanded['kp3_x'], df_expanded['kp3_y'],
                    df_expanded['kp4_x'], df_expanded['kp4_y'], df_expanded['kp5_x'], df_expanded['kp5_y'],
                    df_expanded['kp6_x'], df_expanded['kp6_y'], df_expanded['kp7_x'], df_expanded['kp7_y'],
                    df_expanded['kp8_x'], df_expanded['kp8_y'], self.wing_result['scale'], self.wing_result['area']],
                    axis=1, keys=['path', 'img_width', 'img_height', 'kp1_x', 'kp1_y', 'kp2_x', 'kp2_y', 'kp3_x',
                    'kp3_y', 'kp4_x', 'kp4_y', 'kp5_x', 'kp5_y', 'kp6_x', 'kp6_y', 'kp7_x', 'kp7_y', 'kp8_x', 'kp8_y', 'scale', 'area'])
            # df_expanded.sort("path")
            df_expanded.to_csv(path, mode='w', index=False)

    def check_if_file_exists(self, file, extention, message):
        path = file+extention
        print("checking if {} exists".format(path))
        if os.path.exists(path):
            return path
        filename = QFileDialog.getOpenFileName(parent=self, caption=message, dir='.', filter=extention)
        if not filename:
            return self.check_if_file_exists(file, extention, message)
        return filename


    def normed_to_pixel_coords(self, kpts, img_size):
        kpts[0::2] *= img_size[0]
        kpts[1::2] *= img_size[1]
        return kpts

    def pixel_to_normed_coords(self, kpts, img_size):
        kpts[0::2] /= (img_size[1]*1.0)
        kpts[1::2] /= (img_size[0]*1.0)

    def update_feature_size(self):
        feature_size = self.slider_feature_size.value()
        print("Change feature size to {}".format(feature_size))
        self.scene.update_feature_size(feature_size*0.002+0.001)

def main():
    app = QtWidgets.QApplication(sys.argv)
    form = WingNet()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
