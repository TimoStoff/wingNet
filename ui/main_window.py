# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui',
# licensing of 'main_window.ui' applies.
#
# Created: Thu Jan  2 17:55:43 2020
#      by: pyside2-uic  running on PySide2 5.9.0~a1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(803, 616)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.vlayout_image = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.vlayout_image.setContentsMargins(0, 0, 0, 0)
        self.vlayout_image.setObjectName("vlayout_image")
        self.wingview_layout = QtWidgets.QVBoxLayout()
        self.wingview_layout.setObjectName("wingview_layout")
        self.vlayout_image.addLayout(self.wingview_layout)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_feature_size = QtWidgets.QLabel(self.layoutWidget)
        self.label_feature_size.setObjectName("label_feature_size")
        self.verticalLayout_5.addWidget(self.label_feature_size)
        self.slider_feature_size = QtWidgets.QSlider(self.layoutWidget)
        self.slider_feature_size.setSingleStep(1)
        self.slider_feature_size.setSliderPosition(20)
        self.slider_feature_size.setOrientation(QtCore.Qt.Horizontal)
        self.slider_feature_size.setObjectName("slider_feature_size")
        self.verticalLayout_5.addWidget(self.slider_feature_size)
        self.vlayout_image.addLayout(self.verticalLayout_5)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_image_size = QtWidgets.QLabel(self.layoutWidget)
        self.label_image_size.setObjectName("label_image_size")
        self.verticalLayout_4.addWidget(self.label_image_size)
        self.slider_image_size = QtWidgets.QSlider(self.layoutWidget)
        self.slider_image_size.setMaximum(1024)
        self.slider_image_size.setOrientation(QtCore.Qt.Horizontal)
        self.slider_image_size.setObjectName("slider_image_size")
        self.verticalLayout_4.addWidget(self.slider_image_size)
        self.vlayout_image.addLayout(self.verticalLayout_4)
        self.tableWidget = QtWidgets.QTableWidget(self.splitter)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        self.verticalLayout_3.addWidget(self.splitter)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_label_wings = QtWidgets.QPushButton(self.centralwidget)
        self.btn_label_wings.setObjectName("btn_label_wings")
        self.horizontalLayout.addWidget(self.btn_label_wings)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_2.addWidget(self.progressBar)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 803, 25))
        self.menuBar.setObjectName("menuBar")
        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")
        self.menuTools = QtWidgets.QMenu(self.menuBar)
        self.menuTools.setObjectName("menuTools")
        MainWindow.setMenuBar(self.menuBar)
        self.actionAdd_Wings = QtWidgets.QAction(MainWindow)
        self.actionAdd_Wings.setObjectName("actionAdd_Wings")
        self.actionOpen_Existing_Project = QtWidgets.QAction(MainWindow)
        self.actionOpen_Existing_Project.setObjectName("actionOpen_Existing_Project")
        self.actionSave_Project = QtWidgets.QAction(MainWindow)
        self.actionSave_Project.setObjectName("actionSave_Project")
        self.actionExport_CSV = QtWidgets.QAction(MainWindow)
        self.actionExport_CSV.setObjectName("actionExport_CSV")
        self.actionSet_Scale = QtWidgets.QAction(MainWindow)
        self.actionSet_Scale.setObjectName("actionSet_Scale")
        self.actionTest = QtWidgets.QAction(MainWindow)
        self.actionTest.setObjectName("actionTest")
        self.actionLoad_Model = QtWidgets.QAction(MainWindow)
        self.actionLoad_Model.setObjectName("actionLoad_Model")
        self.menuFile.addAction(self.actionAdd_Wings)
        self.menuFile.addAction(self.actionOpen_Existing_Project)
        self.menuFile.addAction(self.actionSave_Project)
        self.menuFile.addAction(self.actionExport_CSV)
        self.menuFile.addAction(self.actionLoad_Model)
        self.menuTools.addAction(self.actionSet_Scale)
        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuTools.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "MainWindow", None, -1))
        self.label_feature_size.setText(QtWidgets.QApplication.translate("MainWindow", "Feature Size", None, -1))
        self.label_image_size.setText(QtWidgets.QApplication.translate("MainWindow", "Image Size", None, -1))
        self.tableWidget.horizontalHeaderItem(0).setText(QtWidgets.QApplication.translate("MainWindow", "Image Path", None, -1))
        self.tableWidget.horizontalHeaderItem(1).setText(QtWidgets.QApplication.translate("MainWindow", "Wing Area", None, -1))
        self.tableWidget.horizontalHeaderItem(2).setText(QtWidgets.QApplication.translate("MainWindow", "Scale [mm/pixel]", None, -1))
        self.btn_label_wings.setText(QtWidgets.QApplication.translate("MainWindow", "Compute Keypoints", None, -1))
        self.menuFile.setTitle(QtWidgets.QApplication.translate("MainWindow", "File", None, -1))
        self.menuTools.setTitle(QtWidgets.QApplication.translate("MainWindow", "Tools", None, -1))
        self.actionAdd_Wings.setText(QtWidgets.QApplication.translate("MainWindow", "Add Wings", None, -1))
        self.actionOpen_Existing_Project.setText(QtWidgets.QApplication.translate("MainWindow", "Open Existing Project", None, -1))
        self.actionSave_Project.setText(QtWidgets.QApplication.translate("MainWindow", "Save Project", None, -1))
        self.actionExport_CSV.setText(QtWidgets.QApplication.translate("MainWindow", "Export CSV", None, -1))
        self.actionSet_Scale.setText(QtWidgets.QApplication.translate("MainWindow", "Set Scale", None, -1))
        self.actionTest.setText(QtWidgets.QApplication.translate("MainWindow", "Test", None, -1))
        self.actionLoad_Model.setText(QtWidgets.QApplication.translate("MainWindow", "Load Model", None, -1))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

