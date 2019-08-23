# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui',
# licensing of 'main_window.ui' applies.
#
# Created: Fri Aug 23 14:38:25 2019
#      by: pyside2-uic  running on PySide2 5.12.4
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(803, 616)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.splitter_image_list = QtWidgets.QSplitter(self.centralwidget)
        self.splitter_image_list.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_image_list.setObjectName("splitter_image_list")
        self.gv_wing_image = QtWidgets.QGraphicsView(self.splitter_image_list)
        self.gv_wing_image.setObjectName("gv_wing_image")
        self.tableWidget = QtWidgets.QTableWidget(self.splitter_image_list)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        self.verticalLayout.addWidget(self.splitter_image_list)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.btn_label_wings = QtWidgets.QPushButton(self.centralwidget)
        self.btn_label_wings.setObjectName("btn_label_wings")
        self.verticalLayout_2.addWidget(self.btn_label_wings)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_2.addWidget(self.progressBar)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 803, 25))
        self.menuBar.setObjectName("menuBar")
        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menuBar)
        self.menuEdit.setObjectName("menuEdit")
        MainWindow.setMenuBar(self.menuBar)
        self.actionAdd_Wings = QtWidgets.QAction(MainWindow)
        self.actionAdd_Wings.setObjectName("actionAdd_Wings")
        self.actionOpen_Existing_Project = QtWidgets.QAction(MainWindow)
        self.actionOpen_Existing_Project.setObjectName("actionOpen_Existing_Project")
        self.actionSet_Scale = QtWidgets.QAction(MainWindow)
        self.actionSet_Scale.setObjectName("actionSet_Scale")
        self.menuFile.addAction(self.actionAdd_Wings)
        self.menuFile.addAction(self.actionOpen_Existing_Project)
        self.menuEdit.addAction(self.actionSet_Scale)
        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuEdit.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "MainWindow", None, -1))
        self.tableWidget.horizontalHeaderItem(0).setText(QtWidgets.QApplication.translate("MainWindow", "Image Path", None, -1))
        self.tableWidget.horizontalHeaderItem(1).setText(QtWidgets.QApplication.translate("MainWindow", "Wing Area", None, -1))
        self.tableWidget.horizontalHeaderItem(2).setText(QtWidgets.QApplication.translate("MainWindow", "Scale [mm/pixel]", None, -1))
        self.btn_label_wings.setText(QtWidgets.QApplication.translate("MainWindow", "Compute Keypoints", None, -1))
        self.menuFile.setTitle(QtWidgets.QApplication.translate("MainWindow", "File", None, -1))
        self.menuEdit.setTitle(QtWidgets.QApplication.translate("MainWindow", "Edit", None, -1))
        self.actionAdd_Wings.setText(QtWidgets.QApplication.translate("MainWindow", "Add Wings", None, -1))
        self.actionOpen_Existing_Project.setText(QtWidgets.QApplication.translate("MainWindow", "Open Existing Project", None, -1))
        self.actionSet_Scale.setText(QtWidgets.QApplication.translate("MainWindow", "Set Scale", None, -1))

