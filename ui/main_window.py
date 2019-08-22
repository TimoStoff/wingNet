# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui',
# licensing of 'main_window.ui' applies.
#
# Created: Thu Aug 22 18:44:02 2019
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
        self.image_label = QtWidgets.QLabel(self.splitter_image_list)
        self.image_label.setObjectName("image_label")
        self.listWidget = QtWidgets.QListWidget(self.splitter_image_list)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout.addWidget(self.splitter_image_list)
        self.h_layout_buttons = QtWidgets.QHBoxLayout()
        self.h_layout_buttons.setObjectName("h_layout_buttons")
        self.btn_edit_tps = QtWidgets.QPushButton(self.centralwidget)
        self.btn_edit_tps.setObjectName("btn_edit_tps")
        self.h_layout_buttons.addWidget(self.btn_edit_tps)
        self.btn_label_wings = QtWidgets.QPushButton(self.centralwidget)
        self.btn_label_wings.setObjectName("btn_label_wings")
        self.h_layout_buttons.addWidget(self.btn_label_wings)
        self.verticalLayout.addLayout(self.h_layout_buttons)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 803, 25))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "MainWindow", None, -1))
        self.image_label.setText(QtWidgets.QApplication.translate("MainWindow", "                                                                                                           ", None, -1))
        self.btn_edit_tps.setText(QtWidgets.QApplication.translate("MainWindow", "Edit Existing TPS", None, -1))
        self.btn_label_wings.setText(QtWidgets.QApplication.translate("MainWindow", "Label New Wings", None, -1))

