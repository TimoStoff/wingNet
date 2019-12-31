# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gallery_window.ui',
# licensing of 'gallery_window.ui' applies.
#
# Created: Tue Dec 31 14:39:55 2019
#      by: pyside2-uic  running on PySide2 5.13.1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_GalleryWindow(object):
    def setupUi(self, GalleryWindow):
        GalleryWindow.setObjectName("GalleryWindow")
        GalleryWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(GalleryWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.scrollArea_gallery = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_gallery.setWidgetResizable(True)
        self.scrollArea_gallery.setObjectName("scrollArea_gallery")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 780, 533))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayoutWidget = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 781, 531))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout_gallery = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_gallery.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_gallery.setObjectName("gridLayout_gallery")
        self.scrollArea_gallery.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout.addWidget(self.scrollArea_gallery)
        GalleryWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(GalleryWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        GalleryWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(GalleryWindow)
        self.statusbar.setObjectName("statusbar")
        GalleryWindow.setStatusBar(self.statusbar)

        self.retranslateUi(GalleryWindow)
        QtCore.QMetaObject.connectSlotsByName(GalleryWindow)

    def retranslateUi(self, GalleryWindow):
        GalleryWindow.setWindowTitle(QtWidgets.QApplication.translate("GalleryWindow", "MainWindow", None, -1))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    GalleryWindow = QtWidgets.QMainWindow()
    ui = Ui_GalleryWindow()
    ui.setupUi(GalleryWindow)
    GalleryWindow.show()
    sys.exit(app.exec_())

