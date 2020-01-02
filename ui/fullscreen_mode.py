# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'fullscreen_mode.ui',
# licensing of 'fullscreen_mode.ui' applies.
#
# Created: Thu Jan  2 17:55:43 2020
#      by: pyside2-uic  running on PySide2 5.9.0~a1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_FullscreenMode(object):
    def setupUi(self, FullscreenMode):
        FullscreenMode.setObjectName("FullscreenMode")
        FullscreenMode.resize(400, 300)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(FullscreenMode)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.wingview_layout = QtWidgets.QVBoxLayout()
        self.wingview_layout.setObjectName("wingview_layout")
        self.verticalLayout.addLayout(self.wingview_layout)
        self.slider_feature_size = QtWidgets.QSlider(FullscreenMode)
        self.slider_feature_size.setOrientation(QtCore.Qt.Horizontal)
        self.slider_feature_size.setObjectName("slider_feature_size")
        self.verticalLayout.addWidget(self.slider_feature_size)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(FullscreenMode)
        QtCore.QMetaObject.connectSlotsByName(FullscreenMode)

    def retranslateUi(self, FullscreenMode):
        FullscreenMode.setWindowTitle(QtWidgets.QApplication.translate("FullscreenMode", "Fullscreen Mode", None, -1))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FullscreenMode = QtWidgets.QWidget()
    ui = Ui_FullscreenMode()
    ui.setupUi(FullscreenMode)
    FullscreenMode.show()
    sys.exit(app.exec_())

