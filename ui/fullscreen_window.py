from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2 import QtCore, QtWidgets, QtGui

import ui.fullscreen_mode as fullscreen
import ui.wing_view as wing_view
import ui.drag as drag_pt
import sys, os

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
import cv2 as cv

img_default_size = (512, 512)

class FullscreenMode(QtWidgets.QWidget, fullscreen.Ui_FullscreenMode):
    def __init__(self, scene):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.scene = scene
        print("Now in fullscreen")
        self.wingview_layout.addWidget(self.scene)
        self.slider_feature_size.valueChanged.connect(self.update_feature_size)


    def update_feature_size(self):
        feature_size = self.slider_feature_size.value()
        print("Change feature size to {}".format(feature_size))
        self.scene.update_feature_size(feature_size*0.002+0.001)
