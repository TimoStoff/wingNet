import sys
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PySide2 import QtWidgets, QtGui, QtCore
from PySide2.QtGui import *

import numpy as np
import cv2 as cv

# Personal modules
from ui.drag import DraggablePoint


class WingView(FigureCanvas):

    """A canvas that updates itself every second with a new plot."""

    def __init__(self, image_path=None, keypoints=[], parent=None, marker_size=0.02, dpi=100, callback=None):

        self.fig = Figure(figsize=(1, 1), dpi=dpi)
        self.fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        self.axes = self.fig.add_subplot(111)
        self.scale = 1.0
        self.xlim = None
        self.ylim = None
        self.marker_size = marker_size

        self.setAxesSettings()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.callback = callback
        self.toolbar = NavigationToolbar(self, self)
#        self.toolbar.hide()

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.list_points = []
        self.updateImage(image_path=image_path, keypoints=keypoints, dpi=dpi)
        self.fig.canvas.mpl_connect('button_press_event', self.mouseClick)
        self.fig.canvas.mpl_connect('scroll_event', self.scrollMove)

    def scrollMove(self, event, base_scale=1.2):
        # get the current x and y limits
        cur_xlim = self.axes.get_xlim()
        cur_ylim = self.axes.get_ylim()
        # set the range
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location

        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            # do your processing 
            if event.button == 'up':
                self.scale = 1.0/base_scale
            elif event.button == 'down':
                self.scale = base_scale
            else:
                self.scale = 1.0
        self.axes.set_xlim([xdata - cur_xrange*self.scale,
            xdata + cur_xrange*self.scale])
        self.axes.set_ylim([ydata - cur_yrange*self.scale,
            ydata + cur_yrange*self.scale])
        self.xlim = self.axes.get_xlim()
        self.ylim = self.axes.get_ylim()
        self.axes.figure.canvas.draw()

    def mouseClick(self, event):
        if event.dblclick:
            self.clearFigure()
            self.axes.imshow(self.img)
            self.show()
        else:
            if len(self.list_points) < 8:
                self.list_points.append(DraggablePoint(self, x=event.xdata, y=event.ydata,
                    size=self.marker_size, img_shape=self.img_shape, callback=self.update_kpts))
                if len(self.list_points) == 8:
                    self.update_kpts()
        self.updateFigure()

    def get_kpts(self):
        kpts = []
        for pt in self.list_points:
            coords = pt.get_coords()
            kpts.extend(coords)
        return kpts


    def update_kpts(self):
        self.callback(self.get_kpts())


    def updateImage(self, image_path=None, keypoints=[], parent=None, dpi=100):
        self.clearFigure()
        self.list_points.clear()
        if image_path is None:
            img = np.zeros((256, 256))
        else:
            img = cv.imread(image_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.axes.imshow(img)
        self.img_shape = img.shape
        self.img = img

        self.show()
        self.plotDraggablePoints(keypoints=keypoints, size=self.marker_size, img_shape=img.shape)


    def plotDraggablePoints(self, keypoints, size=0.02, img_shape=(200, 200)):
        for keypoint in zip(keypoints[0::2], keypoints[1::2]):
            self.list_points.append(DraggablePoint(self, x=keypoint[0], y=keypoint[1], size=size, img_shape=img_shape, callback=self.update_kpts))
        self.updateFigure()


    def clearFigure(self):
        """Clear the graph"""
        self.axes.clear()
        self.setAxesSettings()
        del(self.list_points[:])
        self.updateFigure()


    def updateFigure(self):
        """Update the graph. Necessary, to call after each plot"""
        if self.ylim is not None and self.xlim is not None:
            self.axes.set_xlim(self.xlim)
            self.axes.set_ylim(self.ylim)
        self.draw()

    def refresh(self):
        kpts = self.get_kpts()
        self.clearFigure()
        self.axes.imshow(self.img)
        self.show()
        self.plotDraggablePoints(keypoints=kpts, size=self.marker_size, img_shape=self.img.shape)

    def update_feature_size(self, marker_size):
        self.marker_size = marker_size
        self.refresh()

    def setAxesSettings(self):
        self.axes.grid(False)
        self.axes.set_axis_off()
        self.axes.margins(0,0)

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    ex = MyGraph()
    sys.exit(app.exec_())
