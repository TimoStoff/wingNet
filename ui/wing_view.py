import sys
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide2 import QtWidgets, QtGui

import numpy as np
import cv2 as cv

# Personal modules
from ui.drag import DraggablePoint


class WingView(FigureCanvas):

    """A canvas that updates itself every second with a new plot."""

    def __init__(self, image_path=None, keypoints=[], parent=None, marker_size=0.02, dpi=100, callback=None):

        self.fig = Figure(figsize=(1, 1), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        self.axes.grid(False)
        self.axes.set_axis_off()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.callback = callback

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.list_points = []
        self.updateImage(image_path=image_path, keypoints=keypoints, marker_size=marker_size, dpi=dpi)
        self.mpl_connect('button_press_event', self.mouseClick)


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

    def update_kpts(self):
        kpts = []
        for pt in self.list_points:
            coords = pt.get_coords()
            kpts.extend(coords)
        self.callback(kpts)


    def updateImage(self, image_path=None, keypoints=[], parent=None, marker_size=0.02, dpi=100):
        self.clearFigure()
        self.list_points.clear()
        self.marker_size = marker_size
        if image_path is None:
            img = np.zeros((256, 256))
        else:
            img = cv.imread(image_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.axes.imshow(img)
        self.img_shape = img.shape
        self.img = img

        self.show()
        self.plotDraggablePoints(keypoints=keypoints, size=marker_size, img_shape=img.shape)


    def plotDraggablePoints(self, keypoints, size=0.02, img_shape=(200, 200)):
        for keypoint in zip(keypoints[0::2], keypoints[1::2]):
            self.list_points.append(DraggablePoint(self, x=keypoint[0], y=keypoint[1], size=size, img_shape=img_shape, callback=self.update_kpts))
        self.updateFigure()


    def clearFigure(self):
        """Clear the graph"""
        self.axes.clear()
        self.axes.grid(False)
        self.axes.set_axis_off()
        del(self.list_points[:])
        self.updateFigure()


    def updateFigure(self):

        """Update the graph. Necessary, to call after each plot"""

        self.draw()

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    ex = MyGraph()
    sys.exit(app.exec_())
