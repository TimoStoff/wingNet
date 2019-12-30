import sys
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5 import QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2 as cv

# Personnal modules
from drag import DraggablePoint


class MyGraph(FigureCanvas):

    """A canvas that updates itself every second with a new plot."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        self.axes.grid(False)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # To store the 2 draggable points
        self.list_points = []

        img = cv.imread("/home/timo/Data2/wingNet/wings/No_TPS/avi_wings/0_wings/fly1.jpg")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.axes.imshow(img)

        self.show()
        self.plotDraggablePoints([0.1, 0.1], [0.2, 0.2], size=[0.02, 0.02], img_shape=img.shape)
#        self.plotDraggablePoints([1, 1], [2, 2], [1, 1])


    def plotDraggablePoints(self, xy1, xy2, size=None, img_shape=(200, 200)):

        """Plot and define the 2 draggable points of the baseline"""

        # del(self.list_points[:])
        self.list_points.append(DraggablePoint(self, x=xy1[0], y=xy1[1], size=size[0], img_shape=img_shape))
        self.list_points.append(DraggablePoint(self, x=xy2[0], y=xy2[1], size=size[1], img_shape=img_shape))
        self.updateFigure()


    def clearFigure(self):

        """Clear the graph"""

        self.axes.clear()
        self.axes.grid(True)
        del(self.list_points[:])
        self.updateFigure()


    def updateFigure(self):

        """Update the graph. Necessary, to call after each plot"""

        self.draw()

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    ex = MyGraph()
    sys.exit(app.exec_())


#import numpy as np
#import math
#
#
#def square_distance_centroid(points, norm_factor):
#    points[0::2] = np.array(points[0::2])*norm_factor[0]
#    points[1::2] = np.array(points[1::2])*norm_factor[1]
#    pts = [np.array([x, y]) for x, y in zip(points[0::2], points[1::2])]
#    centroid = np.array([sum(points[0::2]) / len(pts), sum(points[1::2]) / len(pts)])
#
#    distances = np.array([math.sqrt(sum(x)) for x in ((pts[:] - centroid) ** 2)[:]])
#    distances = distances ** 2
#    metric = math.sqrt(sum(distances))
#    return metric
#
#
#points = [0,0,0,2,2,0,2,2]
#print(square_distance_centroid(points, np.array([2, 5])))
