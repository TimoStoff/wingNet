import sys
import matplotlib
matplotlib.use("Qt5Agg")
from PySide2 import QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2 as cv

# Personnal modules
from ui.drag import DraggablePoint


class WingView(FigureCanvas):

    """A canvas that updates itself every second with a new plot."""

    def __init__(self, image_path=None, keypoints=[], parent=None, marker_size=0.02, dpi=100):

        self.fig = Figure(figsize=(1, 1), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        self.axes.grid(False)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.updateImage(image_path=image_path, keypoints=keypoints, marker_size=marker_size, dpi=dpi)
        self.mpl_connect('button_press_event', self.mouseClick)

    def mouseClick(self, event):
        print("clicked {} {}".format(event.xdata, event.ydata))
        if len(self.list_points) < 8:
            normed_pt = [event.xdata/self.img_shape[1], event.ydata/self.img_shape[0]]
            self.list_points.append(DraggablePoint(self, x=normed_pt[0], y=normed_pt[1], size=self.marker_size, img_shape=self.img_shape))

    def updateImage(self, image_path=None, keypoints=[], parent=None, marker_size=0.02, dpi=100):
        # To store the 2 draggable points
        self.list_points = []
        self.marker_size = marker_size
        if image_path is None:
            img = np.zeros((256, 256))
        else:
            img = cv.imread(image_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.axes.imshow(img)
        self.img_shape = img.shape
        print(self.img_shape)

        self.show()
        self.plotDraggablePoints(keypoints=keypoints, size=marker_size, img_shape=img.shape)


    def plotDraggablePoints(self, keypoints, size=0.02, img_shape=(200, 200)):

        """Plot and define the 2 draggable points of the baseline"""

        # del(self.list_points[:])
#        self.list_points.append(DraggablePoint(self, x=xy1[0], y=xy1[1], size=size[0], img_shape=img_shape))
#        self.list_points.append(DraggablePoint(self, x=xy2[0], y=xy2[1], size=size[1], img_shape=img_shape))
        for keypoint in keypoints:
            self.list_points.append(DraggablePoint(self, x=keypoint[0], y=keypoint[1], size=size, img_shape=img_shape))
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
