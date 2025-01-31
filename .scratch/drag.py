import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np


class DraggablePoint:

    # http://stackoverflow.com/questions/21654008/matplotlib-drag-overlapping-points-interactively

    lock = None #  only one can be animated at a time

    def __init__(self, parent, x=0.1, y=0.1, size=0.1, draw_line=False, img_shape=(200, 200)):

        self.parent = parent
        self.img_shape = img_shape
        self.x = x*max(img_shape[0], img_shape[1])
        self.y = y*max(img_shape[0], img_shape[1])
        self.size = size*max(img_shape[0], img_shape[1])
        self.point = patches.Ellipse((self.x, self.y), self.size, self.size, fc='None', edgecolor='r')
        print("{}, {},{}, sz={}".format(self.point, self.x, self.y, self.size))
        parent.fig.axes[0].add_patch(self.point)
        self.press = None
        self.background = None
        self.connect()
        self.draw_line = draw_line

        if self.parent.list_points and self.draw_line:
            line_x = [self.parent.list_points[0].x, self.x]
            line_y = [self.parent.list_points[0].y, self.y]

            self.line = Line2D(line_x, line_y, color='r', alpha=0.5)
            parent.fig.axes[0].add_line(self.line)

    def connect(self):

        'connect to all the events we need'

        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)


    def on_press(self, event):

        if event.inaxes != self.point.axes: return
        if DraggablePoint.lock is not None: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        if self.draw_line:
            if self == self.parent.list_points[1]:
                self.line.set_animated(True)
            else:
                self.parent.list_points[1].line.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.point.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.point)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)


    def on_motion(self, event):

        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes: return
        self.point.center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)

        canvas = self.point.figure.canvas
        axes = self.point.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.point)

        if self.draw_line:
            if self == self.parent.list_points[1]:
                axes.draw_artist(self.line)
            else:
                self.parent.list_points[1].line.set_animated(True)
                axes.draw_artist(self.parent.list_points[1].line)

        self.x = self.point.center[0]
        self.y = self.point.center[1]

        if self.draw_line:
            if self == self.parent.list_points[1]:
                line_x = [self.parent.list_points[0].x, self.x]
                line_y = [self.parent.list_points[0].y, self.y]
                self.line.set_data(line_x, line_y)
            else:
                line_x = [self.x, self.parent.list_points[1].x]
                line_y = [self.y, self.parent.list_points[1].y]

            self.parent.list_points[1].line.set_data(line_x, line_y)

        # blit just the redrawn area
        canvas.blit(axes.bbox)


    def on_release(self, event):

        'on release we reset the press data'
        if DraggablePoint.lock is not self:
            return

        self.press = None
        DraggablePoint.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)
        if self.draw_line: 
            if self == self.parent.list_points[1]:
                self.line.set_animated(False)
            else:
                self.parent.list_points[1].line.set_animated(False)

        self.background = None

        # redraw the full figure
        self.point.figure.canvas.draw()

        self.x = self.point.center[0]
        self.y = self.point.center[1]
        print("new point location at {},{}".format(self.x, self.y))

    def disconnect(self):

        'disconnect all the stored connection ids'

        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)
