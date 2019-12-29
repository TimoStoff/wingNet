import numpy as np
import math


def square_distance_centroid(points, norm_factor):
    points[0::2] = np.array(points[0::2])*norm_factor[0]
    points[1::2] = np.array(points[1::2])*norm_factor[1]
    pts = [np.array([x, y]) for x, y in zip(points[0::2], points[1::2])]
    centroid = np.array([sum(points[0::2]) / len(pts), sum(points[1::2]) / len(pts)])

    distances = np.array([math.sqrt(sum(x)) for x in ((pts[:] - centroid) ** 2)[:]])
    distances = distances ** 2
    metric = math.sqrt(sum(distances))
    return metric


points = [0,0,0,2,2,0,2,2]
print(square_distance_centroid(points, np.array([2, 5])))
