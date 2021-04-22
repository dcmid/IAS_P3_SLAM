import numpy as np
from slam_utils import map_correlation

im = np.array([[0,0,0,0,0],
               [1,0,1,0,0],
               [1,-1,1,0,0],
               [1,0,1,0,0]])

oc = np.array([[[1,0],[2,0],[3,0]],
               [[1,1],[2,1],[3,1]],
               [[1,2],[2,2],[3,2]]])

bp = np.array([[0,0,0],
               [0,1,0],
               [0,2,0]])

print(map_correlation(im, oc))