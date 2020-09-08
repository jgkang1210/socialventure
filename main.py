import cv2 as cv
from cv2 import aruco
import numpy as np

print(cv.__version__)
print(np.__version__)

# We use 250 ID
aruco_dict = aruco.Dictinoary_get(aruco.DICT_6X6_250)

