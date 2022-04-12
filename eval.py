import numpy as np
import cv2
import math


# calculate diff of two RotationMatrix
def RotMatErr(R1, R2):
    R = np.matmul(R1, R2)
    trace = np.trace(R)
    Rdegree = math.acos((trace-1)/2)
    return abs(Rdegree)
