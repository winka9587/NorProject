import numpy as np
import cv2
import math


# calculate diff of two RotationMatrix
def RotMatErr(R1, R2):
    R = np.matmul(R1, R2)
    trace = np.trace(R)
    Rdegree = math.acos((trace-1)/2)
    return abs(Rdegree)


def TransErr(t1, t2):
    err_t_tmp = t1-t2
    err_t = np.sqrt(np.square(err_t_tmp[0])+np.square(err_t_tmp[1])+np.square(err_t_tmp[2]))
    return err_t

# get Err(s, R, t) of two pose
def poseErr(pose1, pose2):
    err_s = pose2['scale']/pose1['scale']
    err_t = TransErr(pose1['translation'], pose2['translation'])
    err_R = RotMatErr(pose1['rotation'], pose2['rotation'])
    pose_err = {}
    pose_err['translation'] = err_t
    pose_err['scale'] = err_s
    pose_err['rotation'] = err_R
    return pose_err

