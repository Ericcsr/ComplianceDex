import pybullet as pb
import numpy as np
import cv2 as cv
import time

c = pb.connect(pb.GUI)

r = pb.loadURDF("assets/banana/banana.urdf")

i = 0

def getIntrinsicParams(proj_mat, w,h):
    f_x = proj_mat[0] * w / 2
    f_y = proj_mat[5] * h / 2
    c_x = (-proj_mat[2] * w + w) / 2
    c_y = (proj_mat[6] * h + h) / 2
    return np.array([w, h, f_x, f_y, c_x, c_y])

far = 2.0
near = 0.01

for i in range(300):
    projectionMat = pb.computeProjectionMatrixFOV(fov=20, aspect=1, nearVal=near, farVal=far)
    viewMat = pb.computeViewMatrix(cameraEyePosition=[0.5,0.5,0.5], cameraTargetPosition=[0,0,0], cameraUpVector=[0,0,1])
    print(projectionMat)
    img = pb.getCameraImage(640, 640, viewMatrix=viewMat, projectionMatrix=projectionMat)
    cv.imshow("img", near * far /(far - (far - near)*img[3]))
    cv.waitKey(1)
    depth_image = img[3]
    time.sleep(0.01)
np.savez("rgbd.npz", rgb=img[2], depth= near * far /(far - (far - near)*img[3]), intrinsic=getIntrinsicParams(projectionMat, 640, 640))