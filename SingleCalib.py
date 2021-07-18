import os
import sys
import glob
import tqdm
import argparse
import importlib
import time
import json
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import cv2
import torch

sys.path.append('../')

chessboard_size = (6, 8)
images = glob.glob('*.jpg')

def singleCalib(images, chessboard_size):
    # Defining the dimensions of checkerboard

    CHECKERBOARD = chessboard_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #cv2.namedWindow("output", cv2.WINDOW_NORMAL)

    objpoints = []               # 3D world points
    imgpoints = []               # 2D image points

    img_count = 0

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    img_count = 0
    # Getting path of individual image stored in a given directory
    for image in tqdm.tqdm(images):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_FAST_CHECK)

        #Refine the pixel coordinates and display them on the images of checkerboard
        if ret == True:
            print(img_count)
            objpoints.append(objp)

            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            #cv2.imshow('output', img)
            #cv2.waitKey(1)
            img_count += 1

        """
        else:
            print("Corners not found, removing: ", image)
            os.remove(image)
        """

    cv2.destroyAllWindows()

    h, w = img.shape[:2]

    """
    Performing camera calibration by passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the detected corners (imgpoints)
    """

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix : \n", mtx)
    print("Distortion Coefficient : \n", dist)
    print("Rotation Vectors : \n", rvecs)
    print("Translation Vectors : \n", tvecs)

    print('Saving calibration data')
    calib_dict = {'mtx': mtx.tolist(),
                  'dist': dist.tolist()}
    with open('calibdata.json', 'w') as fid:
        json.dump(calib_dict, fid, indent=2)


    print('Computing reprojection error')
    mean_err = 0
    for idx in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[idx], rvecs[idx], tvecs[idx], mtx, dist)
        err = cv2.norm(imgpoints[idx],
                       imgpoints2,
                       cv2.NORM_L2) / len(imgpoints2)
        print("Image Reprojection Error: ", err)
        mean_err += err

    r_error = mean_err / len(objpoints)

    print('Reprojection error: ', r_error)

    return rvecs, tvecs, mtx, dist, r_error

#singleCalib(images, chessboard_size)



def getObjectPoints(image, chessboard_size):
    """
    Gets 3D world points and 2D image points for a single image
    """

    CHECKERBOARD = chessboard_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []               # 3D world points
    imgpoints = []               # 2D image points

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_FAST_CHECK)

    # Refine the pixel coordinates and display them on the images of checkerboard
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

    return objpoints, imgpoints


def reprojectionError(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """
    Computes reprojection error for a single image based off existing intrinsic parameters.
    """
    mean_err = 0
    for idx in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[idx], rvecs[idx], tvecs[idx], mtx, dist)
        err = cv2.norm(imgpoints[idx],
                       imgpoints2,
                       cv2.NORM_L2) / len(imgpoints2)
        mean_err += err

    r_error = mean_err / len(objpoints)

    return r_error