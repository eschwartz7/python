

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import random
import time
from SingleCalib import *



CHECKERBOARD = (6, 8)
data = glob.glob('*.jpg')
sample_size = 10
max_iters = 25
max_error = 1
min_consensus = 45

def RANSAC(data, max_iters, max_error, min_consensus, sample_size):

    """
    Implements RANSAC algorithm to get a best fit for our calibration parameters
    :param data: list of image files to be read
    :param min_images: minimum number of images needed to estimate intrinsic camera parameters
    :param max_iters: number of iterations for algorithm
    :param max_error: maximum allowable error for images to fit the model
    :param min_consensus:
    :return:
    """

    maybe_inliers = []
    largest_consensus = 0
    best_fit = None
    best_error = float('Inf')
    iter = 0
    num_images = len(data)
    obj_dict = {}

    #Run the algorithm a set number of times
    while iter < max_iters:
        print("Iteration Number " + str(iter))
        iter += 1

        #Take 3 random images from dataset
        sample_inliers = random.sample(data, sample_size)
        print("Sample Inliers: ", sample_inliers)

        #Make a function in the single calibration file that takes a random # of images and calibrates camera off of that
        print("Calibrating based off sample inliers:")
        rvecs, tvecs, mtx, dist, r_error = singleCalib(sample_inliers, chessboard_size)
        print("Model Based off Sample Inliers: ", mtx)
        
        also_inliers = []

        # Use the intrinsic parameters for every other image and find projection error
        for image in data:
            objpoints, imgpoints = getObjectPoints(image, chessboard_size)
            obj_dict[image] = [objpoints, imgpoints]
            new_r_error = reprojectionError(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
            print("Individual Reprojection Error: ", new_r_error)

            if new_r_error < max_error:
                also_inliers.append(image)

        consensus = also_inliers + sample_inliers
        print("Number of samples in consensus = " + str(len(consensus)))

        #If a certain number of images fit the model well, make a new calibration model based off all of the images in consensus
        if len(consensus) > min_consensus and len(consensus) > largest_consensus:
            largest_consensus = len(consensus)
            print("Consensus set: ", consensus)

            print("Creating new model:")
            rvecs, tvecs, mtx, dist, r_error = singleCalib(consensus, chessboard_size)

            """
            #Using the new model, find average reprojection error of all images in data set
            r_err = 0
            for image in data:
                r_err += reprojectionError(obj_dict[image][0], obj_dict[image][1], rvecs, tvecs, mtx, dist)
            new_error = r_err / len(consensus)
            """

            print("Overall Error: ", new_error)

        #If the error is better than previous models, make that model the best model
            if new_error < best_error:
                best_error = new_error
                best_fit =  {'mtx': mtx.tolist(), 'dist': dist.tolist(), 'reprojection error': r_error}


    # Save intrinsic and distortion parameters
    with open('bestCalibration.json', 'w') as fid:
        json.dump(best_fit, fid, indent=2)


    print("Best Model: ", best_fit)
    print("Model Error: ", best_error)


start_time = time.time()
RANSAC(data, max_iters, max_error, min_consensus, sample_size)
print("Runtime of %s minutes" % ((time.time() - start_time) / 60))