# Description:
#   Exercise9 python demo.
#
# Copyright (C) 2018 Santiago Cortes, Juha Ylioinas, Tapio Honka
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.

import argparse
import numpy as np
import cv2
from goodFeaturesToTrackFromFace import detect as ShiTomasi_detect
from skimage.transform import SimilarityTransform
from skimage.measure import ransac
#
# This demo illustrates an application of Lucas-Kanade optical flow
#
# Steps:
#   1) detect face region using pretrained haarcascade classifiers
#   2) detect good features to track from face region using Shi-Tomasi corner detector
#   3) track the points using the Lucas-Kanade optical flow
#
parser = argparse.ArgumentParser(description='This is a demo for illustrating an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids. See the detailed flow of operations in the source file.')
parser.add_argument('--input', default='./santi.avi', type=str, help='your input video (default: santi.avi)')

args = parser.parse_args()

# setup a video capture from file (webcam also possible, for that see OpenCV docs)
cap = cv2.VideoCapture(args.input)

# read the first frame from the video file and convert to grayscale
ret, frame = cap.read() 
old_frame = frame.copy()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# detect points to track, take a look at goodFeaturesToTrackFromFace.py
bboxPoints, points = ShiTomasi_detect(gray)

# create a mask image for drawing the trails of the tracked points
mask = np.zeros_like(old_frame)

# display the video and track the points
oldPoints = points
trackingAlive = True

while cap.isOpened():

    # get the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # esc breaks the loop, also wait 30ms between every frame
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # convert to grayscale
    gray_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # params for Lucas-Kanade optical flow
    winSize = 31#9
    maxLevel = 4#1
    maxCount = 4#2

    if trackingAlive == True:
        # track the points (note that some points may be lost)
        points, isFound, err = cv2.calcOpticalFlowPyrLK(gray, gray_new, 
                                            oldPoints, None, winSize = (winSize, winSize), maxLevel = maxLevel, 
                                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, maxCount, 0.03))

        visiblePoints = points[isFound==1]
        oldInliers = oldPoints[isFound==1]

        # need at least two points (otherwise tracks are lost)
        if visiblePoints.shape[0] >= 2:

            # estimate the geometric transformation between the old points and 
            # the new points and eliminate outliers
            tform, inliers = ransac((oldInliers, visiblePoints), SimilarityTransform, min_samples=2,
                               residual_threshold=2, max_trials=200)

            H1to2p = tform.params
            visiblePoints = visiblePoints[inliers, :]
            oldInliers = oldInliers[inliers, :]

            # apply the transformation to the bounding box points
            bboxPoints_homog = np.hstack((bboxPoints, np.ones((bboxPoints.shape[0], 1))))
            bboxPoints_new = np.dot(H1to2p, bboxPoints_homog.T)
            bboxPoints_new = bboxPoints_new[:2,:] / bboxPoints_new[2,:]
            bboxPoints_new = bboxPoints_new.T
            bboxPoints = bboxPoints_new

            bboxPoints_new = bboxPoints_new.astype(np.int)
            bboxPoints_new = bboxPoints_new.reshape((-1, 1, 2))

            # insert a bounding box around the object being tracked
            cv2.polylines(frame, [bboxPoints_new], True, (0, 255, 255), 3)

            # display tracked points
            for i, (new, old) in enumerate(zip(visiblePoints, oldInliers)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b),(c, d), (255,255,255), 2)
                frame = cv2.circle(frame, (a, b), 2, (255, 255, 255), -1)

            # visualize tracks
            mask = 0.7 * mask
            oldPoints = visiblePoints.reshape(-1, 1, 2)
            gray = gray_new.copy()

            # display the number of tracked points
            cv2.putText(frame, 'Number of tracked points: ' + str(visiblePoints.shape[0]), (20,30), 0, 1.1, (255,255,255))
            frame = cv2.add(frame, np.uint8(mask))

        else:
            trackingAlive = False

    # show current frame with possible tracks
    cv2.imshow('Lucas-Kanade tracked Demo', frame)


# close everything
cap.release()
cv2.destroyAllWindows()
