import numpy as np
import cv2

# this function detects faces and a set of trackable points
# within detected face regions
#
# the face detector is based on Haar-cascades and 
# the point detector is based on the Shi-Tomasi corner detector
#
# the function returns a bounding box (all four corner coordinates)
# for a face and the trackable points within the detected face region

def detect(im):
    # load the pretrained haar-cascade for face detection and detect
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    p0 = None
    for x, y, w, h in faces:
        # face bounding box
        roi = im[y:y+h, x:x+w]

        points = cv2.goodFeaturesToTrack(roi, maxCorners=80, qualityLevel=0.01, minDistance=7, blockSize=7)
        points = points.reshape((points.shape[0], 2))

        if p0 is not None:
            p0 = np.vstack((p0, points))
        else:
            p0 = points

        p0[:,0] += x
        p0[:,1] += y

    p0 = p0.reshape((p0.shape[0], 1, 2))

    x, y, w, h = faces[0,:]
    bboxPoints = np.array([[x,y],[x+w,y],[x+w,y+h], [x, y+h]])

    return bboxPoints, p0
