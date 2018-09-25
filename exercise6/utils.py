# Description:
#   Exercise6 utils.py.
#
# Copyright (C) 2018 Santiago Cortes, Juha Ylioinas
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.

import numpy as np
from types import *
from scipy.ndimage.interpolation import map_coordinates

# convert from rgb to grayscale image
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2125 * r + 0.7154 * g + 0.0721 * b

    return gray

# salt-and-pepper noise generator
def imnoise(img, mode, prob):
    imgn = img.copy()
    if mode == 'salt & pepper':
        assert (prob >= 0 and prob <= 1), "prob must be a scalar between 0 and 1"
        h, w = imgn.shape
        prob_sp = np.random.rand(h, w)
        imgn[prob_sp < prob] = 0
        imgn[prob_sp > 1 - prob] = 1

    return imgn

# Gaussian noise generator
def add_gaussian_noise(img, noise_sigma):
    temp_img = np.copy(img)
    h, w = temp_img.shape
    noise = np.random.randn(h, w) * noise_sigma
    noisy_img = temp_img + noise
    return noisy_img

# 2d Gaussian filter
def gaussian2(sigma, N=None):
 
    if N is None:
        N = 2*np.maximum(4, np.ceil(6*sigma))+1

    k = (N - 1) / 2.
            
    xv, yv = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    
    # 2D gaussian filter    
    g = 1/(2 * np.pi * sigma**2) * np.exp(-(xv**2 + yv**2) / (2 * sigma ** 2))

    # 1st order derivatives
    gx = -xv / (2 * np.pi * sigma**4) * np.exp(-(xv**2 + yv**2) / (2 * sigma ** 2))
    gy = -yv / (2 * np.pi * sigma**4) * np.exp(-(xv**2 + yv**2) / (2 * sigma**2)) 

    # 2nd order derivatives
    gxx = (-1 + xv**2 / sigma**2) * np.exp(-(xv**2 + yv**2) / (2*sigma**2)) / (2 * np.pi * sigma**4)
    gyy = (-1 + yv**2 / sigma**2) * np.exp(-(xv**2 + yv**2) / (2*sigma**2)) / (2 * np.pi * sigma**4)
    gxy = (xv * yv) / (2 * np.pi * sigma**6) * np.exp(-(xv**2 + yv**2) / (2*sigma**2))    

    return g, gx, gy, gxx, gyy, gxy

# fit an affine model between two 2d point sets
def affinefit(x, y):
    # Ordinary least squares (check wikipedia for further details):
    # 
    # Y                          = P*X_aug            % X_aug is in homogenous coords (one sample per col)
    # Y'                         = X_aug'*P'          % take transpose from both sides
    # X_aug*Y'                   = X_aug*X_aug'*P'    % multiply both sides from left by X_aug
    # inv(X_aug*X_aug')*X_aug*Y' = P'                 % multiply both sides from left by the inverse of X_aug*X_aug' 
    n = x.shape[0]
    x = x.T
    y = y.T
    x_aug = np.concatenate((x, np.ones((1, n))), axis=0)
    y_aug = np.concatenate((y, np.ones((1, n))), axis=0)
    xtx = np.dot(x_aug, x_aug.T)
    xtx_inv = np.linalg.inv(xtx)
    xtx_inv_x = np.dot(xtx_inv, x_aug)   
    P = np.dot(xtx_inv_x, y_aug.T)  
    A = P.T[0:2, 0:2]
    b = P.T[0:2, 2]
    
    return A, b

#  
def maxinterp(v):
    a = 0.5*v[0] - v[1] + 0.5*v[2]
    b = -0.5*v[0] + 0.5*v[2]
    c = v[1]
    
    loc = (-b/2.0/a) 
    m = np.dot(np.array([a,b,c]), np.array([loc**2, loc, 1]))
    return m, loc

def show_all_circles(I, cx, cy, rad, color='r', ln_wid=1.5):
    # I: image on top of which you want to display the circles
    # cx, cy: column vectors with x and y coordinates of circle centers
    # rad: column vector with radii of circles.
    # The sizes of cx, cy and rad must all be the same
    # color: optional paramtere specifying the color of the circles 
    #        to be displayed (red by default)    
    # ln_wid: line width of circles (1.5 by default)

    # Calculate circle points for each keypoint
    theta = np.arange(0, 2*np.pi+0.1, 0.1)
    cx1 = np.tile(cx, (theta.size, 1)).T
    cy1 = np.tile(cy, (theta.size, 1)).T
    rad1 = np.tile(rad, (theta.size, 1)).T
    theta = np.tile(theta, (cx1.shape[0], 1))
    X = cx1 + np.cos(theta) * rad1
    Y = cy1 + np.sin(theta) * rad1
    # Show image and plot circles
    plt.figure()
    plt.imshow(I, cmap='gray')
    for i in range(X.shape[0]):
        plt.plot(X[i, :], Y[i, :], color, linewidth=ln_wid)

def circle_points(cx, cy, rad):
    # Calculate circle points for each keypoint
    theta = np.arange(0, 2*np.pi+0.1, 0.1)
    x = cx + np.cos(theta) * rad
    y = cy + np.sin(theta) * rad

    return x, y

def matchFeatures(desc1, desc2):
    xTy = np.inner(desc1, desc2)
    
    xTx = np.expand_dims(np.diag(np.inner(desc1, desc1)), axis=1)
    xTx = np.tile(xTx, (1, desc2.shape[0]))

    yTy = np.expand_dims(np.diag(np.inner(desc2, desc2)), axis=0)
    yTy = np.tile(yTy,(desc1.shape[0], 1))
    
    # Vectorized implementation of an Euclidean distance computation
    # between two vector sets desc1 and desc2
    distmat = xTx + yTy - 2*xTy
    
    # Matching pairs
    ids1 = np.argmin(distmat, axis=1)
    ids2 = np.argmin(distmat, axis=0)

    pairs = []
    for k in range(desc1.shape[0]):
        if k == ids2[ids1[k]]:
            pairs.append(np.array([k, ids1[k]]))
    pairs = np.array(pairs)
    
    distmat_sorted = np.sort(distmat, axis=1)

    # Lowe's ratio test to filter out bad matches
    good_pairs = []
    for i in range(pairs.shape[0]):
        k = int(pairs[i,0])
        nearestd_1 = distmat_sorted[k,0]
        nearestd_2 = distmat_sorted[k,1]
        if nearestd_1 < 0.75*nearestd_2:
            good_pairs.append(pairs[i,:])
    
    good_pairs = np.array(good_pairs)
    
    return good_pairs

def imwarp(srcI, tform, xlims, ylims):
    xMin=np.amin(xlims[:,0])
    xMax=np.amax(xlims[:,1])
    yMin=np.amin(ylims[:,0])
    yMax=np.amax(ylims[:,1])

    # Width and height of panorama.
    width  = int(np.floor(xMax - xMin))
    height = int(np.floor(yMax - yMin))
    
    stepx = (xMax-xMin)/(width-1)
    stepy = (yMax-yMin)/(height-1)

    xv, yv = np.meshgrid(np.arange(xMin, xMax+stepx, stepx), 
                     np.arange(yMin, yMax+stepy, stepy))
    
    pts_proj_homog = np.dot(np.linalg.inv(tform), 
               np.vstack((xv.flatten(), yv.flatten(), np.ones((1, xv.size)))))
    pts_proj = pts_proj_homog[:2,:] / pts_proj_homog[2,:]
    
    xvt = pts_proj[0,:].reshape(xv.shape[0], xv.shape[1])
    yvt = pts_proj[1,:].reshape(yv.shape[0], yv.shape[1])

    warpedI = None
    
    if len(srcI.shape) == 3:
        for ch in range(3):
            warpedI_ = map_coordinates(srcI[:,:,ch], (yvt, xvt))
            if warpedI is not None:
                warpedI[:,:,ch] = warpedI_
            else:
                warpedI = np.zeros((warpedI_.shape[0], warpedI_.shape[1],3))
                warpedI[:,:,ch] = warpedI_
        for ch in range(3):
            warpedI[:,:,ch] = warpedI[:,:,ch] / np.amax(warpedI[:,:,ch]) 
    else:
        warpedI = map_coordinates(srcI, (yvt, xvt))

   

    return warpedI

def blend(img0, mask0, img1):
    output = np.zeros(img0.shape)
    if len(img0.shape) == 3:
        for ch in range(3):
            output[:,:,ch] = (img1[:,:,ch] * (1 - mask0)) + img0[:,:,ch]
    else:
        output = (img1 * (1 - mask0)) + img0
    return output
