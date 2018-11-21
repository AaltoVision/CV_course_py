import numpy as np
from pyflann import *

def kpsObjToArray(kp):
    kps = np.array([p.pt for p in kp])
    kps_rad = np.array([p.size / 2 for p in kp]) # rad==scale
    kps = np.hstack([kps,kps_rad[:,np.newaxis]])
    return kps

def findNeighbours(descrs1, descrs2, numNeighbors=1):
    flann = FLANN()
    # "linear" : brute-force search
    ind, dist = flann.nn(descrs2, descrs1, numNeighbors, algorithm="linear",
        trees=8, checks=100)
    return ind, dist

def plotMatches(ax, im1, im2, kps1, kps2, pairs):
    ax.imshow(np.hstack((im1, im2)), cmap="gray")
    t = np.arange(0, 2*np.pi, 0.1)

    # Display matches
    colors = ['C{}'.format(i) for i in range(10)]

    for k in range(pairs.shape[0]):
        ic = k % 10
        
        pid1 = pairs[k, 0]
        pid2 = pairs[k, 1]
    
        loc1 = kps1[int(pid1),0:2]
        r1 = kps1[int(pid1), 2] * 3 # Treble the radius for seeing the keypoints better
        loc2 = kps2[int(pid2),0:2]
        r2 = kps2[int(pid2), 2] * 3

        ax.plot(loc1[0]+r1*np.cos(t), loc1[1]+r1*np.sin(t), 'c-', linewidth=1)
        ax.plot(loc2[0]+r2*np.cos(t)+im1.shape[1], loc2[1]+r2*np.sin(t), 'c-', linewidth=1)
        ax.plot([loc1[0], loc2[0]+im1.shape[1]], [loc1[1], loc2[1]], color='{}'.format(colors[ic]), linestyle='-')
        
# Direct linear transformation (DLT) is an algorithm which 
# solves a set of variables from a set of similarity relations

def camcalibDLT(Xworld, Xim):
    N = Xworld.shape[0]
    A = np.zeros((1,12))
    for i in range(N):
        tmp = np.hstack((np.zeros((4)), Xworld[i,:], -Xim[i,1]*Xworld[i,:]))
        tmp2 = np.hstack((Xworld[i,:], np.zeros(4), -Xim[i,0]*Xworld[i,:]))
        A = np.vstack((A,tmp,tmp2))
    A = np.delete(A,0,0)
        
    M = np.dot(A.T, A)
    
    u,s,v = np.linalg.svd(M)
    idmin = np.argmin(s)
    ev = v[idmin]
    P = np.reshape(ev, (3,4))
    
    return P

from pylab import *

# vgg_contreps  Contraction with epsilon tensor.
#
# B = vgg_contreps(A) is tensor obtained by contraction of A with epsilon tensor.
# However, it works only if the argument and result fit to matrices, in particular:
#
# - if A is row or column 3-vector ...  B = [A]_x
# - if A is skew-symmetric 3-by-3 matrix ... B is row 3-vector such that A = [B]_x
# - if A is skew-symmetric 4-by-4 matrix ... then A can be interpreted as a 3D line Pluecker matrix
#                                               skew-symmetric 4-by-4 B as its dual Pluecker matrix.
# - if A is row 2-vector ... B = [0 1; -1 0]*A', i.e., A*B=eye(2)
# - if A is column 2-vector ... B = A'*[0 1; -1 0], i.e., B*A=eye(2)
#
# It is vgg_contreps(vgg_contreps(A)) = A.

# werner@robots.ox.ac.uk, Oct 2001


def vgg_contreps(X):
    if prod(shape(X)) == 3:
        Y = array([[0, X[2], -X[1]], 
                   [-X[2], 0, X[0]], 
                   [X[1], -X[0], 0]])
    elif all(shape(X) == (1,2)):
        Y = dot(array([[0,1], [-1,0]]), X.T)
    elif all(shape(X) == (2,1)):
        Y = dot(X.T, array([[0,1], [-1,0]]))
    elif all(shape(X) == (3,3)):
        Y = array([X[1,2], X[2,0], X[0,1]])
    elif all(shape(X) == (4,4)):
        Y = array([[0, X[2,3], X[3,1], X[1,2]],
                   [X[3,2], 0, X[0,3], X[2,1]],  
                   [X[1,3], X[3,0], 0, X[0,1]],
                   [X[2,1], X[0,2], X[1,0], 0]])
    else:
        raise ValueError('Wrong matrix size')
        
    return Y

#vgg_X_from_xP_lin  Estimation of 3D point from image matches and camera matrices, linear.
#   X = vgg_X_from_xP_lin(x,P,imsize) computes projective 3D point X (column 4-vector)
#   from its projections in K images x (2-by-K matrix) and camera matrices P (K-cell
#   of 3-by-4 matrices). Image sizes imsize (2-by-K matrix) are needed for preconditioning.
#   By minimizing algebraic distance.
#
#   See also vgg_X_from_xP_nonlin.

# werner@robots.ox.ac.uk, 2003


def vgg_X_fromxP_lin(u, P, imsize):
    
    K = len(P)
    for k in range(K):
        H = array([[2.0 / imsize[0, k], 0, -1],
                   [0, 2.0 / imsize[1, k], -1],
                   [0, 0, 1]])
        P[k] = dot(H, P[k])
        u[:,k] = (dot(H[0:2, 0:2], u[:,k]).T + H[0:2,2]).T
    A = zeros(4)
    for k in range(K):
        tmp = dot(vgg_contreps(vstack((u[:,k], 1))), P[k])
        A = vstack((A,tmp))
    A = delete(A,0,0)
    _,_, X = svd(A)
    X = X[-1, :]
    
    # Get orientation right
    tmp = zeros(4)
    for i in range(K):
        tmp = vstack((tmp, P[i][2,:]))
    tmp = delete(tmp,0,0)    
    s = dot(tmp, X)
    if any(s < 0):
        X = -X

    return X

#P = vgg_P_from_F(F)  Compute cameras from fundamental matrix.
#   F has size (3,3), P has size (3,4).
#
#   If x2'*F*x1 = 0 for any pair of image points x1 and x2,
#   then the camera matrices of the image pair are 
#   P1 = eye(3,4) and P2 = vgg_P_from_F(F), up to a scene homography.
 
# Tomas Werner, Oct 2001
 
def vgg_P_from_F(F):

    u,s,v = np.linalg.svd(F)
    e = u[:,2]
    P = np.hstack((-np.dot(vgg_contreps(e),F), np.expand_dims(e,1)));
 
    return P