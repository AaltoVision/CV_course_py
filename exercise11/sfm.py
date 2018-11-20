# -*- coding: utf-8 -*-
import numpy as np
import ransac

## Copyright (c) 2012, Jan Erik Solem
## All rights reserved.

## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met: 

## 1. Redistributions of source code must retain the above copyright notice, this
##    list of conditions and the following disclaimer. 
## 2. Redistributions in binary form must reproduce the above copyright notice,
##    this list of conditions and the following disclaimer in the documentation
##    and/or other materials provided with the distribution. 

## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
## ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
## WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
## ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
## (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
## LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
## ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
## SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

class RansacModel(object):
    """ Class for fundmental matrix fit with ransac.py from
    http://www.scipy.org/Cookbook/RANSAC """
    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        """ Estimate fundamental matrix using eight 
        selected correspondences """ 
        # transpose and split data into the two point sets
        data = data.T
        x1 = data[:3,:8]
        x2 = data[3:,:8]
    
        # estimate fundamental matrix and return
        F = compute_fundamental_normalized(x1,x2)
        return F

    def get_error(self,data,F):
        """ Compute x^T F x for all correspondences,
        return error for each transformed point. """
        # transpose and split data into the two point
        data = data.T
        x1 = data[:3]
        x2 = data[3:]
        
        # Sampson distance as error measure
        Fx1 = np.dot(F,x1)
        Fx2 = np.dot(F,x2)
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        err = ( np.diag(np.dot(x1.T,np.dot(F,x2))) )**2 / denom
        
        # return error per point
        return err
    
def compute_fundamental_normalized(x1,x2):
    """ Computes the fundamental matrix from corresponding points
    (x1,x2 3*n arrays) using the normalized 8 point algorithm. """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise Value
        Error("Number of points don’t match.")
        
    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = np.dot(T1,x1)
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = np.dot(T2,x2)
    
    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)
    
    # reverse normalization
    F = np.dot(T1.T,np.dot(F,T2))
    return F/F[2,2]

def compute_fundamental(x1,x2):
    """ Computes the fundamental matrix from corresponding points
    (x1,x2 3*n arrays) using the normalized 8 point algorithm.
    each row is constructed as
    [x’*x, x’*y, x’, y’*x, y’*y, y’, x, y, 1] """
    
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don’t match.")
        
    # build matrix for equations
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
        x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
        x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
        
    # compute linear least square solution
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
    
    # constrain F
    # make rank 2 by zeroing out last singular value
    
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U,np.dot(np.diag(S),V))
    return F

def F_from_ransac(x1,x2,model,maxiter=5000,match_theshold=1e-6):
    import ransac
    data = np.vstack((x1,x2))
    # compute F and return with inlier index
    F,ransac_data = ransac.ransac(data.T,model,8,maxiter,match_theshold,20,return_all=True)
    return F, ransac_data['inliers']

def skew(a):
    """ Skew matrix A such that a x v = Av for any v. """
    return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])

def compute_P_from_essential(E):
    """ Computes the second camera matrix (assuming P1 = [I 0])
    from an essential matrix. Output is a list of four
    possible camera matrices. See section 9.6.2 from H&Z book (p.258)."""
    # make sure E is rank 2
    U,S,V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U,V))<0:
        V = -V
    E = np.dot(U,np.dot(np.diag([1,1,0]),V))
    # create matrices (Hartley p 258)
    Z = skew([0,0,-1])
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    # return all four solutions
    P2 = [np.vstack((np.dot(U,np.dot(W,V)).T,U[:,2])).T,
    np.vstack((np.dot(U,np.dot(W,V)).T,-U[:,2])).T,
    np.vstack((np.dot(U,np.dot(W.T,V)).T,U[:,2])).T,
    np.vstack((np.dot(U,np.dot(W.T,V)).T,-U[:,2])).T]
    return P2

def triangulate_point(x1,x2,P1,P2):
    """ Point pair triangulation from
    least squares solution. """
    M = np.zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2
    U,S,V = np.linalg.svd(M)
    X = V[-1,:4]
    return X / X[3]

def triangulate(x1,x2,P1,P2):
    """ Two-view triangulation of points in
    x1,x2 (3*n homog. coordinates). """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don’t match.")
    X = [ triangulate_point(x1[:,i],x2[:,i],P1,P2) for i in range(n)]
    return np.array(X).T

class Camera(object):
    def __init__(self,P):
        self.P = P
        
    def project(self,X):
        x = np.dot(self.P,X)
        for i in range(3):
            x[i] /= x[2]
        return x