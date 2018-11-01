import numpy as np
from pyflann import *
import matplotlib.pyplot as plt
from skimage.io import imread

## Some function definitions that we need in this exercise
# pyflann - https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf

def findNeighbours(descrs1, descrs2, numNeighbors=1):
    flann = FLANN()
    ind, dist = flann.nn(descrs2, descrs1, numNeighbors, algorithm="kdtree",
        trees=8, checks=100)
    return ind, dist

def kdtreequery(vocab, descrs, maxNumChecks=64):
    flann = FLANN()
    ind, dist = flann.nn(vocab, descrs, 1, algorithm="kdtree",
        trees=8, checks=maxNumChecks)
    return ind

def matchWords(words1, words2, vocab_size=100000):
    a = words1.squeeze().copy()
    b = words2.squeeze().copy()

    # Make a histogram out of the words (100000 different possibilities)
    ha,_ = np.histogram(a, bins=np.arange(vocab_size+1))
    hb,_ = np.histogram(b, bins=np.arange(vocab_size+1))
    
    # Remove such words from both a and b that contain more than 5
    ha[ha>5] = -1
    hb[hb>5] = -1
    
    # Destroy all such words that occur more than 5 times
    sela = np.take(ha,a)
    selb = np.take(hb,b)
    a[sela==-1] = -1
    b[selb==-1] = -1
    
    matches = []
    for i in range(a.shape[0]):
        if a[i] == -1:
            continue
        else:
            word = a[i]
            indices = np.where(b == word)
            sel = indices[0]
            if sel.size > 0:
                b[indices] = -1
                match = [i,sel[0]]
                matches.append(match)
                
    return np.array(matches)

def plotMatches(ax, im1, im2, kps1, kps2, pairs):
    ax.imshow(np.hstack((im1, im2)), cmap="gray")
    t = np.arange(0, 2*np.pi, 0.1)

    # Display matches
    for k in range(pairs.shape[0]):
        pid1 = pairs[k, 0]
        pid2 = pairs[k, 1]
    
        loc1 = kps1[int(pid1),0:2]
        r1 = kps1[int(pid1), 2] * 3
        loc2 = kps2[int(pid2),0:2]
        r2 = kps2[int(pid2), 2] * 3

        ax.plot(loc1[0]+r1*np.cos(t), loc1[1]+r1*np.sin(t), 'm-', linewidth=3)
        ax.plot(loc2[0]+r2*np.cos(t)+im1.shape[1], loc2[1]+r2*np.sin(t), 'm-', linewidth=3)
        #plt.plot(loc2[0]+r2*np.cos(t), loc2[1]+r2*np.sin(t), 'm-', linewidth=3)
        ax.plot([loc1[0], loc2[0]+im1.shape[1]], [loc1[1], loc2[1]], 'c-')

def toAffinity(f):
    #print(f)
    T = np.expand_dims(f[0:2], axis=1)
    a0 = np.expand_dims(f[2:4], axis=1)
    a1 = np.expand_dims(f[4:6], axis=1)
    A0 = np.hstack([a0,a1,T])
    A = np.vstack((A0, np.array([[0,0,1]])))
    return A

def centering(x):
    I = np.identity(2)
    m = np.mean(x, axis=1)
    m = m[0:2]
    m = m[:,np.newaxis]
    v = np.array([[0, 0, 1]])
    #print(I.shape, m.shape)
    T_ = np.hstack((I, m))
    T = np.vstack((T_,v))
    
    x = np.dot(T,x)
    std1 = np.std(x[0,:])
    std2 = np.std(x[1,:])
    
    S = np.array([[1./std1,0,0],[0,1./std2,0],[0,0,1]])
    C = np.dot(S,T)
    
    return C

def geometricVerification(f1, f2, matches, numRefIterations=3):

    x1 = f1.copy()
    x2 = f2.copy()
    
    tolerance1 = 20
    tolerance2 = 15
    tolerance3 = 8
    minInliers = 6
    numRefinementIterations = numRefIterations
    
    numMatches = matches.shape[0]
    #print("Number of matches: {}".format(numMatches))
    inliers = {}
    numInliers = {}
    H = {}
    
    x1 = x1[matches[:,0], 0:2] - 1.
    x2 = x2[matches[:,1], 0:2] - 1.
    
    ones1 = np.ones((x1.shape[0], 1))
    ones2 = np.ones((x2.shape[0], 1))
    
    x1hom = np.hstack((x1, ones1)).T
    x2hom = np.hstack((x2, ones2)).T
    
    for m in range(numMatches):
        for t in range(numRefinementIterations):
            if t == 0:
                A1 = toAffinity(f1[matches[m,0],:])
                #print(A1)
                A2 = toAffinity(f2[matches[m,1],:])
                #print(A2)
                H21 = np.dot(A2, np.linalg.inv(A1))
                x1p = np.dot(H21[0:2,:], x1hom)
                tol = tolerance1
            #else:
            elif t <= 3:
                # affinity
                H21 = np.dot(x2[inliers[m],:].T, np.linalg.pinv(x1hom[:, inliers[m]]))
                #print(H21)
                x1p = np.dot(H21[0:2,:], x1hom)
                H21 = np.vstack((H21, np.array([[0,0,1]])))
                tol = tolerance2
            #####JUST#COMMENT#AWAY#####
            else:
                x1in = x1hom[:,inliers[m]]
                x2in = x2hom[:,inliers[m]]
                
                S1 = centering(x1in)
                S2 = centering(x2in)
                
                x1c = np.dot(S1, x1in)
                x2c = np.dot(S2, x2in)
                
                x2c_0 = -x2c[0,:]
                x2c_0 = x2c_0[np.newaxis,:]
                #print(x2c_0.shape)
                x2c_1 = -x2c[1,:]
                x2c_1 = x2c_1[np.newaxis,:]
                #print(x2c_1.shape)
                #print(x1c.shape)
                M1 = np.hstack((x1c, np.zeros(x1c.shape)))
                M2 = np.hstack((np.zeros(x1c.shape), x1c))
                M3 = np.hstack((x1c*x2c_0, x1c*x2c_1))
                
                M = np.vstack((M1,M2,M3))
                H21,_,_ = np.linalg.svd(M)
                H21 = H21[:,-1].reshape((3,3))
                H21 = np.dot(np.dot(np.linalg.inv(S2),H21),S1)
                H21 = H21 / H21[-1,-1]
                x1phom = np.dot(H21,x1hom)
                x1p_0 = x1phom[0,:] / x1phom[2,:]
                x1p_1 = x1phom[1,:] / x1phom[2,:]
                x1p = np.vstack((x1p_0[np.newaxis,:],x1p_1[np.newaxis,:]))
                tol = tolerance3
            ###########################
            dist2 = np.sum((x2 - x1p.T) ** 2, 1)
            ind = np.where(dist2 < tol**2)
            inliers[m] = ind[0]
            numInliers[m] = len(ind[0])
            H[m] = H21
            if numInliers[m] < minInliers:
                break
            if numInliers[m] > 0.7 * numMatches: 
                break
    best = max(numInliers.items(), key=lambda x: x[1])[0]
    #print(best)
    #print(H[best])
    inliers = inliers[best]
    H_ = np.linalg.inv(H[best])

    return inliers, H_ 

from scipy import sparse

def getHistogramFromDescriptor(vocab, idf, descrs):
    words = kdtreequery(vocab, descrs, maxNumChecks=1024)

    #Make a histogram out of the words (100000 different possibilities)
    ha,_ = np.histogram(words, bins=np.arange(100001))
    ha = np.expand_dims(ha, axis=1)
    sha = sparse.csr_matrix(ha)
    sha_idf = sha.multiply(idf)
    #ha = sha_idf.todense()
    #print(np.linalg.norm(ha))
    ha = ha / np.linalg.norm(ha)
    
    return ha

def getImage(imPath, imName, ii):
    import sys  
    if sys.version_info[0] > 2:
        imPath = imPath.decode('utf-8')
        imName = imName.decode('utf-8')
    #imPath = "{}/{}".format(imPath, imName)
    imPath = imPath + '/' + imName
    im = imread(imPath)
    return im
    
def plotRetrievedImages(imDir, imNames, scores, num=25):
    # sort scores in descending order
    sorted_scores = np.sort(scores, axis=0)
    id_sorted_scores = np.argsort(scores, axis=0)
    sorted_scores = sorted_scores[::-1]
    id_sorted_scores = id_sorted_scores[::-1]
    
    fig, axes = plt.subplots(nrows=(num//5), ncols=5, figsize=(32,16))
    ax = axes.ravel()
    
    for rank in range(25):
        ii = id_sorted_scores[rank]
        im0name = imNames[ii]
        im0 = getImage(imDir, im0name, ii)
        ax[rank].imshow(im0, cmap='gray')
        ax[rank].axis('off')
        ax[rank].set_title("Score: {}".format(sorted_scores[rank]))
        
def plotFrame(ax, im, kps):
    ax.imshow(im)
    ax.axis("off")
    K = kps.shape[0]
    nv = 40
    thr = np.linspace(0,2*np.pi, nv)
    
    Xp = np.vstack((np.cos(thr), np.sin(thr)))

    # plot features
    for k in range(K):
        xc = kps[k,0]
        yc = kps[k,1]
        A = kps[k,2:].reshape((2,2)).T
        X = np.dot(A,Xp)
        X[0,:] = X[0,:] + xc
        X[1,:] = X[1,:] + yc

        ax.plot(X[0,:], X[1,:], color='cyan') 

        xf = xc + np.array([0, A[0,1]])
        yf = yc + np.array([0, A[1,1]])

        ax.plot(xf, yf, color='cyan')
def plotImBoth(ax, im1, im2):
    if im1.shape[0] < im2.shape[1]:
        #pad left image
        im1 = np.vstack((im1,np.zeros((im2.shape[0]-im1.shape[0],im1.shape[1],3))))
    else:
        #pad right image
        im2 = np.vstack((im2,np.zeros((im1.shape[0]-im2.shape[0],im2.shape[1],3))))
    
    ax.imshow(np.hstack((im1, im2)))

def plotFrameBoth(ax, im1, im2, kps1, kps2, matches_geom=[], plotMatches=False):
    if im1.shape[0] < im2.shape[1]:
        #pad left image
        im1 = np.vstack((im1,np.zeros((im2.shape[0]-im1.shape[0],im1.shape[1],3))))
    else:
        #pad right image
        im2 = np.vstack((im2,np.zeros((im1.shape[0]-im2.shape[0],im2.shape[1],3))))
    
    ax.imshow(np.hstack((im1, im2)))
    
    o = im1.shape[1]
    
    if matches_geom == []:
        kps1_ = kps1
        kps2_ = kps2
        K1 = kps1_.shape[0]
        K2 = kps2_.shape[0]
    else:
        kps1_ = kps1[matches_geom[:,0],:]
        kps2_ = kps2[matches_geom[:,1],:]
        K1 = kps1_.shape[0]
        K2 = K1
    
    kps2_[:,0] = kps2_[:,0] + o

    nv = 40
    thr = np.linspace(0,2*np.pi, nv)
    
    Xp = np.vstack((np.cos(thr), np.sin(thr)))

    # left image
    for k in range(K1):
        xc = kps1_[k,0]
        yc = kps1_[k,1]
        A = kps1_[k,2:].reshape((2,2)).T
        X = np.dot(A,Xp)
        X[0,:] = X[0,:] + xc
        X[1,:] = X[1,:] + yc

        ax.plot(X[0,:], X[1,:], color='cyan') 

        xf = xc + np.array([0, A[0,1]])
        yf = yc + np.array([0, A[1,1]])

        ax.plot(xf, yf, color='cyan')
        
    # right image
    for k in range(K2):
        xc = kps2_[k,0]
        yc = kps2_[k,1]
        A = kps2_[k,2:].reshape((2,2)).T
        X = np.dot(A,Xp)
        X[0,:] = X[0,:] + xc
        X[1,:] = X[1,:] + yc

        ax.plot(X[0,:], X[1,:], color='cyan') 

        xf = xc + np.array([0, A[0,1]])
        yf = yc + np.array([0, A[1,1]])

        ax.plot(xf, yf, color='cyan')
    
    if plotMatches:
        # lines to demonstrate matching pairs
        colors = ['C{}'.format(i) for i in range(10)]

        for k in range(K1):
            ic = k % 10
    
            loc1 = kps1_[k,0:2]
            loc2 = kps2_[k,0:2]

            ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], color='{}'.format(colors[ic]), linestyle='-')