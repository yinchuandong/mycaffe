import os
import sys
import shutil
from PCV.tools import imtools, pca
from PIL import Image, ImageDraw
from PCV.localdescriptors import sift
from pylab import *
import glob
from scipy.cluster.vq import *


def loadData(path):
    """
    load data and tranform rgb to histogram, it is 3-d.
    Each cube represents num, not rgb any`
    """
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    imlist = imlist[0:100]
    
    # extract feature vector (8 bins per color channel)
    features = zeros((len(imlist), 8000))
    for i, f in enumerate(imlist):
        im = array(Image.open(f))
        # multi-dimensional histogram
        h, edges = histogramdd(im.reshape(-1, 3), 20, normed=False, range=[(0, 255), (0, 255), (0, 255)])
        features[i] = h.flatten()
    return imlist, features


if __name__ == '__main__':
    path = '../data_exterior/exterior/'
    result_path = '../cluster_result/'
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    imlist = imlist[0:100]
    imnbr = len(imlist)
    
    # Load images, run PCA.
    immatrix = array([array(Image.open(im)).flatten() for im in imlist], 'f')
    V, S, immean = pca.pca(immatrix)
    print 'the shape of V', shape(V)

    # Project on 40% PCs.
    projected = array([dot(V[:int(imnbr) * 0.4], immatrix[i] - immean) for i in range(imnbr)])  

    n = len(projected)
    # compute distance matrix
    S = array([[ sqrt(sum((projected[i]-projected[j])**2))
    for i in range(n) ] for j in range(n)], 'f')
    # create Laplacian matrix
    rowsum = sum(S,axis=0)
    D = diag(1 / sqrt(rowsum))
    I = identity(n)
    L = I - dot(D,dot(S,D))
    # compute eigenvectors of L
    U,sigma,V = linalg.svd(L)

    k = 5
    features = array(V[:k]).T
    features = whiten(features)
    # centroids,distortion = kmeans(newFeatures, k)
    centroids,distortion = kmeans(features, k)
    code,distance = vq(features, centroids)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
        os.mkdir(result_path)

    for c in range(k):
        ids = where(code == c)[0]
        cpath = result_path + str(c) + '_' + str(len(ids)) + '/'
        os.mkdir(cpath)
        print 'cluster %d has %d images' % (c, len(ids))
        for i in range(len(ids)):
            srcPath = imlist[ids[i]]
            imName = srcPath.split(path)[1]
            shutil.copy(srcPath, cpath + imName)

    # for c in range(k):
    #     ind = where(code==c)[0]
    #     cpath = '../cluster_result/' + str(c) + '/'
    #     os.mkdir(cpath)
    #     figure()
    #     gray()

    #     for i in range(minimum(len(ind), 20)):
    #         im = Image.open(imlist[ind[i]])
    #         subplot(5,4,i+1)
    #         imshow(array(im))
    #         axis('equal')
    #         axis('off')
    # show()