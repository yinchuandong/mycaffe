import os
import sys
from PIL import Image
from PCV.clustering import hcluster
from matplotlib.pyplot import *
from numpy import *

# hierarchy clustering
def loadData(path):
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
    imlist, features = loadData(path)

    tree = hcluster.hcluster(features)
    # visualize clusters with some (arbitrary) threshold
    clusters = tree.extract_clusters(0.23 * tree.distance)
    # plot images for cluster_result with more than 3 elements

    # print len(clusters)
    # sys.exit()
    for c in clusters:
        elements = c.get_cluster_elements()
        nbr_elements = len(elements)
        # print nbr_elements
        # continue
        if nbr_elements >= 3:
            figure()
            for p in range(minimum(nbr_elements, 20)):
                subplot(4, 5, p + 1)
                im = array(Image.open(imlist[elements[p]]))
                imshow(im)
                axis('off')
    show()
    # hcluster.draw_dendrogram(tree, imlist, filename='exterior_result.png')