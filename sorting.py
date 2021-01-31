import cv2
import numpy as np
from macbeth import MACBETH_LIST
import time
import utils as U

"""
Notes:
Replacements:
    domColor -> k_means
    scaleToSizeImage -> to_size
    L2difference -> L2_diff
    CCV_difference -> CCV_diff
    get_albums -> getImages(path)
    minDist -> minRGB
    minLabDist -> minLAB
    minL2diff(p, q_vec) -> minHST and inputs should be entries

    findMinimum replaces minRGB, minLAB, minHST, minCCV

    one function can potentially replace the sorting functions 

    loopSort works hist, CCV, RGB distance sort, what about insertionSort?
    will have to rewrite that part of code to work within existing framework

"""

def k_means(k, img):  # arguments reversed
    # cv2 k-means clustering algorithm:
    # https://en.wikipedia.org/wiki/K-means_clustering
    # https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
    pixels = np.float32(img.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return palette, labels, counts

def findMinimum(p_entry, q_entries, func): # func = U.diffRGB, U.diffLAB, U.diffHST, U.diffCCV
    val = -1
    p = p_entry[1]

    i = 0
    for i in range(0, len(q_entries)): 
        q = q_entries[i][1]
        #print("q_entries[i] ", q_entries[i])
        f = func(p, q)
        if val == -1 or f < val:
            val = f
            mindex = i

    return val, mindex 

# Works for hist, CCV, RGB distance sort, what about insertionSort?
def loopSort(entries, func): # an entry has 3 pieces of information: index, name, value
    loop = []
    #print("entries = ", entries)
    for i in range(0, len(entries)):
        if i == 0:
            loop.append(entries.pop(0))
        else:
            val, mindex = findMinimum(loop[i-1], entries, func)
            loop.append(entries.pop(mindex))
    
    return loop


