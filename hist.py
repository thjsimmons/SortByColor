import cv2
import numpy as np
from macbeth import MACBETH_LIST
import time
import utils as U
import sorting as S

"""
Notes:
    Changes:
        macbethImg-> to_macbeth, only returns macbeth_array
        img2histogram -> img2HST
        don't need sRGB2macbeth(rgb)

        should implement normalized histograms that don't depend on image size
        you just dump your images in the images folder the loop is spit out 

"""

def RGB2MAC(img):
    
    M = np.zeros([img.shape[0], img.shape[1]]).tolist() # why can't just pass in shape?

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            pixel = img[i][j]
            _, mindex = S.findMinimum(["pixel", pixel], MACBETH_LIST, U.dist3)
            M[i][j] = mindex

    return M #np.array

def img2HST(name, img, resize):
    hist = [[MACBETH_LIST[i][0], 0] for i in range(0, len(MACBETH_LIST))]  # indexes should be macbeth indexes
    img = U.to_size(resize[0], resize[1], img) # resize to something else
    MACBETH_IMG = RGB2MAC(img)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            #print("MACBETH_IMG[i][j] = ", MACBETH_IMG[i][j])
            index = int(MACBETH_IMG[i][j])
            #print("hist[index][1]= ", hist[index][1])
            hist[index][1] += 1
            
    return [name, hist]


