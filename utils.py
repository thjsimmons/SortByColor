
"""
Notes:
Replacements:
    domColor -> k_means
    scaleToSizeImage -> to_size
    L2difference -> diffHST
    CCV_difference -> diffCCV
    get_albums -> getImages(path)
    macbeth_array -> macbethArray
"""
import cv2
import numpy as np

def getImages(path):
    # Return list of album names from AlbumList.txt
    f = open(path)
    l = []
    while True:
        line = f.readline()
        if not line:
            break
        l.append(line.replace(" ", "-").replace("\n", ""))
    f.close()
    return l

def to_size(x, y, img): # arguments reversed
    return cv2.resize(img, (x, y), interpolation = cv2.INTER_CUBIC)

def scale_by(factor, img): # arguments reversed
    return cv2.resize(img, None, fx = factor, fy = factor, interpolation = cv2.INTER_CUBIC)

def rgb2lab(rgb):   # entry -> vector
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0][0]
    
def rgb2bgr(rgb):   # entry -> vector
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2BGR)[0][0]

def bgr2rgb(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2RGB)[0][0]

def diffHST(H1, H2): # sum of abs differences, use minDist again between each album
    return sum([abs(H1[i][1] - H2[i][1]) for i in range(0, len(H1))])

def diffCCV(V1, V2):
    return sum([abs(V1[i][0] - V2[i][0]) + abs(V1[i][1] - V2[i][1]) for i in range(0, len(V1))])

def dist3(A, B):
    return ((A[0]-B[0])**2.0 + (A[1]-B[1])**2.0 + (A[2]-B[2])**2.0)**0.5

def dist3LAB(A, B): # assumes RGB inputs, allows passing function into min search function
    return dist3(rgb2lab(A), rgb2lab(B))

"""
def average(img): # more concise version 
    avg = sum([sum([np.array(img[i][j]) for j in range(0, img.shape[1])]) for i in range(0, img.shape[0])])/(img.shape[0]*img.shape[1])
    return avg
"""


def average(img):

    sum_rgb = np.array([0,0,0])

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            sum_rgb += np.array(img[i][j])
    
    avg_rgb = sum_rgb / (img.shape[0]*img.shape[1])
    #avg_rgb = sum([img[i][j] for i in range(0,img.shape[0] )))/(img.shape[0]*img.shape[1])
    #aimg = img.copy()
    #aimg[0:aimg.shape[0], 0:aimg.shape[1]] = avg_rgb
    
    return avg_rgb
    
def newImg(w,h,color):
    img = np.zeros((w, h, 3), np.uint8)
    img[:] = color
    return img