import cv2
import numpy as np
from macbeth import MACBETH_LIST
import time

import utils as U
import sorting as S
import hist as H

"""
Notes:
    Changes:
        macbethImg-> to_macbeth, only returns MACBETH_IMG

"""
def blobExtract(M): # MACBETH_IMG is np.array filled with macbeth indexes
    
    BLOB_GRAPH = np.zeros([M.shape[0], M.shape[1]]).tolist()
    BLOB_COUNT = 0
    
    for index in range(0, len(MACBETH_LIST)):
        count, labels = cv2.connectedComponents(np.where(M == index, 1, 0).astype('uint8')) # Labels are numbered region with single color
        labels[labels > 0] += BLOB_COUNT                       
        BLOB_GRAPH += labels
        
        if count > 1:
            BLOB_COUNT += (count-1) # Shifts BLOB_COUNT ("blob" = connected components)

    return BLOB_COUNT, BLOB_GRAPH

def img2CCV(name, img, resize): # convert image to color coherence vector (pair of histograms)
    # still need a function to count pixels in continuous region
    #scaleToSizeImage
    img = U.to_size(resize[0], resize[1], img)
    THRESHOLD = round(0.01 * img.shape[0] * img.shape[1])
    BLOB_GRAPH = [[0 for i in range(0, img.shape[1])] for j in range(0, img.shape[0])]  # could just be np.array([])
 
    # Get BLOB_GRAPH (labels), also known as BLOB_GRAPH (with no zeros)
    MACBETH_IMG = H.RGB2MAC(img)
    print("MACBETH_IMG = ", MACBETH_IMG)
    BLOB_COUNT, BLOB_GRAPH = blobExtract(np.array(MACBETH_IMG))  # BLOB_COUNT is number of different regions

    # BLOB_GRAPH is matrix of regions numbered by connection starting at 1
    # Get macbeth color bin from same index in MACBETH_IMG as BLOB_GRAPH
    # contains [[color_index, size], [color_index, size] ] where ith index is ABC_index (A, B, C, D, E, F) 

    BLOB_TABLE = [[0,0] for i in range(0, BLOB_COUNT)] # index offset by 1 so that (0-4) refers to (1-5)

    for i in range(0, BLOB_GRAPH.shape[0]):
        for j in range(0, BLOB_GRAPH.shape[1]):
            blob_index = int(BLOB_GRAPH[i][j])   # in range of BLOB_COUNT
            color_index = int(MACBETH_IMG[i][j])
            #print("ABC_index, color_index = ", ABC_index, ", ", color_index)
            BLOB_TABLE[blob_index-1] = [color_index, BLOB_TABLE[blob_index-1][1] + 1] #BLOB_TABLE[ABC_index][1] + 1] # increment count of region size 

    # Form CCV: [[coherent, incoherent], [coherent, incoherent] ] by color_index

    CCV = [[0,0] for i in range(0, len(MACBETH_LIST))] 

    for entry in BLOB_TABLE:
        color_index = entry[0]
        size = entry[1]

        if size >= THRESHOLD:
            CCV[color_index][0] += size 
        else:
            CCV[color_index][1] += size 
    
    return [name, CCV] 

