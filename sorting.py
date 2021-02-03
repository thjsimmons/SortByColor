#======================================================================
# Github: https://github.com/thjsimmons
#======================================================================

import cv2
import numpy as np
import copy
from utils import *

    
def k_means(k, img):  # Source: https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv

    # Uses cv2 k-means clustering to find k most prominant colors in img (palette),
    # labels each pixel with only those k colors. 
    # function returns palette, labels, count of labels of each colors
    
    pixels = np.float32(img.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return palette, labels, counts

def findMinimum(p_entry, q_entries, func): # Find index of element in entries that minimizes value of 'func':
    # entry has form ["name", value], entries has form [["name", value], ["name", value], ...]
    # func is one of dist3, dist3LAB, distHST, distccv
    val = -1
    p = p_entry[1]

    for i in range(len(q_entries)): 
        q = q_entries[i][1]
        f = func(p, q)
        if val == -1 or f < val:
            mindex, val = i, f

    return mindex 

def loopSort(entries, func): # Sort entries in order that minimizes 'func' between adjacent elements:
    # loop, entries have form [["name", value], ["name", value], ...]
    # func is one of dist3, dist3LAB, distHST, distccv.
    loop = []
    for i in range(len(entries)):
        if i == 0:
            loop.append(entries.pop(0))
        else:
            loop.append(entries.pop(findMinimum(loop[i-1], entries, func)))
    
    return loop


def reSort(loop, func, total): 
    # Passes back through sorted loop (fixed # of passes) and tries to minimize
    # either total distance of the image loop or the local distances if total = False. 
    # The end element is popped and tested at each position in the loop for a distance improvement

    nloop =  copy.deepcopy(loop) # convert loop entry names to numbers
    for i in range(len(nloop)):
        nloop[i][0] = i
   
    loop_length = len(nloop)

    # Pre-compute distances between all possible loop combinations and store in distance matrix:
    distance_matrix = np.zeros([loop_length, loop_length])

    for i in range(loop_length):
        for j in range(loop_length):
            if i != j:
                distance_matrix[i][j] = func(loop[i][1], loop[j][1])   # symmetric about diagonal which is all 0s, order of i, j does not matter
    
    max_pass_count = 50 # max number of passes or re-sorts
    pass_count = 0

    # while effectively shuffles the nloop for distance improvements:
    while pass_count < max_pass_count:
  
        moving_loop_entry = nloop.pop(-1)   # end element popped off
        moving_index = moving_loop_entry[0] # index of end element 

        mindex = -1 # becomes new index for moving entry 
        val = -1    # total distance value given moving entry at mindex
       
        for i in range(loop_length-1): # iterate over loop 
            avg_of_distances =0
            total_distance = 0

            # If total = False then avg_of_distances (average of local distances between adjacent loop entries) is minimized:
            if i == 0 or i == loop_length-1:             # insert between end and starting element
                behind_index = nloop[loop_length-2][0]   # end element
                ahead_index = nloop[0][0]                # starting element
                avg_of_distances = (distance_matrix[behind_index, moving_index] + distance_matrix[ahead_index, moving_index])/2.0
            else:                                        # insert at i, between (i-1) and i which is pushed forward
                behind_index = nloop[i-1][0]             
                ahead_index = nloop[i][0]   
                avg_of_distances = (distance_matrix[behind_index, moving_index] + distance_matrix[ahead_index, moving_index])/2.0

            nloop_test = copy.deepcopy(nloop) # create copy and insert the moving entry to test

            if i == loop_length-1: # inserted between end and 0
                nloop_test.append(moving_loop_entry)
            else:
                nloop_test.insert(i, moving_loop_entry)
      
            N = distance_matrix.shape[0]
            # Get total distance of loop with inserted moving entry using the pre-computed distance matrix:
            for k in range(N):  
                if k == 0:    # distance between end and starting element
                    d = distance_matrix[nloop_test[-1][0], nloop_test[0][0]]
                else:
                    d = distance_matrix[nloop_test[k-1][0], nloop_test[k][0]]

                total_distance += d
          
            if total: 
                if mindex == -1 or total_distance < val:
                    val = total_distance
                    mindex = i 
            else:
                if mindex == -1 or avg_of_distances < val:
                    val = avg_of_distances
                    mindex = i
        
        # Got index of position that minimizes distance -> insert at index

        if mindex == loop_length-1: # inserted between end and 0
            nloop.append(moving_loop_entry)
        else:
            nloop.insert(mindex, moving_loop_entry)

        print "Resort pass =, ", pass_count, "New Index for image = ", mindex
        pass_count += 1

    # Convert re-sorted nloop entries back to named entries:
    new_loop = nloop

    for i in range(len(new_loop)):
        original_index = new_loop[i][0]
        new_loop[i][0] = loop[original_index][0]
    
    return new_loop 

def RGB2MAC(img): # Converts RGB image to array labeled with binned Macbeth color of each pixel:
    # img has form [[[r,g,b], ...], [[r,g,b], ...], ...], mac has form [[C, ...], [C, ...], ...]
    M = np.zeros([img.shape[0], img.shape[1]]).astype('uint32').tolist() 

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            M[i][j] = int(findMinimum(['', img[i][j]], MACBETH_LIST, dist3lab)) # Find nearest Macbeth color to pixel in MACBETH_LIST

    return M 

def img2HST(name, img): # Convert image to color histogram binned by the 24 Macbeth colors: 
    hist = [0 for i in range(0, len(MACBETH_LIST))] 
    mac = RGB2MAC(img)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            hist[mac[i][j]] += 1

    for value in hist: # Normalize histogram to image size so that different image sizes can be compared on the same scale:
        value = value * int(10.0e8/(img.shape[0] * img.shape[1]))

    return [name, hist]

# Following 2 functions implement algorithm for computing color coherence vectors discussed in refs/ccvsPaper.pdf:

def blobExtract(mac): # Convert macbeth-labeled array to array of numbered blobs (or "connected-components"): 

    # e.g. if mac =  [0  2  2  2  19] then blob =  [1  2  2  2  3 ]
    #                [0  0  2  2  2 ]              [1  1  2  2  2 ]
    #                [16 16 24 24 24]              [4  4  5  5  5 ]
    #                [16 16 24 24 23]              [4  4  5  5  6 ]    
    ##               [16 16 23 23 23]              [4  4  6  6  6 ] 

    blob = np.zeros([mac.shape[0], mac.shape[1]]).astype('uint32').tolist() 
    n_blobs = 0
    
    for index in range(0, len(MACBETH_LIST)):
        count, labels = cv2.connectedComponents(np.where(mac == index, 1, 0).astype('uint8')) # Label regions for the index-th color:
        
        labels[labels > 0] += n_blobs # Raise labels by count in superimposed blob/graph:         
        blob += labels # Superimpose new regions on blob/graph:
        
        if count > 1: 
            n_blobs += (count-1) 
    
    return n_blobs, blob

def img2CCV(name, img): # Convert image to color coherence vector (CCV)
    # CCV is a pair of histograms, one for coherent pixels, one for incoherent pixels.
    # Coherent pixels are part of blob with size >= threshold, incoherent pixels are part of a blob with size < threshold
    
    threshold = round(0.01 * img.shape[0] * img.shape[1]) # Threshold = 1% of image area

    blob = np.zeros([img.shape[0], img.shape[1]]).tolist() 
    mac = RGB2MAC(img) 
    n_blobs, blob = blobExtract(np.array(mac))   

    table = [[0,0] for i in range(0, n_blobs)] # Table stores blob sizes and their colors, form is [[color bin #, size], [color bin #, size], ... ]
    for i in range(0, blob.shape[0]):
        for j in range(0, blob.shape[1]):
            table[blob[i][j]-1] = [mac[i][j], table[blob[i][j]-1][1] + 1] 

    # Get CCV of form [[coherent, incoherent], [coherent, incoherent], ... ] indexed by color bin #:

    ccv = [[0,0] for i in range(0, len(MACBETH_LIST))] 

    for entry in table:
        color_index = entry[0]
        size = entry[1]

        if size >= threshold: 
            ccv[color_index][0] += size # increment coherent pixels of color_index
        else:
            ccv[color_index][1] += size # increment incoherent pixel of color_index
    
    return [name, ccv] 

