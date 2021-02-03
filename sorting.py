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
    '''
    can pre-compute distances between all histograms into
    an 89x89 matrix then just retrieve the result

    '''

    # First convert loop entries into numbered entries 'nloop'
    # at the end the names will be retrieved into a new loop
    
    nloop =  copy.deepcopy(loop) # numbered loop
    for i in range(len(nloop)):
        nloop[i][0] = i
    #print "loop = ", loop
    loop_length = len(nloop)
    distance_matrix = np.zeros([loop_length, loop_length])

    for i in range(loop_length):
        for j in range(loop_length):
            if i != j:
                distance_matrix[i][j] = func(loop[i][1], loop[j][1])   # symmetric about diagonal which is 0, order of i, j does not matter
    
    #print("distance matrix = ", distance_matrix)
    '''
    Try iterating backwards from end of loop and insert the end element
    into the new place minimizes it's function value w.r.t adjacent elements
    and w.r.t all other positions

    '''
    max_pass_count = 50
    pass_count = 0

    while pass_count < max_pass_count:
        #print("\n PASS # ", pass_count, "\n")
        #print("nloop = ", nloop)
        # nloop gets rearranged but the 1st value of each entry keeps track of the original index w.r.t the distance_matrix
        # nloop length never changes, stays the same only for each pass
        moving_loop_entry = nloop.pop(-1) # [loop index, CCV]
        moving_index = moving_loop_entry[0]
        mindex = -1
        val = -1
       
        # This can be its own function like findMinimum
        # Iterate forward and find a better place for last element 
        for i in range(loop_length-1): # has 1 less elements after pop, 
            avg_of_distances =0
            total_distance = 0
            # compare moving_value to static_value i
         
            # Get distance between moving entry and current entry from distance matrix
            # minimize average of distances
            # to insert at location i would be push that element forward to i+1
            # still have to get index 
            # i is current position of entry in nloop, nloop[i][0] is original index

            # Have to fix it back with the beginning of the loop!

            # iterate from 0 to len-1 ... sum of 
            # iterate from 0 to len-1 ... sum of 
            if i == 0 or i == loop_length-1: # beginning, never actually equals loop_length - 1
                behind_index = nloop[loop_length-2][0] # behind
                ahead_index = nloop[0][0]   # ahead/at - INSERTING AT PUSHES THIS ELEMENT FORWARD
                avg_of_distances = (distance_matrix[behind_index, moving_index] + distance_matrix[ahead_index, moving_index])/2.0
   

            else: 
                behind_index = nloop[i-1][0] # behind
                ahead_index = nloop[i][0]   # ahead/at - INSERTING AT PUSHES THIS ELEMENT FORWARD
                avg_of_distances = (distance_matrix[behind_index, moving_index] + distance_matrix[ahead_index, moving_index])/2.0

            # Calc total distance 

            '''
            so its like you're assuming that moving_entry is inserted at nloop[i]
            and compensating for that in the sum:

            when i == 0, moving_entry is inserted at the beginning at nloop[0]

            '''

            nloop_copy = copy.deepcopy(nloop)

            if i == loop_length-1: # inserted between end and 0
                nloop_copy.append(moving_loop_entry)
            else:
                nloop_copy.insert(i, moving_loop_entry)
            #total_distance = distance_matrix[behind_index, moving_index] + \
            #                 distance_matrix[ahead_index, moving_index] + \
            
            # if i = 0 then behind_index = next to last column in matrix, ahead_index = 0th column
            # 
            N = distance_matrix.shape[0]

            # at N-1 its index N-1 minus index N-2 
            for k in range(N):  # index never reaches end which 
                if k == 0:
                    d = distance_matrix[nloop_copy[-1][0], nloop_copy[0][0]]
                else:
                    d = distance_matrix[nloop_copy[k-1][0], nloop_copy[k][0]]

                total_distance += d
            #total_distance = avg_of_distances
            if total:
                if mindex == -1 or total_distance < val:
                    val = total_distance
                    mindex = i # moving entry goes between i, i+1 therefore inserted at i+1
            else:
                if mindex == -1 or avg_of_distances < val:
                    val = avg_of_distances
                    mindex = i # moving entry goes between i, i+1 therefore inserted at i+1
        
        # Got index of minimum position 

        if mindex == loop_length-1: # inserted between end and 0
            nloop.append(moving_loop_entry)
        else:
            nloop.insert(mindex, moving_loop_entry)

        print "Resort pass =, ", pass_count, "New Index for image = ", mindex
        pass_count += 1

    # once have nloop through 100 passes, convert first element back to name:

    new_loop = nloop

    for i in range(len(new_loop)):
        original_index = new_loop[i][0]
        #print("loop[original_index][0]= ", loop[original_index][0])
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

