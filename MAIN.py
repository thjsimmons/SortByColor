#======================================================================
# Github: https://github.com/thjsimmons
#======================================================================

import cv2
import numpy as np
import time
import copy
from utils import *
from sorting import *

global DIR 
DIR = "images/"

# MAIN function is SORT_IMAGES
# 'arg' is thing to sort by:
# arg = 0 -> sort by average,                   (2nd worst)
#       1 -> sort by dominant color,            (worst)
#       2 -> sort by nearest color histogram,   (2nd best)
#       3 -> sort by nearest CCV                (best)
# 

def SORT_IMAGES(arg): 

    names = fromFolder(DIR) # Get image names from "images/" folder
    entries = []            # list of form [name, average], [name, dominant], [name, hist], or [name, ccv]

    
    RESIZE = True       # RESIZE = true -> Resize images down (or up) to certain max size, runtime scales linearly with image size
    MAX_SIZE =  4*4

    BLUR = True         # Use box filter (3,3) blur on image before computing avg, hists
    RESORT = True       # Continue sorting after initial sort to improve adjacent image distances
    MOSAIC = True       # Get mosaic.jpg out of loop images
    TOTAL = True        # Resort trys to minimize the total distance of the loop, else it trys to minimize avg of individual distances

    print "Getting entries ..."
    for name in names:                  # for image name in image name list
        start = time.time()             # record start time
        img = cv2.imread(DIR + name)    # get image from directory
        
        if RESIZE:   
            img = scale_by(np.sqrt(MAX_SIZE * 1.0 / (img.shape[0]*img.shape[1])), img)
       
        if BLUR: 
            img = cv2.blur(img,(3,3))
            
        # Get Entry (average, dominant, hist, or CCV) from image:
        if arg == 0:   # avg
            entries.append([name, average(img)]) 
        elif arg == 1: # dom 
            palette, _, counts = k_means(3, img)
            entries.append([name, palette[np.argmax(counts)]])
        elif arg == 2: # hst
            entries.append(img2HST(name, img))
        elif arg == 3: # ccv
            entries.append(img2CCV(name, img))

        end = time.time() # record end time 
        print "Got entry from: {0:40} <-- {1} seconds" \
            .format(name if len(name) < 30 else name[0:30] + "...",  str('{0:.4f}'.format(end-start))) 
           
    loop = []
    FOLDER  = ""
    print("\nBegin Sorting ...")
    
    # Sort entries by corresponding minimizing function:
    if arg == 0:   # avg
        loop = loopSort(entries, dist3)
        FOLDER = "avgLoop/"

        if reSort:
            loop = reSort(loop, dist3, TOTAL)

    elif arg == 1: # dom 
        loop = loopSort(entries, dist3)
        FOLDER  = "domLoop/"

        if reSort:
            loop = reSort(loop, dist3, TOTAL)

    elif arg == 2: # hst
        loop = loopSort(entries, distHST)
        FOLDER  = "hstLoop/"

        if reSort:
            loop = reSort(loop, distHST, TOTAL)

    elif arg == 3: # ccv
        loop = loopSort(entries, distCCV)
        FOLDER  = "ccvLoop/"

        if reSort:
            loop = reSort(loop, distCCV, TOTAL)

    print "\nSorting Complete.\n"
    

    # Write ordered images to corresponding loop folder:
    images = []
    for i in range(0, len(loop)):
        print "Writing File: {0:10}_{1}".format(FOLDER + str(i), name if len(name) < 30 else name[0:30] + "...")
        name = loop[i][0]
        img = cv2.imread(DIR + name)
        images.append(img)
        cv2.imwrite(FOLDER  + str(i) + "_" + name, img) 

    if MOSAIC: # Make mosaic image out of all square images:
        square_images = []

        for image in images:
            if image.shape[0] == img.shape[1]:
                image = to_size(640, 640, image)
                square_images.append(image)

        M = mosaic(square_images)
        cv2.imwrite("refs/mosaic.jpg", scale_by(0.33, M)) 

    print "\nDONE!"
    return 0

arg = 0 # 4 -> sort by CCV (best)
SORT_IMAGES(arg) 

