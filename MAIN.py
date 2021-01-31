import cv2
import numpy as np
from macbeth import MACBETH_LIST
import time

import utils as U
import sorting as S
import hist as H
import ccv as C

"""
ccv: 
     img = cv2.imread("images/" + name + ".jpg")
     entries.append(img2CCV(name, img, [50,50])) # [name, CCV]
hist: 
    img = cv2.imread("images/" + name + ".jpg")
    entries.append(img2HST(name, img, [50,50]))

"""

"""
arg 0 -> avg
arg 1 -> dom
arg 2 -> hst
arg 3 -> ccv
"""

def SORT_IMAGES(path, arg): #  hist

    names = U.getImages(path)
    entries = []

    for name in names:
        print "Getting entry from: ", name 
        img = cv2.imread("images/" + name + ".jpg")

        if arg == 0: # avg
            img = U.to_size(50,50, img)
            entries.append([name, U.average(img)]) # averages always zero 
        elif arg == 1: # dom 
            palette, _, counts = S.k_means(3, img)
            entries.append([name, palette[np.argmax(counts)]])
        elif arg == 2: # hst
            entries.append(H.img2HST(name, img, [10,10]))
        elif arg == 3: # ccv
            entries.append(C.img2CCV(name, img, [10,10]))

    loop = []
    folderName = ""

    print("Begin Sorting ...")
    if arg == 0:
        loop = S.loopSort(entries, U.dist3)
        folderName = "avgLoop/"
    elif arg == 1:
        loop = S.loopSort(entries, U.dist3)
        folderName = "domLoop/"
    elif arg == 2:
        loop = S.loopSort(entries, U.diffHST)
        folderName = "hstLoop/"
    elif arg == 3:
        loop = S.loopSort(entries, U.diffCCV)
        folderName = "ccvLoop/"

    print("Sorting Complete.")
    
    for i in range(0, len(loop)):
        entry = loop[i]
        img = cv2.imread("images/" + entry[0] + ".jpg")
        cv2.imwrite(folderName + str(i) + "_" + entry[0] + ".jpg", img) 
        
    
    return 0

SORT_IMAGES("AlbumList.txt", 3)