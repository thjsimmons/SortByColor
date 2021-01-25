"""
Returns N most prominant colors in image 

"""
import cv2
import numpy as np

# in OpenCV, Hue range is [0,179], saturation range is [0,255], and value range is [0,255]

# cv2 k-means clustering algorithm:
def domColor(img, n_colors): # https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
    
    pixels = np.float32(img.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return dominant, palette 


def insertSort(sortedImgs, entry):
    # insertion sort array of type [["banana", 10], ["currents", 30], ["inner", 5], ["meta", 7], ["orc", 1], ["poly", 67]]
    placed = False
    if sortedImgs == []:
        sortedImgs.append(entry)
    else:
        for i in range(0, len(sortedImgs)):
            if entry[1] <= sortedImgs[i][1]:
                sortedImgs.insert(i, entry)
                placed = True
                break
        if not placed:
            sortedImgs.append(entry)
        
    return sortedImgs

def sortImg(sortedImgs, dom_img, path, ext, name):
    # Get Hue of dom_img, make entry:
    dom_img = cv2.cvtColor(dom_img, cv2.COLOR_BGR2HSV)
    hue = dom_img[0,0,0]
    entry = [name, hue] 
    # Insert sort by hue:
    sortedImgs = insertSort(sortedImgs, entry)
    return sortedImgs


def dist3(A, B):
    return np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2)

def minDist(p, q_vec):

    dist = 999999
    min_q = []

    for i in range(0, len(q_vec)):
        q = q_vec[i]
        if dist3(p[1], q[1]) < dist:
            dist = dist3(p[1], q[1])
            #min_q = q
            mindex = i

    return dist, mindex 

def RGBDistanceSort(rgb_list): # Unused example: https://www.geeksforgeeks.org/find-simple-closed-path-for-a-given-set-of-points/
    """
    Goal: Sort RGB Vectors by creating a closed
    loop in RGB space where points connecting eachother minimize the 
    distance of the loop 
    """
    rgb_loop = [] # [["banana", 10], ["currents", 30], ["inner", 5], ["meta", 7], ["orc", 1], ["poly", 67]]

    for i in range(0, len(rgb_list)):
        if i == 0:
            rgb_loop.append(rgb_list.pop(0)) # rgb_loop[0]
        else: # i > 0
            # distance between prev loop point (removed from rgb_list) and next point in rgb_list
            dist, mindex = minDist(rgb_loop[i-1], rgb_list) 
            rgb_loop.append(rgb_list.pop(mindex)) # rgb_loop[i], rgb_loop[i-1] on next iteration

    return rgb_loop

def ScatterPlot3D():
    return 0

def get_albums():
    # Open album list and iterate through
    f = open("AlbumList.txt")
    l = []
    while True:
        line = f.readline()
        # break loop when done so we don't try stupid shiz
        if not line:
            break
        # get art for every song and add it to list when exists
        #get_artwork(line)
        l.append(line.replace(" ", "-").replace("\n", ""))
    f.close()
    return l

def main():
    path = "images/"
    ext = ".jpg"
    imList = get_albums() #["banana", "currents", "inner", "meta", "orc", "poly"]
    print("imList = ", imList)
    n_colors = 5 # Number of dominant colors to search for 
    #n_bins = 6 # Number of colors to sort dominant into

    rgb_list = []
    sortedImgs = [] # [[name, hue], [name, hue], ...]
    for name in imList:
        print(path + name + ext)
        img = cv2.imread(path + name + ext)
        
        # Get "n_colors" most dominant RGB values 
        dominant, palette = domColor(img, n_colors)
        rgb_list.append([name, dominant])
        # Get pure dominant color image:
        #dom_img[0:dom_img.shape[0],0:dom_img.shape[1]] = dominant
        # Insert image name and hue into sortedImgs
        #sortedImgs = sortImg(sortedImgs, dom_img, path, ext, name)

    rgb_loop = RGBDistanceSort(rgb_list)
    print("rgb_loop = ", rgb_loop)
    # Fill rgbLoop folder:
    for i in range(0, len(rgb_loop)):
        entry = rgb_loop[i]
        img = cv2.imread(path + entry[0] + ext)
        cv2.imwrite("rgbLoop/" + str(i) + "_" + entry[0] + ext, img) 
        
    return 0

main()