#======================================================================
# Github: https://github.com/thjsimmons
#======================================================================

import cv2
import numpy as np

def getColors(path): 
    # 24 "Macbeth" colors and their RGB coordinates read to global list of colors MACBETH_LIST,
    # with form [["name", [r,g,b]], ["name", [r,g,b]], ... ].
    # RGB to Macbeth table source is "sRGB" column on page 5 of refs/RGB2MacbethPaper.pdf.
    f = open(path)
    LIST = []
    while True:
        line = f.readline()
        if not line:
            break
        l = line.split(",")
        LIST.append([l[0], [int(l[i+1]) for i in range(0,3)]])

    f.close()
    return LIST

# Create the global colors list:
global MACBETH_LIST 
MACBETH_LIST = getColors("./colors.txt")

def fromTxt(path):   # Read and retrieve list of image names from .txt file:
    f = open(path)
    l = []
    while True:
        line = f.readline()
        if not line:
            break
        l.append(line.replace(" ", "-").replace("\n", ""))
    f.close()
    return l

def fromFolder(folder): # Get image names from folder:
    import os
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

def newImg(w,h,color):  # Create const color image of with size wh:
    img = np.zeros((w, h, 3), np.uint8)
    img[:] = color
    return img

def average(img): # Get average color vector (RGB, BGR, or LAB format) from image:

    sum_rgb = np.array([0,0,0])

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            sum_rgb += np.array(img[i][j])
    
    return sum_rgb / (img.shape[0]*img.shape[1])

def mosaic(images): # Creates large mosaic image tiled with images:

    h = images[0].shape[0]
    w = images[1].shape[1]
    
    N = int(np.floor(np.sqrt(len(images))))
    M = newImg(N * w, N * h, [0,0,0])

    for i in range(0, N**2):
        x = (i % N) * w
        y = int(np.floor(i*1.0/N)*h)   
        M[y:y+h, x:x+w] = images[i]

    return M

def to_size(x, y, img): # Resize image to size xy:
    return cv2.resize(img, (x, y), interpolation = cv2.INTER_CUBIC)

def scale_by(factor, img): # Scale image size by factor:
    return cv2.resize(img, None, fx = factor, fy = factor, interpolation = cv2.INTER_CUBIC)

def bgr2lab(v): # Numerical conversion of BGR to LAB Coordinates without using cv2.BGR2LAB, which seems to be less accurate:
    v = v[::-1] 
    RGB = [100*( ((element/255.0+0.055)/1.055)**2.4 if element/255.0 > 0.04045 else (element/255.0)/12.92) for element in v]
    RGB2XYZ_MAT = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])

    XYZ = [round(element, 4) for element in np.matmul(RGB2XYZ_MAT,RGB)]
    XYZ = np.matmul(np.array([[1/95.047,0,0], [0, 1/100.0, 0], [0, 0, 1/108.883]]), XYZ) 
    XYZ = [element**0.33333 if element > 0.008856 else 7.787*element + 16.0/116 for element in XYZ]
    
    XYZ2LAB_MAT = np.array([[0, 116, 0], [500, -500, 0], [0, 200, -200]])
  
    return [round(element, 4) for element in np.matmul(XYZ2LAB_MAT, XYZ)+np.array([-16,0,0])]

def dist3(A, B): # 3D Euclidean vector distance formula:
    return ((A[0]-B[0])**2.0 + (A[1]-B[1])**2.0 + (A[2]-B[2])**2.0)**0.5

def dist3lab(A, B): # Convert from RGB to Lab, get dist3: 
    return dist3(bgr2lab(A), bgr2lab(B))

def distHST(H1, H2): # Compute sum of absolute value differencees between histograms or "L1 Distance":
    return sum([abs(H1[i] - H2[i]) for i in range(0, len(H1))])

def distCCV(V1, V2): # Sum L1 distances for the coherent & incoherent histograms of color coherence vectors:
    weight = 3 # weight given to coherence over incoherence
    return sum([weight * abs(V1[i][0] - V2[i][0]) + abs(V1[i][1] - V2[i][1]) for i in range(0, len(V1))])

#def rgb2lab(rgb): # Convert image format from RGB to CIE LAB:
#    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0][0]
    
#def rgb2bgr(rgb): # Convert image format from RGB to BGR:
#    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2BGR)[0][0]

#def bgr2rgb(bgr): # Convert image format from BGR to RGB:
#    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2RGB)[0][0]

#def bgr2lab(bgr): # Convert image format from BGR to RGB:
#    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]
