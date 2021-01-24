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


def binImg(dom_img, n_bins, path, ext, name):
    # img is the input.jpg, rgb is input_color.jpg (pure dominant color jpg)
    # get hue of the dominant color 
    dom_img = cv2.cvtColor(dom_img, cv2.COLOR_BGR2HSV)
    
    # Write binned colors to images folder:
    hue = dom_img[0,0,0]
    hue_bin = 180.0/n_bins * np.ceil(n_bins * hue/180.0)
    hue_img = cv2.cvtColor(dom_img, cv2.COLOR_BGR2HSV)
    hue_img[0:hue_img.shape[0],0:hue_img.shape[1]] = [hue_bin, 166, 166]
    cv2.imwrite("images/" + name + "_" + "bin" + ext, cv2.cvtColor(hue_img, cv2.COLOR_HSV2BGR)) 

    # Write possible colors to bins folder:
    for i in range(0, n_bins):
        hue_of_bin = 180.0/n_bins * i 
        #print(hue_of_bin)
        bins_img = cv2.cvtColor(dom_img, cv2.COLOR_BGR2HSV)
        bins_img[0:bins_img.shape[0], 0:bins_img.shape[1]] = [hue_of_bin, 166, 166]
        rgb_bins_img = cv2.cvtColor(bins_img, cv2.COLOR_HSV2BGR)
        cv2.imwrite("bins/" + "color" + str(i+1) + ext, rgb_bins_img) 
    
    return 0

def main():
    path = "images/"
    ext = ".jpg"
    imList = ["banana", "currents", "inner", "meta", "orc", "poly"]

    n_colors = 2 # Number of dominant colors to search for 
    n_bins = 6 # Number of colors to sort dominant into

    sortedImgs = [] # [[name, hue], [name, hue], ...]
    for name in imList:
        print(path + name + ext)
        img = cv2.imread(path + name + ext)
        # Get 3 most dominant RGB values 
        dominant, palette = domColor(img, n_colors)
        # Write pure color images to colors folder:
        count = 0 
        for color in palette:
            color_img = img 
            color_img[0:color_img.shape[0],0:color_img.shape[1]] = color
            cv2.imwrite("colors/" + name + "_" + str(count) + "_of_" + str(n_colors) + ext, color_img) 
            count = count + 1
        # Write dominant color image to images folder 
        dom_img = img 
        dom_img[0:dom_img.shape[0],0:dom_img.shape[1]] = dominant
        cv2.imwrite(path + name + "_color" + ext, dom_img) 
        binImg(dom_img,n_bins, path, ext, name)
        
    return 0

main()