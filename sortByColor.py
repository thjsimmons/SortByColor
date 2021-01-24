"""
Returns N most prominant colors in image 

"""
import cv2
import numpy as np

path = "images/"
name = "green"
ext = ".jpg"

imName = path + name + ext
my_img = cv2.imread(imName)

n_colors = 3

# cv2 k-means clustering algorithm:
def domColor(img, nColors): # https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
    pixels = np.float32(img.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return dominant, palette 

def main(img):
    # Get 3 most dominant RGB values 
    dominant, palette = domColor(my_img, n_colors)
    # Write pure color images to colors folder:
    count = 0 
    for color in palette:
        color_img = my_img 
        color_img[0:my_img.shape[0],0:my_img.shape[1]] = color
        cv2.imwrite("colors/" + name + "_" + str(count) + "_of_" + str(n_colors) + ext, color_img) 
        count = count + 1
    # Write dominant color image to images folder "...avg.jpg"
    avg_img = img 
    avg_img[0:img.shape[0],0:img.shape[1]] = dominant
    cv2.imwrite(path + name + "_avg" + ext, avg_img) 
    return 0

main(my_img)
