# Based on (sRGB (GMB) page 5) 
# https://www.babelcolor.com/index_htm_files/RGB%20Coordinates%20of%20the%20Macbeth%20ColorChecker.pdf
# MacBeth Values in RGB format

def getColors(path): # path = "AlbumList.txt"
    # Returns list of form [[name0, rgb0], [name1, rgb1], ...]
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

MACBETH_LIST = getColors("./colors.txt")
global MACBETH_LIST 