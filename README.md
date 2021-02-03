
# SortByColor

### Requirements:
Python2.7, numpy, cv2 

### Description
Python functions for sorting a collection of images by color (tested with album covers).

Images can be compared by color in several ways, the simplest of which are sorting
by average color and sorting by dominant color. Comparing average colors uses color 
information over the whole image but fails to measure similarities between the (defining) 
component colors. Sorting by average then minimizes the rgb vector distances 
between average colors of adjacent images. 

Comparing by dominant color, via k-means clustering, first labels each pixel into 
the closest 1 of k most-occuring colors in the image (a size k palette). The 
dominant color is the rgb vector with the most-occuring label in the image. 
Like in sorting by average, sorting by dominant minimizes the rgb vector distance 
between dominant colors of adjacent images. This method fails when it discards too much 
relevant information about the non-dominant colors. 

A third method is comparing by color histogram. RGB pixels in the image are first binned into
1 of 24 color bins (the nearest bin RGB coordinate) and converted to a histogrm. 
Sorting by color histogram minimizes the L1-distance between adjacent image's histograms. 
This method succeeds in comparing proportions of different colors in images but fails 
in capturing a color's spatial distribution which can act to dillute its coherence. 

A fourth method uses color coherence vectors (CCVs) which are a pair of color
histograms, one for "coherent" pixels and one for "incoherent" pixels. Coherent pixels
are part of blob with size > threshold (e.g. 1 % of image size) and incoherent pixels 
are part of a blob with less. Sorting by CCV minimizes the sum of the L1-distances for 
both histograms between adjacent images.  This is one of the simplest color-comparison 
algorithms that uses spatial information. 

### mosaic2.jpg
![](refs/mosaic.jpg?raw=true)

