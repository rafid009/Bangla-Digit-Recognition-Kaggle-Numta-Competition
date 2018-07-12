import numpy as np
import cv2
from glob import glob
import os
import sys

def white_to_median(img):
	median = np.median(img)
	img[img==255] = median
	return img


os.chdir("./blurred/")
im_files = glob('*.png')
print(im_files[0])
#sys.exit(0)
for j in range(len(im_files)): 
	img = cv2.imread(im_files[j], 0)
	thresholded = white_to_median(img)
	cv2.imwrite( "../white_to_median/"+im_files[j], thresholded)
	print("done file ", im_files[j])
