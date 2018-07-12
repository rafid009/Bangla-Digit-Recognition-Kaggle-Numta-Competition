import numpy as np
import cv2
from glob import glob
import os
import sys

def otsu(img):
	window_size = 11
	ret2,otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #
	return otsu


os.chdir("./white_to_median/")
im_files = glob('*.*')
print(im_files[0])
#sys.exit(0)
for j in range(len(im_files)): 
	img = cv2.imread(im_files[j], 0)
	thresholded = otsu(img)
	cv2.imwrite( "../otsu/"+im_files[j], thresholded)
	print("done file ", im_files[j])
