import numpy as np
import cv2
from glob import glob
import os
import sys

def adaptiveThreshold(img):
	window_size = 11
	median = cv2.medianBlur(img, 9)
	th2 = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
	            cv2.THRESH_BINARY,window_size,2)
	#_, th2 = cv2.threshold(median,127,255,cv2.THRESH_TOZERO_INV)
	median_adaptive_mean = cv2.medianBlur(th2, 5)
	return median_adaptive_mean


os.chdir("./new_testing_f/")
im_files = glob('*.JPG')
print(im_files[0])
#sys.exit(0)
for j in range(len(im_files)): 
	img = cv2.imread(im_files[j], cv2.IMREAD_COLOR)
	thresholded = adaptiveThreshold(img)
	cv2.imwrite( "../new_testing_threshold/"+im_files[j], thresholded)
	print("done file ", im_files[j])
