import numpy as np
import cv2
from glob import glob
import os
import sys

# def hitogram_equalization(img):
# 	median = np.median(img)
# 	img[img==255] = median
# 	return img

def equalize_hist_color(img):
    for c in range(0, 2):
       img[:,:,c] = cv2.equalizeHist(img[:,:,c])
    return img


os.chdir("./new_testing_f/")
im_files = glob('*.JPG')
print(im_files[0])
#sys.exit(0)
for j in range(len(im_files)): 
	img = cv2.imread(im_files[j], 0)
	thresholded = equ = cv2.equalizeHist(img)#hitogram_equalization(img)
	cv2.imwrite( "../histogram_equalized_new_testing_f/"+im_files[j], thresholded)
	print("done file ", im_files[j])
