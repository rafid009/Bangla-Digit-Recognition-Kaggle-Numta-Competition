import numpy as np
import cv2
from glob import glob
import os
import sys

def brighten(img):
	img[:,:,:] += 50
	return img


os.chdir("./new_testing_f/")
im_files = glob('*.JPG')
print(im_files[0])
#sys.exit(0)
for j in range(len(im_files)): 
	img = cv2.imread(im_files[j], cv2.IMREAD_COLOR)
	thresholded = brighten(img)
	cv2.imwrite( "../brighten/"+im_files[j], thresholded)
	print("done file ", im_files[j])
