import numpy as np
import cv2
from glob import glob
import os
import sys

#img = cv2.imread('image (4).png')

def denoise(img):
	median = cv2.medianBlur(img,9)
	return median

# cv2.imshow('Image', median)
# cv2.waitKey(0)

#cv2.imwrite('output.png', median)

os.chdir("./all/")
im_files = glob('*.png')
print(im_files[0])
#sys.exit(0)
for j in range(len(im_files)): 
	img = cv2.imread(im_files[j], 0)
	denoised = denoise(img)
	cv2.imwrite( "../blurred/"+im_files[j], denoised)
	print("done file ", im_files[j])
