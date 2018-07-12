import numpy as np
import cv2
from glob import glob
import os
import sys

def invert(img):
	imagem = cv2.bitwise_not(img)
	return imagem

def removeRect(img):
	threshold_value = 0
	median_blur_value = 3 #7
	#img = cv2.imread('image (2).png')
	inverted = invert(img)
	ret,thresh2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
	ret,thresh3 = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
	minus = invert(thresh2 - img)
	#minus = minus*1.1
	median = cv2.medianBlur(minus, median_blur_value)
	return median

# cv2.imshow('Image', median)
# cv2.waitKey(0)
# cv2.imwrite('output.png', median)


os.chdir("./all/")
im_files = glob('*.png')
print(im_files[0])
#sys.exit(0)
for j in range(len(im_files)): 
	img = cv2.imread(im_files[j], 0)
	rectangle_removed = removeRect(img)
	cv2.imwrite( "../black_removed/"+im_files[j], rectangle_removed)
	print("done file ", im_files[j])
