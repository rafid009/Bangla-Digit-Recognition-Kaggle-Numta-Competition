import numpy as np
import cv2
from glob import glob
import os
import sys

def addborder(img):
	borderVale = 30
	COLOR = [255,255,255] # TOP, BOTTOM, LEFT, RIGHT
	constant= cv2.copyMakeBorder(img,borderVale,borderVale,borderVale,\
		borderVale,cv2.BORDER_CONSTANT,value=COLOR)
	return constant

os.chdir("./inverted/")
im_files = glob('*.png')
print(im_files[0])
#sys.exit(0)
for j in range(len(im_files)): 
	img = cv2.imread(im_files[j], 0)
	bordered = addborder(img)
	cv2.imwrite( "../added_border/"+im_files[j], bordered)
	print("done file ", im_files[j])
