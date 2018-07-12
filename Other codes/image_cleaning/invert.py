import numpy as np
import cv2
from glob import glob
import os
import sys

def invert(img):
	imagem = cv2.bitwise_not(img)
	return imagem


os.chdir("./all/")
im_files = glob('*.png')
print(im_files[0])
#sys.exit(0)
for j in range(len(im_files)): 
	img = cv2.imread(im_files[j], 0)
	inverted = invert(img)
	cv2.imwrite( "../inverted/"+im_files[j], inverted)
	print("done file ", im_files[j])
