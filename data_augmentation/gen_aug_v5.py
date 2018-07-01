import cv2
from skimage import transform
import matplotlib.pyplot as plt
import math
import numpy as np
import glob
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--nb_augments', type=int, default=2)
args = parser.parse_args()


np.random.seed(1)


def rotate_image(img):
	angle = np.random.uniform(-45, 45)
	
	height, width = img.shape
	image_center = (width / 2, height / 2)

	rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

	radians = math.radians(angle)
	sin = math.sin(radians)
	cos = math.cos(radians)
	bound_w = int((height * abs(sin)) + (width * abs(cos)))
	bound_h = int((height * abs(cos)) + (width * abs(sin)))

	rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
	rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

	rotated_mat = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))
	return rotated_mat


# salt pepper noise
def sp_noise(img):
	prob = np.random.uniform(0.01, 0.05)

	height, width = img.shape
	rnd = np.random.rand(height, width)
	noisy = img.copy()
	noisy[rnd < prob] = 0
	noisy[rnd > 1 - prob] = 255
	return noisy


def erase(img):
	height, width = img.shape
	edge = int(math.sqrt(height * width) * 0.1)
	res = img.copy()
	nblocks = np.random.randint(1, 5)

	for i in range(nblocks):
		ranx = int(edge * np.random.uniform(0, 1))
		rany = int(edge * np.random.uniform(0, 1))
		urx = np.random.randint(width * 0.2, width * 0.8 - edge)
		blx = urx + edge + ranx
		ury = np.random.randint(height * 0.2, height * 0.8 - edge)
		bly = ury + edge + rany
		res[urx:blx, ury:bly] = 0

	return res


def blur(img):
	m = np.random.choice([3, 5, 7, 9, 11])

	return cv2.blur(img, (m, m))


def shear(img):
	rate = np.random.uniform(-0.4, 0.4)
	
	p1 = np.float32([[1,1],[2,1],[1,2]])
	p2 = np.float32([[1,1],[2,1],[1 + rate,2]])

	shear_mat = cv2.getAffineTransform(p1, p2)

	res = cv2.warpAffine(img, shear_mat, img.shape)
	return res


def border(img):
	w = 2
	res = cv2.copyMakeBorder(img, w, w, w, w, 
		cv2.BORDER_CONSTANT, value=[0, 0, 0])
	return res


def random_image():
	d = np.random.choice(dirs)
	file = np.random.choice(files[d])
	return cv2.imread(file, 0)


def superimpose(img):
	p = np.random.uniform(0.1, 0.2)
	height, width = img.shape
	rand_image = random_image()
	back_img = cv2.flip(rand_image, 1)
	if back_img.shape != img.shape:
		back_img = cv2.resize(back_img, (width, height))
	res = cv2.addWeighted(back_img, p, img, 1 - p, 0)
	return res


def shift(img):
	height, width = img.shape
	p = np.random.uniform(0, 1)
	if p < 0.5:
		x, y = np.random.uniform(0, width * 0.4), 0
	else:
		x, y = 0, np.random.uniform(0, height * 0.4)

	trans_mat = np.float32([[1,0,x],[0,1,y]])
	res = cv2.warpAffine(img, trans_mat, img.shape, borderMode=cv2.BORDER_REPLICATE)
	return res


def zoom(img):
	h, w = img.shape
	p = np.random.uniform(1.2, 2)
	nh, nw = int(p * h), int(p * w)
	res = cv2.resize(img, 
		(nw, nh), 
		interpolation=cv2.INTER_LINEAR)
	ch, cw = nh // 2, nw // 2
	bh, bw = h // 2, w // 2
	cropped = res[ch - bh:ch + bh, cw - bw:cw + bw]
	return cropped


def hsv_shift(img):
	colored = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	hsved = cv2.cvtColor(colored, cv2.COLOR_BGR2HSV)
	h, s = np.random.randint(0, 100), np.random.randint(0, 100)
	hsved[:, :, 0] += h
	hsved[:, :, 1] += s
	colored = cv2.cvtColor(hsved, cv2.COLOR_HSV2BGR)
	return colored


# next idea: tight, bold, dashed, parchment


function_list = [rotate_image, sp_noise, erase, blur, shear, border, superimpose, shift, zoom]
function_prob = [0.20, 0.20, 0.1, 0.1, 0.10, 0.05, 0.1, 0.05, 0.1]


def random_augment(img):
	#np.random.shuffle(function_list)
	naugs = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
	#fns = function_list[:naugs]
	fns = np.random.choice(function_list, size=naugs, replace=False, p=function_prob)
	res = img.copy()
	for fn in fns:
		res = fn(res)
	u = np.random.uniform(0, 1)
	if u < 0.75:
		res = hsv_shift(res)
	return res


dirs = os.listdir('train')
files = {}

nb_augments = args.nb_augments

for d in dirs:
	files[d] = glob.glob('train/' + d + '/*.*')

for d in dirs:	
	for file in files[d]:
		filename = os.path.basename(file)
		# if filename[0] == 'e':
		# 	continue
		image = cv2.imread(file, 0)
		last_dot = file.rfind('.')
		for i in range(nb_augments):
			new_file = file[:last_dot] + str(i) + file[last_dot:]
			cv2.imwrite(new_file, random_augment(image))
	print('Dir', d, 'Done')


# image = cv2.imread('b00000.png', 0)
# cv2.imwrite('b.png', zoom(image))