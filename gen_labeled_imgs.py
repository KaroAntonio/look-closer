import numpy as np
import cv2
from PIL import Image
from scipy import misc
from scipy.ndimage import sobel
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage as ndi
from skimage import feature
from skimage.filters import rank
from skimage.morphology import disk
from ftdetect import features as ft_feat
import phasepack as pp
from scipy.misc import imresize
from filters import *
from utils import *

def to_rgb(im):
    # I think this will be slow
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

def show_img(img):
	import matplotlib.pyplot as plt
	# draw things to plot ...
	plt.imshow(img,cmap = plt.get_cmap('gray'))
	plt.show()

def sobel_edges(im,mode='constant',):
	dx = sobel(im, 1,mode=mode)  # horizontal derivative
	dy = sobel(im, 0,mode=mode)  # vertical derivative
	mag = np.hypot(dx, dy)  # magnitude
	mag *= 255.0 / np.max(mag)  # normalize (Q&D)

	return mag

def sharpen(f):
	kernel = np.zeros( (9,9), np.float32)
	kernel[4,4] = 2.0   #Identity, times two! 
	boxFilter = np.ones( (9,9), np.float32) / 81.0
	kernel = kernel - boxFilter
	return cv2.filter2D(f, -1, kernel)

def threshold(img,thresh):
    hi_idxs = img > thresh
    lo_idxs = img < thresh
    img[hi_idxs] = 1
    img[lo_idxs] = 0
    return img

def simplify(img):
	''' return a simplified version of the img '''
	img = gaussian_filter(img, 8)
	img = normalize(img)
	img = threshold(img, img.mean())
	return img

def gen_img_pair(img):
	simple = simplify(img)
	simple_edges = sobel_edges(simple,mode='constant')
	img_edges = sobel_edges(img,mode='constant')
	img_pair = np.concatenate([simple_edges.T, img_edges.T]).T	
	return img_pair

def gen_pairs(in_path, out_path, cond, size=256):
	'''
	cond: a string that is a condition on the path of the image
	'''

	if in_path.endswith('/'):
		in_path = in_path[0:-1]
	if out_path.endswith('/'):
		out_path = out_path[0:-1]
	
	fids = [fid for fid in list(os.listdir(in_path)) if fid.startswith(cond)]
	n = len(fids)	
	n_train = int(0.5 * n)
	n_val = int(0.25 * n)
	n_test = int(0.25 * n)

	for i,fid in enumerate(fids): 
		print(i,len(fids))
		img_data = Image.open(in_path+'/'+fid)
		img = np.asarray( img_data, dtype='uint8' )
		img = crop_centre_square(img)
		img = imresize(img,(size, size))
		ps = PencilSketch(img.shape[:-1])
		ps_img = ps.render(img)
		img = rgb2gray(ps_img)
		img = normalize(img)
		pair = to_rgb(gen_img_pair(img))
		
		if pair.max() < 1: pair *= 255
		pair = 255 - pair
		result = Image.fromarray((pair).astype(np.uint8))

		if i < n_train: subdir = 'train'	
		elif i < n_train+n_val: subdir = 'val'
		else: subdir = 'test'

		result.save(out_path + '/' + subdir + '/' + fid)


in_path = 'data/leedsbutterfly/images/'
out_path = 'data/pairs'
condition = '001_'

img_data = Image.open(in_path+'003_0001.jpg')

img_data = Image.open('data/leedsbutterfly/images/001_0001.jpg')

size = 256  # size the image large first, and then resize it smaller, 
				# for better lines
#img = img_data.reshape(3,1024).T.reshape(32,32,3) #for cifar
img = np.asarray( img_data, dtype='uint8' )
og_img = img
img = crop_centre_square(img)
img = imresize(img,(size, size))
gray_img = rgb2gray(img)
ps = PencilSketch(gray_img.shape)
ps_img = ps.render(img)
img = rgb2gray(ps_img)
gray_img = img
img = normalize(img)
#img = equalize(img)
#img = sharpen(img)
#img = imresize(img,(64,64))

#img = pp.phasecong(img)[2] # whaaaat is this??
#img = pp.phasecong(img)[0]
img = sobel_edges(img,mode='constant') # very strokey!

#img = gaussian_filter(img, 10)

#show_img(img)

