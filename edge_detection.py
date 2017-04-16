import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import misc
from scipy.ndimage import sobel
from scipy import ndimage as ndi
from skimage import feature
from skimage.filters import rank
from skimage.morphology import disk
from ftdetect import features as ft_feat
#from pst import *
import phasepack as pp
from scipy.misc import imresize
from filters import *
from utils import *

def sobel_edges(im,mode='constant',):
	dx = sobel(im, 1,mode=mode)  # horizontal derivative
	dy = sobel(im, 0,mode=mode)  # vertical derivative
	mag = np.hypot(dx, dy)  # magnitude
	mag *= 255.0 / np.max(mag)  # normalize (Q&D)

	return mag

def canny_edges(img):
		
	img = np.gradient(img)[0]
	'''
	img_grey /= img_grey.max()
	img_grey = np.abs(img_grey - img_grey.max()/2)*2
	img_grey *= 255
	img_grey = np.abs(img_grey - img_grey.max()/4)*4/3
	'''
	#img_grey = ndi.gaussian_filter(img_grey,2)

	edges = feature.canny(img,sigma=6.5)
	return edges

def susan_edges(img):
	return ft_feat.susanEdge(img)

def sharpen(f):
	kernel = np.zeros( (9,9), np.float32)
	kernel[4,4] = 2.0   #Identity, times two! 
	boxFilter = np.ones( (9,9), np.float32) / 81.0
	kernel = kernel - boxFilter
	return cv2.filter2D(f, -1, kernel)

data_folder_fid = 'data/leedsbutterfly/images/'
img_data = Image.open(data_folder_fid+'003_0009.jpg')

img_data = Image.open('data/out_butter.jpg')

size = 256  # size the image large first, and then resize it smaller, 
				# for better lines
#img = img_data.reshape(3,1024).T.reshape(32,32,3) #for cifar
img = np.asarray( img_data, dtype='uint8' )
og_img = img
#img = crop_centre_square(img)
#img = imresize(img,(size, size))
gray_img = rgb2gray(img)
ps = PencilSketch(gray_img.shape)
ps_img = ps.render(img)
img = rgb2gray(ps_img)
img = normalize(img)
#img = equalize(img)
#img = imresize(img,(64,64))
'''
img = ps.render(img)
img = normalize(img)
img = equalize(img)
img = normalize(img)
img = sharpen(img)
img = equalize(img)
'''

#img = susan_edges(img)
#img = pst(img_grey)
#img = pp.phasecong(img)[2] # whaaaat is this??
#img = pp.phasecong(img)[0]
img = sobel_edges(img,mode='constant') # very strokey!
#img = canny_edges(img)

plt.imshow(img,cmap = plt.get_cmap('gray'))
plt.show()
