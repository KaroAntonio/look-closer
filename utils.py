import cPickle
import numpy as np
from scipy.misc import imresize
from filters import *
import os
from PIL import Image
from skimage.morphology import disk
from skimage.filters import rank

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def crop_centre_square(img):
    h = img.shape[0]
    w = img.shape[1]
    dim = min((h,w))
    w_pad = (w-dim)//2
    h_pad = (h-dim)//2
    return img[h_pad:h_pad+dim,w_pad:w_pad+dim]

def normalize(img):
    # histogram equalization (do better)
    normed = img / float(img.max())
    return normed

def equalize(img):
    selem = disk(30)
    return rank.equalize(img, selem=selem)

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

class ButterLoader:
	def __init__(self,fid,batch_size=100):
		f = open(fid,'r')
		self.imgs = cPickle.load(f)
		self.i = 0
		self.batch_size = batch_size

	def next_batch(self):
		self.i = (self.i+self.batch_size) % (len(self.imgs) - self.batch_size)
		batch = self.imgs[self.i:self.i+self.batch_size]
		batch = np.array(batch)
		batch = batch.reshape(100,4096)/255.
		return batch

def load_butterflies(fid,batch_size = 100):
	""" 
	Returns a batch loader for the x train 
	fid: the path to a pickle file
	"""
	f = open(fid,'r')
	imgs = cPickle.load(f)
	i = 0	
	while (i+batch_size < len(imgs)):
		batch = imgs[i:i+batch_size]
		i += batch_size
		batch = np.array(batch)
		batch = batch.reshape(100,4096)/255.
		yield batch

def preprocess_butterflies(dataset_dir,save_dir,size=64):
	data = []
	fids = list(os.listdir(dataset_dir))
	for i,fid in enumerate(fids):
		print("%s / %s" % (i,len(fids) ))
		img_data = Image.open(dataset_dir+fid)	
		img = np.asarray( img_data, dtype='uint8' )
		img = crop_centre_square(img)
		img = imresize(img,(size, size))
		ps = PencilSketch((size,size))
		img = ps.render(img)
		img = rgb2gray(img)
		img = normalize(img)
		img = equalize(img)
		img = imresize(img,(64,64))
		data += [img]
	
	f = open(save_dir+'pickled_butterflies','w')
	cPickle.dump(data,f)

if __name__ == '__main__':
	dataset_dir = 'data/leedsbutterfly/images/'
	pickle_dir = 'data/leedsbutterfly/'
	#preprocess_butterflies(dataset_dir,pickle_dir)
	bl = ButterLoader(pickle_dir+'pickled_butterflies')
