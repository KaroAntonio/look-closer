import numpy as np
from PIL import Image
from scipy.ndimage.filters import convolve
from scipy.misc import imresize

def rad(s):
	return int(s/2)

def correlate(img, mask, x,y): 
	''' 
	give the correlation score of the filter 
	centered at that point in the img 
	assume the img is normalized to 1/0
	assume the img is padded, and that x, y are adjusted accordingly
	'''
	if (img.max() > 1 or img.min() < 0):
		raise Exception('img must be normalized to (0,1)')
	s = mask.shape[0]
	o = rad(s)
	img_crop = img[y-o:y+o+1,x-o:x+o+1]

	return (img_crop * mask).mean()/ mask.mean()

def circle_mask(s):
	y,x = np.ogrid[-s: s+1, -s: s+1]
	mask = s**2 - (x**2+y**2) + 1
	low_values_indices = mask < 0
	mask[low_values_indices] = 0  
	hi_vals_indices = mask > 0
	mask[hi_vals_indices] = 1
	return mask

def threshold(img,thresh):
	hi_idxs = img > thresh
	lo_idxs = img < thresh 
	img[hi_idxs] = 1
	img[lo_idxs] = 0
	return img

def cover_point(img,s, x,y):
	r = rad(s) 
	mask = circle_mask(r)
	img_mask = np.zeros(img.shape)
	img_mask[y-r:y+r+1,x-r:x+r+1] = mask
	mask_idxs = img_mask == 1
	img[mask_idxs] = 0.5 

def display_img(img):
	import matplotlib.pyplot as plt
	plt.imshow(img,cmap='Greys')
	plt.show()

def display_strokes(img, strokes,s):
	r = rad(s) 
	stroke_img = np.zeros(img.shape)
	for stroke in strokes:
		for p in stroke:
			cover_point(stroke_img, s, p[0], p[1])

	display_img(stroke_img)
	'''
	import matplotlib.pyplot as plt
	fig=plt.figure(1)
	plt.axis([0,img.shape[0],0,img.shape[1]])
	ax=fig.add_subplot(1,1,1)
	for stroke in strokes:
		for p in stroke:
			circle = plt.Circle(p, radius=r, color='b')
			ax.add_patch(circle)
	plt.show()
	'''


def rgb2grey(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def spiral(X, Y):
    x = y = 0
    dx = 0
    dy = -1
    for i in range(max(X, Y)**2):
        if (-X/2 < x <= X/2) and (-Y/2 < y <= Y/2):
            yield (x+int(X/2)-1, y+int(Y/2)-1)
            # DO STUFF...
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
       	x, y = x+dx, y+dy

def choose_point(img, s):
	o = rad(s) 
	mask = circle_mask(o)
	points = convolve(img, mask,mode='constant', cval =0)/mask.sum()
	
	X = img.shape[1]
	Y = img.shape[0]
	for x,y in spiral(X, Y):
	#for x,y in ((x,y) for x in range(X) for y in range(Y)):
		if points[y,x] > 0.8:
				return (x,y)
	return None

def choose_stroke(img, s, length):
	'''
	img -> [p1,p2,...]
	return a single stroke covering as much of the image as possible of a certain length, of a certain stroke width
	
	'''
	o = rad(s) 

	p = choose_point(img,s)
	cover_point(img,s, p[0],p[1])
	stroke = [p]
	while len(stroke) < length and p:
		y = p[1]
		x = p[0]
		left = max(0, x-s-o)
		right = min(img.shape[1],x+s+o+1)
		top = max(0, y-s-o)
		bottom = min(img.shape[0],y+s+o+1)
		p = choose_point(img[top:bottom,left:right], s)

		if p: 
			p  = p[0]+left, p[1]+top
			cover_point(img,s, p[0],p[1])
			stroke += [p]

	return stroke

def choose_strokes(img, s, length, n=None):
	''' 
	assume img is thresholded to contain only {0,1}
	return an array of paths s.t. the paths cover as much of the black of the image as possible)
	stroke = [(x1,y1),(x2,y2)]
	return [stroke0, stroke1, ... ]
	'''
	
	# make a copy of the image that we'll use to keep track of where the strokes have covered
	stroke_img = img.copy()
	strokes = []
	while choose_point(stroke_img, s) and (len(strokes) < n or not n):
		print(len(strokes))
		strokes += [choose_stroke(stroke_img, s , length)]

	print(stroke_img)
	
	return strokes

def avg_stroke_len(strokes):
	return sum([len(s) for s in strokes])/float(len(strokes))

if __name__ == '__main__':
	# find strokes by finding chains of connected lines that match a predefined 'line' filter
	# start from an arbitrary point in the image
	# 
	s = 3
	img = rgb2grey(np.array(Image.open('data/squanch_gash.jpg')))
	img_og = img
	img = imresize(img, (300,300))
	img = threshold(img/255., 0.5)
	img = 1-img
	#display_img(img)
	#img = np.ones((8,8))
	#img = np.pad(img, r, 'constant')

	#cv = correlate(img, mask, 10,10)

	strokes = choose_strokes(img, s, 5, 2000)
	display_strokes(img, strokes, s)
	#stroke = choose_stroke(img, s, 5)
