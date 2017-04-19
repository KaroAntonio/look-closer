from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import math
import xml.etree.ElementTree

def remove_bounding_box(fid, out_fid='out.svg'):
	''' janky way '''
	f = open(fid)
	lines = f.readlines()
	new_lines = lines[:11]+lines[19:]
	f.close()
	f = open(out_fid, 'w')
	f.writelines(new_lines)
	f.close()


def save_strokes(strokes, dip_freq=0, az=1, el=1, fid='strokes.svg'):
	''' handles 2d and 3d strokes '''
	import matplotlib.pyplot as plt

	fig = plt.figure()
	ax = plt.gca()
	if len(strokes[0][0]) == 3:
		ax = Axes3D(plt.gcf())
		ax.view_init(elev=el, azim=az) 
		ax.azimuth = az
		ax.elevation = el
		print('WARNING: 3d is not properly formatted for mDraw')

	for i,stroke in enumerate(strokes):
		if dip_freq and i%dip_freq==0:
			z = [0,1] if len(stroke[0]) == 3 else 0
			ax.plot([0,1], [0,1], z, 'k-',lw=2)
		xs = [p[0] for p in stroke]
		ys = [p[1] for p in stroke]
		zs = [p[2] for p in stroke] if len(stroke[0]) == 3 else 0
		ax.plot(xs, ys, zs,'k-', c='r',lw=1)

	plt.axis('off')
	#plt.gca().set_position([0, 0, 1, 1])
	ax.set_aspect('equal')
	plt.savefig(fid)
	remove_bounding_box(fid,fid)

def save_wireframe(X,Y,Z, fid='wireframe_3d.svg'):
	'''
	X,Y,Z are 2D arrays
	'''
	fig = plt.figure(figsize=plt.figaspect(1))
	ax = fig.add_subplot(1, 1, 1, projection='3d')

	ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)

	plt.axis('off')
	plt.gca().set_aspect('equal')
	plt.savefig(fid)
	remove_bounding_box(fid,fid)

if __name__ == '__main__':
	an = np.arange(0,100,0.5)

	# draw vertical line from (70,100) to (70, 250)
	plt.plot([70, 70], [100, 250], 'k-', lw=2)

	# draw diagonal line from (70, 90) to (90, 200)
	plt.plot([70, 90], [90, 200], 'k-')

	#plt.figure(figsize=[6,6])
	plt.axis('off')
	plt.gca().set_position([0, 0, 1, 1])
	plt.savefig("test.svg")
