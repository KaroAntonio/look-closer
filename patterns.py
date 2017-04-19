import math
import numpy as np
from save_svg import *
import noise


def fx(i):
	dy = math.cos(i*0.3)
	return [
			i*2+5,
			i*2+5
		]

def fy(i):
	dy = 3*math.cos(i*0.3)
	return [
			i+5,
			55+i+dy+5
		]

def fx1(i):
	dx = math.cos(i*0.5)-(i**0.8)
	return [
			i,
			55+i	
		]

def fy1(i):
	dy = 2*math.cos(i*0.3)
	dy1 = 2*math.cos(i*0.3+math.pi/2)
	return [
			i,
			i+dy
		]

def plot_circles(plt):
	import matplotlib.patches as patches
	for i in range(30):
		#circle = plt.Circle((25+i, 25+i), (i+1)*2, color='black', fill=False)
		dr = (1/30.)*math.cos(i*0.2+1)
		plt.gca().add_patch(
			patches.Circle(
				(i/60.+0.25, dr*2+0.3),   # (x,y)
				0.2+dr,          # radius
				color='black',
				fill=False
			)
		)

def circle_points(x,y,r, n=50):
	step = math.pi/n
	th = np.arange(0,2*math.pi+step,step)
	xs = r * np.cos(th) + x
	ys = r * np.sin(th) + y
	return np.array([xs,ys]).T

def noise_line(i,n):
	xs = np.arange(n)
	ys = np.array([noise.pnoise2(e/float(n)*4,i*4) for e in xs])
	ys *= 50
	ys += i*230
	return np.array([xs,ys]).T

def curve(i,n):
	xs = np.arange(n)+i
	ys = 30 * np.sin((xs-i -20) * 1/25. ) 
	ys += i*3
	ys *= (math.sin(i*(math.pi/50.))*0.5)
	ys += i*5
	return np.array([xs,ys]).T

def curve2(i,n):
	xs = np.arange(n)+i
	ys = 30 * np.sin((xs-i -20) * 1/25. ) 
	ys += i*3
	ys *= (math.sin(i*(math.pi/50.))*0.5)
	ys += i*4
	return np.array([ys,xs]).T

def noise_mesh():
	lines = []
	for i in range(100):
		lines += [noise_line(i/100.,200)]
	for i in range(0):
		lines += [circle_points(75+i*2,75+i*2,i*4)]
	return lines

def gen_curves(n):
	lines = []
	for i in range(100):
		lines += [curve(i,200)]
	for i in range(0):
		lines += [circle_points(75+i*2,75+i*2,i*4)]
	return lines

def gen_lines(n):
	lines = []
	for i in range(n):
		lines += [[fx(i),fy(i)]]
	for i in range(n):
		pass
		#lines += [[fx1(i),fy1(i)]]
	return lines

def gen_3d_mesh():
	''' gen lines in 3d space '''


	return lines

def wireframe_to_lines(X,Y,Z):
	lines = []
	for i in range(len(X)):
		lines += [[[X[i,j],Y[i,j],Z[i,j]] for j in range(X.shape[1])]]
	for j in range(len(X)):
		lines += [[[X[i,j],Y[i,j],Z[i,j]] for i in range(X.shape[1])]]
	return lines

def gen_wireframe(n):
	idxs = np.matrix(np.arange(n))
	ones = np.matrix(np.ones(n))
	X = ones.T * idxs/float(n)
	Y = X.T
	Z = np.sin(X*math.pi*1+1)-0.5*np.cos(Y*math.pi*1-3)
	return X,Y,Z

def gen_sphere(rs=1, n=50, x=0, y=0, z=0):
	''' return lines that compose a sphere in 3d '''
	lines = []
	n = float(n)
	#rs = 1 # radius sphere
	for i in range(int(n)):
		# radius inscribed circle
		a = abs(1 - i/(n/2.))*rs # distance along sphere axis
		rc = math.sqrt(rs**2 - a**2)
		c2d = circle_points(x,y,rc, 100)
		c3d = np.array([list(p)+[i/n*rs+(z-(rs/2.))] for p in c2d])
		lines += [c3d]
	return np.array(lines)

def perlin_sphere(rs=1, n=50, x=0, y=0, z=0, s=1, ns=0.04):
	sphere = gen_sphere(rs, n, x, y, z)
	rs = float(rs)
	for line in sphere:
		for p in line:
			px = x-p[0]
			py = y-p[1]
			pz = z-p[2]
			nse = noise.pnoise3(px/(rs*s),py/(rs*s), pz/(rs*s)) 
			p[0] += (x-p[0])*nse*rs*ns
			p[1] += (y-p[1])*nse*rs*ns
			p[2] += (z-p[2])*nse*rs*ns
	return sphere

def project_p3d(P,p3d):
	'''
	p3d: [x,y,z]
	p2d: [x,y]
	'''
	p3d = np.concatenate([p3d,[1]])
	p2d = P * np.matrix(p3d).T
	p2d = np.array(p2d.T)[0]
	p2d = (p2d/(p2d[-1]+1e-10))[:2]
	return p2d

def transform_p3d(T,p3d):
	'''
	T: transformation matrix
	p3d: [x,y,z]
	'''

def project_lines(lines_3d, P=None ):
	if not P:
		P = np.matrix([[1, 0, 0, 0],
					[0, 1, 0, 0],	
					[0, 0, 1, 0]])

	lines_2d = []
	for line_3d in lines_3d:
		line_2d = []
		for p3d in line_3d:
			p2d = project_p3d(P,p3d)
			line_2d += [p2d]
		lines_2d += [line_2d]
	return np.array(lines_2d)

#lines = gen_curves(50)
#save_strokes(lines, dip_freq=0)

lines = perlin_sphere(60,70,150,0,50, s=0.4, ns=0.01)
lines = []
lines2 = perlin_sphere(70, 30, s=0.9, ns=0.02)
lines2 = gen_sphere(z=50)
#lines2 = []
lines_3d = list(lines) + list(lines2)  
lines_2d = project_lines(lines_3d)
save_strokes(lines_2d)

