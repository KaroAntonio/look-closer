import math
import numpy as np
from save_svg import *
import noise
import copy
from geometry_primitives import *
from grow_lines import *
from obj import *
from topography import *

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

def gen_wireframe(n):
	idxs = np.matrix(np.arange(n))
	ones = np.matrix(np.ones(n))
	X = ones.T * idxs/float(n)
	Y = X.T
	Z = np.sin(X*math.pi*1+1)-0.5*np.cos(Y*math.pi*1-3)
	return X,Y,Z

def gen_form(rs=1, n=50, rf=None, af=None, x=0, y=0, z=0, cp=200):
	''' 
	return lines that compose a sphere in 3d 
	cp: the number of points per circle
	rf: radius funtction
	'''
	lines = []
	n = float(n)
	#rs = 1 # radius sphere
	for i in range(int(n)):
		# radius inscribed circle
		a = af(n,i,rs)
		rc = rf(rs, a) 
		c2d = circle_points(x,y,rc, cp)
		# circle z coord
		cz = i/(n/2.)*rs+(z-(rs/2.))
		c3d = np.array([list(p)+[cz] for p in c2d])
		lines += [c3d]
	return np.array(lines)

def perlin_cubes(d=1, n=50, x=0, y=0, z=0, s=1, ns=0.5, o=3):
	step = d/float(n)
	cubes = []
	for i in range(1,n+1):
		cube = unit_cube()
		cube = scale_lines(cube,i*step,i*step,i*step)
		thx = i*step * math.pi*0.15
		thy = i*step * math.pi*0.1
		thz = i*step * math.pi*0.08*0
		#cube = rotate_lines(cube,thx,thy,thz)
		cube = apply_perlin(cube, s, ns, o)
		cubes += list(cube)
	return np.array(cubes)

def sphere_array():
	n = 5
	n2 = n/2
	array = []
	for i in range(-n2,n2+1):
		for j in range(-n2,n2+1):
			sphere = perlin_sphere(rs=1,n=10, s = 0.9, ns = (0)*.1) 
			th_x = (math.pi*0.5)
			th_z = math.pi*n/2. * 0
			sphere = rotate_lines(sphere,th_x = th_x, th_z=th_z)
			sphere = translate_lines(sphere, dz=20,dx =i*2, dy=j*2)

			array += list(sphere)

	return np.array(array)

def gen_triangles(n=50):
	triangles = []
	for i in range(n):
		for j in range(1):
			t = unit_triangle(40)
			n = float(n)
			di = i/n
			dj = j/n 
			th_x = math.pi*((-.5+dj)*0.1)*0 
			th_z = math.pi*((-.5+di)) 
			#t = rotate_lines(t,th_x=th_x, th_z=th_z)
			sy = n
			sx = n * ((1/3.)**.5)
			a = ((1/3.)**.5)
			b = .5
			t = scale_lines(t,di*a,di*b,1)
			triangles += list(t)
			n = int(n)

	return triangles 

def apply_sin(lines):
	lines = [np.array(l) for l in lines]
	for line in lines:
		for v in line:
			v[0] += math.sin(v[1]*2+math.pi)*0.4
	return lines

def apply_perlin_axis(lines):
	''' 
	apply 2d perlin noise along a specific axis, 
	given the other 2 as input
	'''
	mx = 0
	my = 0
	mz = 0
	c = [0,0]
	for line in lines:
		for p in line:
			if abs(p[0])>mx: mx = p[0]
			if abs(p[1])>my: my = p[1]	
			if abs(p[2])>mz: mz = p[2]	

	lines = [np.array(l) for l in lines]
	for line in lines:
		for v in line:
			v[0] += noise.pnoise2(float(v[1])/my, v[2]/mz)
	return lines

'''
# perlin blob
p_sphere = perlin_sphere(1, 130, s=0.9, ns=0.6, o=3)
p_sphere = rotate_lines(p_sphere,th_x = math.pi*0.49, th_z=math.pi*.0)
p_sphere = translate_lines(p_sphere, dz=50)

# perlin blob 2
p_sphere2 = perlin_sphere(50, 30, s=0.9, ns=0.05)
p_sphere2 = rotate_lines(p_sphere2,th_x = -math.pi*0.1, th_y=math.pi*0.3)
p_sphere2 = translate_lines(p_sphere2, dz=100)


# perlin form 
af = lambda n,i,rs: ((i/n)*rs)*0.95
rf = lambda rs,a: 1.4+a
p_form = gen_form(1, n=50, rf=rf, af=af)
p_form = apply_perlin(p_form, ns=0.5, s=0.9, o=4)
p_form = rotate_lines(p_form, th_x = math.pi*0., th_y=math.pi*1)
p_form = translate_lines(p_form, dz=8)

# perlin form 2
af = lambda n,i,rs: (i/n)*rs
rf = lambda rs,a: 1 
p_form2 = perlin_form(1, 120, rf=rf,af=af, s=0.9, ns=0.5, o=3)
p_form2 = rotate_lines(p_form2, th_y = math.pi*0., th_x=math.pi*.48)
p_form2 = translate_lines(p_form2, dz=100, dx=0.3)

# another sphere
small_sphere = gen_sphere(rs=1,n=130) 
small_sphere = rotate_lines(small_sphere,th_x = math.pi*0., th_z=math.pi*0)
small_sphere = translate_lines(small_sphere, dz=50, dy=-.5, dx=.5)

# create sphere
sphere = gen_sphere(rs=1,n=110) 
sphere = rotate_lines(sphere,th_x = math.pi*0.49, th_z=math.pi*.0)
sphere = translate_lines(sphere, dz=20)

# create cube
cube = unit_cube()
cube = rotate_lines(cube, th_x = math.pi/5., th_y=math.pi/5)
cube = translate_lines(cube, dz=50)

# wireframe
wireframe = wireframe_to_lines(*gen_wireframe(100))
wireframe = rotate_lines(wireframe, th_z=math.pi*0.5)
wireframe = apply_perlin(wireframe, s=1.1, ns=0.5, o=3)
wireframe = apply_sin(wireframe)
wireframe = rotate_lines(wireframe, th_x = math.pi*-0.5, th_z=math.pi*0.01)
wireframe = translate_lines(wireframe, dz=3, dy=-0.4, dx=-0.3)

s1 = copy.deepcopy(small_sphere)
s2 = copy.deepcopy(small_sphere)
s3 = copy.deepcopy(small_sphere)
s4 = copy.deepcopy(small_sphere)

o=0.11
s1 = translate_lines(s1, dy=o, dx=o)
s2 = translate_lines(s2, dy=o, dx=-o)
s3 = translate_lines(s3, dy=-o, dx=o)
s4 = translate_lines(s4, dy=-o, dx=-o)
spheres = list(s1) + list(s2) + list(s3) + list(s4) 
spheres = rotate_lines(spheres,th_z = math.pi*.25, th_y=math.pi*0)
spheres = translate_lines(spheres, dz=50)

p_cubes = perlin_cubes(n=40,ns=0)
p_cubes = rotate_lines(p_cubes,th_y = math.pi*-.05, th_x=math.pi*-.06)
p_cubes = translate_lines(p_cubes, dz=2)


triangles = gen_triangles()
triangles = translate_lines(triangles, dz=2, dx=0)
triangles = apply_perlin(triangles, s=0.7, ns=0.8, o=3)
triangles = rotate_lines(triangles,th_z = math.pi*-.25, th_x=math.pi*0)
triangles = translate_lines(triangles, dz=50)
'''

fid = 'data/obj/rind.obj'
objs = load_obj(fid)
oid = objs.keys()[0]
vs,fs,vns = objs[oid]
cl = build_topography(vs, fs, n=120,ax=1)
cl = apply_perlin(cl, s=0.8, ns=0.5, o=3)
cl = rotate_lines(cl,th_x = math.pi*0.2, th_y=math.pi*.1)
cl = translate_lines(cl, dz=200)

#mesh = obj_to_lines(objs[oid])
#mesh = rotate_lines(cl,th_x = math.pi*0.0, th_y=math.pi*.0)
#mesh = translate_lines(mesh, dz=200)

lines = []
lines2 = []
lines = cl 
#lines2 = mesh

lines_3d = list(lines) + list(lines2)  
lines_2d = project_lines(lines_3d, f=1)

#lines_2d = apply_perlin2d(lines_2d,ns=.3,s=.3,o=2)

save_strokes(lines_2d, dip_freq=0)

