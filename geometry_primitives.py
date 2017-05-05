import math
import numpy as np
import noise
import copy

def get_bounds(lines):
	'''
	get bounds of shape (n_dims, 2) 
	for min, max in each dimension
	'''
	flat_lines = []
	for line in lines:
		flat_lines += list(line)

	flat_lines = np.array(flat_lines)
	
	bounds = np.zeros((flat_lines.shape[1],2))
	for i in range(len(bounds)):
		bounds[i][0] = flat_lines[:,i].min()
		bounds[i][1] = flat_lines[:,i].max()
	return bounds

def normalize_lines(lines):
	''' normalize the points to lie in the range (0,1) '''
	# get maxes for each dimension
	mxs = [0,0,0]
	for line in lines:
		for p in line:
			for i in range(len(mxs)):
				if abs(p[i])>mxs[i]: mxs[i] = p[i]

	raise Exception('Function not Implemented')

def smooth(lines):
	''' 
	smooth the lines such that they use more points and have softer corners
	(bezier?)
	'''
	raise Exception('Function not Implemented')
	return lines

def dist(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))

def circle_points(x,y,r, n=50):
    step = (2*math.pi)/n
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

def line(p1,p2,n=20):
    ''' create a line out of n+2 points '''
    dx = (p2[0]-p1[0])/float(n)
    dy = (p2[1]-p1[1])/float(n)
    dz = (p2[2]-p1[2])/float(n)

    line = []
    for i in range(n):
        line += [[p1[0]+dx*i, p1[1]+dy*i, p1[2]+dz*i]]

    return line + [p2]

def unit_triangle(n=40):
	b2 = (1/3.)**.5
	a2 = 1/2. 
	vs = [[0,a2,0],[-b2,-a2,0],[b2,-a2,0]]
	lines = [line(v1,v2,n) for v1 in vs for v2 in vs if v1 != v2]
	return np.array(lines)

def unit_cube(x=0,y=0,z=0,n=30):
    vals = [-.5,.5]
    vs = [[e1+x,e2+y,e3+z] for e1 in vals for e2 in vals for e3 in vals]
    edges = [(1,2),(1,3),(4,3),(2,4),(5,6),(6,8),(7,8),(5,7)]
    edges += [(3,7),(4,8),(2,6),(1,5)]
    lines = []
    for e in edges:
        lines += [line(vs[e[0]-1],vs[e[1]-1],n)]
    return lines

def wireframe_to_lines(X,Y,Z):
    lines = []
    for i in range(len(X)):
        lines += [[[X[i,j],Y[i,j],Z[i,j]] for j in range(X.shape[1])]]
    for j in range(len(X)):
        lines += [[[X[i,j],Y[i,j],Z[i,j]] for i in range(X.shape[1])]]
    return lines

def remove_edge(adj_mat,edges,e):
	if e in edges:
		del edges[e]
	rev_e = tuple(list(e)[::-1])
	if rev_e in edges:
		del edges[rev_e]
	
	adj_mat[e[0],e[1]] = 0
	adj_mat[e[1],e[0]] = 0


def get_neighbours(adj_mat, v):
	return adj_mat[v].nonzero()[0]

def obj_to_lines(obj):
	vs = [None] + obj[0] # pad with none because v#s start at 1
	faces = obj[1]
	adj_mat = np.zeros((len(vs)+1,len(vs)+1))
	edges = {}
	for f in faces:
		f_edges = [(v1,v2) for v1 in f for v2 in f if v1 != v2]
		for e in f_edges:
			adj_mat[e[0],e[1]] = 1
			adj_mat[e[1],e[0]] = 1
			e = tuple(e)
			edges[e] = 1

	# traverse object and build a collection of lines that draw the obj
	lines = []
	while(edges):
		# build a line
		e0 = edges.keys()[0]
		remove_edge(adj_mat,edges, e0)
		v0 = e0[0]
		new_line = line(vs[v0],vs[e0[1]])
		last_v = e0[1] 
		while(edges):
			nvs = get_neighbours(adj_mat,last_v)
			if not len(nvs): break
			next_v = nvs[0]
			next_e = (last_v,next_v)
			remove_edge(adj_mat,edges, next_e)
			new_line += line(vs[last_v], vs[next_v])
			last_v = next_v
		lines += [new_line]
	return lines

def perlin_sphere(rs=1, n=50, x=0, y=0, z=0, s=1, ns=0.04, o=3):
    sphere = gen_sphere(rs, n, x, y, z)
    rs = float(rs)
    for line in sphere:
        for p in line:
            px = x-p[0]
            py = y-p[1]
            pz = z-p[2]
            nse = noise.pnoise3(px/(rs*s),py/(rs*s), pz/(rs*s), octaves=o)
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
    T: transformation matrix (4x4)
    p3d: [x,y,z]
    '''
    p3d = np.concatenate([p3d,[1]])
    p3d = T * np.matrix(p3d).T
    p3d = np.array(p3d.T)[0]
    p3d = (p3d/(p3d[-1]+1e-10))[:3]
    return p3d

def rot_x(th):
    c = math.cos(th)
    s = math.sin(th)
    Rx = np.array([ [1, 0, 0, 0],
                    [0, c,-s, 0],
                    [0, s, c, 0],
                    [0, 0, 0, 1]])
    return Rx

def rot_y(th):
    c = math.cos(th)
    s = math.sin(th)
    Ry = np.array([ [ c, 0, s, 0],
                    [ 0, 1, 0, 0],
                    [-s, 0, c, 0],
                    [ 0, 0, 0, 1]])
    return Ry

def rot_z(th):
    c = math.cos(th)
    s = math.sin(th)
    Rz = np.array([ [ c,-s, 0, 0],
                    [ s, c, 0, 0],
                    [ 0, 0, 1, 0],
                    [ 0, 0, 0, 1]])
    return Rz

def project_lines(lines_3d, P=None, f=1 ):
    if not P:
        P = np.matrix([[f, 0, 0, 0],
                    [0, f, 0, 0],
                    [0, 0, 1, 0]])

    lines_2d = []
    for line_3d in lines_3d:
        line_2d = []
        for p3d in line_3d:
            p2d = project_p3d(P,p3d)
            line_2d += [p2d]
        lines_2d += [line_2d]
    return np.array(lines_2d)

def rotate_lines(lines,th_x=0, th_y=0, th_z=0):
    '''
    rotations in radians
    '''
    rotated_lines = []
    for line in lines:
        rot_line = []
        for p in line:
            p = transform_p3d(rot_x(th_x),p)
            p = transform_p3d(rot_y(th_y),p)
            p = transform_p3d(rot_z(th_z),p)
            rot_line += [p]
        rotated_lines += [rot_line]
    return np.array(rotated_lines)


def translate_lines(lines,dx=0,dy=0,dz=0):
    '''
    assume lines are 3d
    '''
    dp = np.array([dx,dy,dz])
    lines = [np.array(l) for l in lines]
    for line in lines:
        line += dp
    return lines

def scale_lines(lines,sx=1,sy=1,sz=1):
    sp = np.array([sx,sy,sz])
    lines = np.array(lines)
    for line in lines:
        line *= sp
    return lines

def apply_perlin2d(lines,s=1,ns=0.5,o=4):
    mx = 0
    my = 0
    c = [0,0]
    for line in lines:
        for p in line:
            if abs(p[0])>mx: mx = p[0]
            if abs(p[1])>my: my = p[1]

    lines = np.array(lines)
    for line in lines:
        for p in line:
            px = c[0]-p[0]
            py = c[1]-p[1]
            nse = noise.pnoise2(px/(mx*s),py/(my*s), octaves=o)
            p[0] += (c[0]-p[0])*nse*ns
            p[1] += (c[1]-p[1])*nse*ns

    return lines

def apply_perlin(lines,s=1, ns=0.5, o=2):
    ''' assume lines are centered at 0
    (or else we'd have to find the center)
    '''
    mx = 0
    my = 0
    mz = 0
    c = [0,0,0]
    for line in lines:
        for p in line:
            if abs(p[0])>mx: mx = p[0]
            if abs(p[1])>my: my = p[1]
            if abs(p[2])>mz: mz = p[2]

    lines = np.array(lines)
    for line in lines:
        for p in line:
            px = c[0]-p[0]
            py = c[1]-p[1]
            pz = c[2]-p[2]
            eps = 1e-10
            nse = noise.pnoise3(px/(mx*s+eps),py/(my*s+eps), pz/(mz*s+eps), octaves=o)
            p[0] += (c[0]-p[0])*nse*ns
            p[1] += (c[1]-p[1])*nse*ns
            p[2] += (c[2]-p[2])*nse*ns
    return lines

def gen_sphere(rs=1, n=50, x=0, y=0, z=0, cp=200):
    '''
    return lines that compose a sphere in 3d
    cp: the number of points per circle
    '''
    lines = []
    n = float(n)
    #rs = 1 # radius sphere
    for i in range(int(n)):
        # radius inscribed circle
        a = abs(1 - i/(n/2.))*rs # distance along sphere axis
        rc = math.sqrt(rs**2 - a**2)
        c2d = circle_points(x,y,rc, cp)
        # circle z coord
        cz = i/(n/2.)*rs+(z-(rs/2.))
        c3d = np.array([list(p)+[cz] for p in c2d])
        lines += [c3d]
    return np.array(lines)

if __name__ == '__main__':
	from obj import *
	objs = load_obj('data/obj/sphere.obj')
	oid = objs.keys()[0]
	lines = obj_to_lines(objs[oid])
