from geometry_primitives import *
import random


'''
ok, so this is an alg to grow lines in a sort of semi random walk
(maybe w momentum?) 
and uses a collision box to check for collisions
'''

def hit_2d(cb,v,r):
	''' tracking collisions in the collision box is much more efficient '''
	s = cb.shape[0]
	yl = max(v[1]-r,0)
	yu = min(v[1]+r+1,s)
	xl = max(v[0]-r,0)
	xu = min(v[0]+r+1,s)
	cb[yl:yu,xl:xu] = 1	
	cb[v[1],v[0]] = 2

def potential_neighbours_2d(collision_box,v,r):
	nr = r + 1
	pn = []
	s = len(collision_box)
	for x in range(max(v[0]-nr,0),min(v[0]+nr+1,s)):
		for y in range(max(v[1]-nr,0),min(v[1]+nr+1,s)):
			nv = [x,y]
			if not collision_box[y,x]:
				pn += [nv]
	return pn

def mid(line):
	line = np.array(line)
	return np.array([line[:,0].mean(),line[:,1].mean()])

def choose_next_2d(line,pn):
	''' 
	choose next v out of pn 
	balance of forces:
		stay away from last vertex, 
		move towards first vertex
	'''
	best_v = None
	best_F = float('-inf')

	for v in pn:
		Ff = max(1-len(line)/40.,0) * dist(mid(line),v)
		Fl = ((dist(line[-1],v)-0.4)*0.2)
		Fi = dist(line[0],v)/10.
		Fr = (random.random()-.5)*0.03
		F = Ff+Fi+Fl
		if F > best_F:
			best_v = v
			best_F = F
	return best_v

def is_end(cb,v,r):
	r = r+1
	s = cb.shape[0]
	yl = max(v[1]-r,0)
	yu = min(v[1]+r+1,s)
	xl = max(v[0]-r,0)
	xu = min(v[0]+r+1,s)
	return (cb[yl:yu,xl:xu]==2).sum() >= 3
	
def get_vertices(lines):
	vertices = []
	for line in lines:
		vertices += line
	return vertices

def grow_line_2d(lines,collision_box,r):
	cb = collision_box
	# choose starting vertex
	vs = None
	vertices = get_vertices(lines)
	random.shuffle(vertices)
	while not vs and vertices:
		# choose_starting neighbour vertex
		start_vn = vertices.pop() 
		pn = potential_neighbours_2d(collision_box, start_vn, r)
		if pn: vs = random.choice(pn)

	if vs:
		line = [start_vn,vs]
		hit_2d(cb,vs,r)
		# choose vertices to grow line
		while potential_neighbours_2d(cb,line[-1], r):
			pn = potential_neighbours_2d(cb,line[-1], r)
			next_v = choose_next_2d(line,pn)
			hit_2d(cb,next_v,r)
			line += [next_v]
			if is_end(cb,line[-1],r): break

		lines += [line]
	else:
		print('no line added')
	return lines,collision_box

def grow_lines_2d(n=10,s=100,r=1):
	collision_box = np.zeros((s,s))
	mid = int(s/2.)

	# start with a single point
	vs = [mid,mid] 
	lines = [[vs]]
	hit_2d(collision_box,vs,r)
	for i in range(n):
		lines,collision_box = grow_line_2d(lines,collision_box,r)

	# scale points in lines
	scale = 1. / s
	scaled_lines = []
	for line in lines:
		line = np.array(line)	
		line = line.astype('float64')
		line *= scale
		scaled_lines += [line]	
	return scaled_lines

if __name__ == '__main__':
	grown = grow_lines_2d(n=1)
	r = 1
	cb = np.zeros((10,10))
	lines = [[[5,5],[7,7]]]
	line = lines[0]
	collision_box = cb
	hit_2d(cb,[5,5],1)
	hit_2d(cb,[7,7],1)
	# normalize points to have max val of 1
