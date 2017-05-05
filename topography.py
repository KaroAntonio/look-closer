from obj import *
from geometry_primitives import *
import itertools

''' 
given an input mesh (edges and vertices)
return the topographical lines of that mesh along a given axis = {x,y,z}
'''

def get_max_v(fs):
	return max([max(e) for e in fs])

def process_faces(vs,fs,axis):
	''' build edges with euclidean vertices '''
	es = []
	vs = [None]+vs # pad bc indxs start at 1
	n_e = len(fs) * 5 # conservative number of edges 
	face_mat = np.zeros((n_e,n_e))
	n_v = get_max_v(fs)+1
	edge_idx_mat = np.zeros((n_v,n_v))-1 # mat to track the idx of edges
	for f in fs:
		e_idxs = []
		for v1,v2 in itertools.combinations(f, 2):
			if edge_idx_mat[v1][v2] == -1:
				e = [vs[v1],vs[v2]]
				e.sort(key=lambda x:x[axis])
				es += [e] 
				idx = len(es)-1
				edge_idx_mat[v1][v2] = idx
				edge_idx_mat[v2][v1] = idx 
				e_idxs += [idx]
			else:
				e_idxs += [int(edge_idx_mat[v1][v2])]
		# track which edges share a face ~ by edge index
		for e1,e2 in itertools.combinations(e_idxs, 2):
			face_mat[e1][e2] = 1
			face_mat[e2][e1] = 1

	return es, face_mat, edge_idx_mat

def edge_in_lvl(edge, lvl, ax):
	return edge[0][ax] <= lvl and edge[1][ax] >= lvl

def get_edges_at_lvl(s_idx, es, last_idxs, lvl, ax):
	# keep all edges from the last level 
	lvl_es = [ei for ei in last_idxs if edge_in_lvl(es[ei], lvl, ax)]
	# keep going till the the edge isnt in the lvl anymore
	while s_idx < len(es) and es[s_idx][0][ax] < lvl:
		if edge_in_lvl(es[s_idx], lvl, ax): 
			lvl_es += [s_idx]
		s_idx += 1 
	return lvl_es

def lvl_intersect(edge, lvl, ax):
	''' return the point where the lvl intersects the edge at ax'''
	v1,v2 = np.array(edge)
	dv = v2-v1
	u = dv / np.linalg.norm(dv)
	s = lvl-v1[ax]
	try:
		m = s / (u[ax]+1e-10)
	except:
		print(s, u[ax])
	return u * m  + v1

def add_edge_to_contour(es, lvl, ax, lvl_idxs, line, line_idxs, e, d):
	v = lvl_intersect(es[e], lvl, ax)
	if d: line += [v]
	else: line.insert(0,v)
	line_idxs[e] = 0
	del lvl_idxs[e]

def build_contour_line(f_mat, es, lvl_idxs, lvl, ax):
	# Build One Contour Line
	
	line = []
	all_lvl_idxs = copy.copy(lvl_idxs)
	start_idx = list(lvl_idxs)[0]
	line_idxs = {}
	add_edge_to_contour(es,lvl,ax,lvl_idxs,line,line_idxs,start_idx,1)
	last_edges = {}
	# choose a random first edge
	# build the line in each direction
	for d in [1,0]:
		curr = start_idx 
		while lvl_idxs:
			# get es that share the face that are not the curr edge
			nbrs = np.nonzero(f_mat[curr])[0]
			fes = [ e for e in nbrs if e in lvl_idxs and e != curr]

			if not fes: break

			# if there are no repeats, choose the next edge
			curr = fes.pop()
			add_edge_to_contour(es,lvl,ax,lvl_idxs,line,line_idxs,curr,d)
		last_edges[d] = curr

	# if the line is a loop, close the loop
	nbrs = np.nonzero(f_mat[last_edges[0]])[0]
	if last_edges[1] in nbrs:
		for d in [0,1]:
			e = last_edges[1-d]
			lvl_idxs[e]=0
			add_edge_to_contour(es,lvl,ax,lvl_idxs,line,line_idxs,e,d)

	return line

def build_contour_lines(f_mat, es, lvl_idxs, lvl, ax):
	'''
	given a set of edges that occur at a lvl, 
	lvl_idxs : a list of the idxs at this lvl
	'''
	lines = []
	lvl_idxs = { e:0 for e in lvl_idxs }	

	while lvl_idxs:
		lines += [build_contour_line(f_mat, es, lvl_idxs, lvl, ax) ]
			
	return lines

def sort_preserving_indices(es,f_mat,ax):
	''' sort edges and update the idxs in f_mat, edge_idx_mat '''

	idx_es = [[es[i],i] for i in range(len(es))]

	# sort all edges along axis by first v in edge
	idx_es.sort(key = lambda e: e[0][0][ax])

	old_new_idxs = {idx_es[i][1]:i for i in range(len(idx_es))}
	es = [e[0] for e in idx_es]
	
	n_e = f_mat.shape[0]
	new_f_mat = np.zeros((n_e,n_e))
	for v1 in range(len(f_mat)):
		nbrs = np.nonzero(f_mat[v1])[0]
		for v2 in nbrs:
			v1_ = old_new_idxs[v1]
			v2_ = old_new_idxs[v2]
			new_f_mat[v1_,v2_] = 1
			new_f_mat[v2_,v1_] = 1

	return es,new_f_mat

def build_topography(vs, fs, n=200, ax=2):
	'''
	vs: vertices
	fs: faces where the vertices (assume triangular) are indexed as {1,...,i}
	n: number of contour lines of topography
	ax: {0:x,1:y,2:z}
	'''
	# build edges s.t. the first vertex is ordered lower along the axis in question
	es,f_mat,edge_idx_mat = process_faces(vs,fs,ax)
	es,f_mat = sort_preserving_indices(es,f_mat,ax)
	

	contour_lines = []
	bounds = get_bounds(es)
	ax_min = bounds[ax][0]
	ax_max = bounds[ax][1]
	s_idx = 0 # the index of the first line with a beginning vertex < z
	# keep track of the indices of all edges that fit the last lvl
	last_idxs = list(range(len(es))) # for now use all indexes as the indexes to use
	for i in range(n):
		lvl = (i+1)/(n+1.) * (ax_max - ax_min) + ax_min 
		
		lvl_idxs = get_edges_at_lvl(s_idx, es, last_idxs, lvl, ax)
		print('contour lvl # {}'.format(i))
		#last_idxs = lvl_idxs[:]
		s_idx += len(lvl_idxs)
	
		# build contour lines
		contour_lines += build_contour_lines(f_mat, es, lvl_idxs, lvl, ax)
	
	return contour_lines
		
if __name__ == '__main__':
	objs = load_obj('data/obj/tree.obj')
	#objs = load_obj('data/obj/tetrahedron.obj')
	oid = objs.keys()[0]
	vs,fs,vns = objs[oid]
	ax = 1
	es,f_mat,edge_idx_mat = process_faces(vs,fs,ax)
	es.sort(key = lambda e: e[0][ax])
	bounds = get_bounds(es)
	ax_min = bounds[ax][0]
	ax_max = bounds[ax][1]
	s_idx = 0
	i=3
	n=9
	lvl = (i+1)/(n+1.) * (ax_max - ax_min) + ax_min
	last_idxs = list(range(len(es)))
	lvl_idxs = get_edges_at_lvl(s_idx, es, last_idxs, lvl, ax)
	contour_lines = build_topography(vs, fs, n=n)
