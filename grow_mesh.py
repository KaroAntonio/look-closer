from geometry_primitives import *
import random

def is_frontier(V,E,v):
	''' 
	check if v is a fronteir vertex 
	aka check if there is any space for it to be grown from anymore
	'''
	pass


def add_vertex_slow(V,E,F):
	''' add a vertex and approp edges '''

	# choose frontier vertex
	while F:
		v = F.pop()
		if is_frontier(V,E,v):
			


	

def grow_mesh(n=100, add_vertex=add_vertex_slow):
	''' return the vertices and edges of a grown mesh '''

	# start with a single triangle
	V = [[0,0,0),(1,1,0),(1,0,0)]
	E = [(0,1),(1,2),(2,0)]
	F = [0,1,2] # frontier (active) vertices

	for i in range(n):
		random.shuffle(F)
		V,E,F = add_vertex(V,E,F)
	
	
	

	

	
