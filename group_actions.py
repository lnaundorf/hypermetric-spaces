from numpy import dot, transpose

def orthogonal_symmetric_group_action(g,a):
	gt = transpose(g)
	ret = dot(g,a)
	ret = dot(ret,gt)
	
	return ret

def orthogonal_group_action(g,a):
	return dot(g,a)
