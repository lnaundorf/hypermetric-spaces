from numpy.linalg import eigvals, eigvalsh, eigh, svd
from math import sqrt

def norm_abs_diff_eig(mat, symmetric=True, diag=False):
	n = len(mat)

	M = [-(n-1) + 2*i for i in range(n)]

	if diag:
		eigvals = sorted(diag(mat))
	else:
		if symmetric:
			eigvals = eigvalsh(mat)
		else:
			eigvals = eigvals(mat)

	ret = 0		
	for k in range(n):
		ret += M[k]*eigvals[k]
	
	return ret

def norm_abs_eig(mat, symmetric=True, diag=False):
	if diag:
		eigvals = diag(mat)
	else:
		if symmetric:
			eigvals = eigvalsh(mat)
		else:
			eigvals = eigvals(mat)

	return sum_abs_val(eigvals)

	
def norm_nuclear(mat):
	U, svals, V = svd(mat)
	
	#print "Singular values: %s" % str(svals)
	
	return sum_abs_val(svals)

def norm_operator(mat):
	U, svals, V = svd(mat)

	return svals[0]
	
def norm_euclidean(mat):
	ret = 0.0
			
	for row in mat:
		for entry in row:
			ret += entry**2
			
	return sqrt(ret)

def get_norm_config(norm_string):
	if norm_string == "abs":
		norm = norm_abs_eig
		symmetric = True
		printstring = "Eigen"
	elif norm_string == "diff":
		norm = norm_abs_diff_eig
		symmetric = True
		printstring = "Eigen"
	elif norm_string == "nuclear":
		norm = norm_nuclear
		symmetric = False
		printstring = "Singular"
	elif norm_string == "operator":
		norm = norm_operator
		symmetric = False
		printstring = "Singular"
	elif norm_string == "euclidean":
		norm = norm_euclidean
		symmetric = False
		printstring = "Euclidean"
	else:
		return -1

	return [norm, symmetric, printstring]
	
	
""" Returns the sum of the absolute values of the entries of an array """
def sum_abs_val(arr):
	return reduce(lambda x,y: abs(x) + abs(y), arr)
