import random
from numpy import array as nparray
from numpy import reshape, diagflat, trace, transpose, zeros
from numpy.random import rand as nprand
from numpy.linalg import qr

RANDOM_RANGE_MIN = -5
RANDOM_RANGE_MAX = 5
PERTURBATION_RANGE_MIN = -1e-1
PERTURBATION_RANGE_MAX = 1e-1

class MatrixGenerator:
	def __init__(self, only_int, only_diag, p_only_diag=False, q_only_diag=False):
		self.only_int = only_int
		
		if only_diag:
			self.p_only_diag = True
			self.q_only_diag = True
		
		self.p_only_diag = p_only_diag
		self.q_only_diag = q_only_diag

	def get_random_matrix(self, dim, perturb=False, dynamic_range=None, target_trace=None, zero=False, type="", symmetric=False):
		if zero:
			return zeros((dim, dim))
			
		if perturb:
			if self.only_int:
				range_min = -1
				range_max = 1

			else:
				if dynamic_range is not None:
					range_min = -10**-dynamic_range
					range_max = -range_min
				else:
					global PERTURBATION_RANGE_MIN, PERTURBATION_RANGE_MAX
	
					range_min = PERTURBATION_RANGE_MIN
					range_max = PERTURBATION_RANGE_MAX
		else:
			global RANDOM_RANGE_MIN, RANDOM_RANGE_MAX
		
			range_min = RANDOM_RANGE_MIN
			range_max = RANDOM_RANGE_MAX
		
		if (self.p_only_diag and type == "p") or (self.q_only_diag and type == "q") or (self.p_only_diag and self.q_only_diag):
			B = get_random_square_matrix(dim, range_min, range_max, only_int=self.only_int, diag=True)
		else:
			B = get_random_square_matrix(dim, range_min, range_max, only_int=self.only_int, symmetric=symmetric)

		if target_trace is not None:
			tr = trace(B)
			tr_min = tr / dim
		
			for i in range(dim):
				B[i][i] -= tr_min
		
		return B
		
def get_random_orthogonal_matrix(dim):
	A = nprand(dim, dim)
	Q, R = qr(A)
	
	return Q

def get_next_zero_one_perturbation(p):
	np = list()

	incremented = False

	for entry in p:
		if incremented:
			np.append(entry)
		else:
			if entry == 1:
				np.append(-1)
			else:
				np.append(entry + 1)
				incremented = True
	
	if not incremented:
		return None
	else:
		return np


def get_matrices(p, dim, k):
	ind = 0

	pi = list()
	for i in range(k):
		pi.append(nparray(p[ind:ind+dim**2]).reshape(dim, dim))
		ind = ind + dim**2

	qi = list()
	for i in range(k + 1):
		pi.append(nparray(p[ind:ind+dim**2]).reshape(dim, dim))
		ind = ind + dim**2

	return pi, qi

def zero_one_perturbations(dim, k):
	nentries = (2*k + 1) * dim**2

	per = [-1 for i in range(nentries)]

	while per is not None:
		yield (get_matrices(per, dim, k))
		per = get_next_zero_one_perturbation(per)
		
def get_random_square_matrix(dim, range_min, range_max, only_int=False, diag=False, symmetric=False):
	if only_int:
		rand_func = random.randint
	else:
		rand_func = random.uniform
		
	if diag:
		return diagflat([rand_func(range_min,range_max) for i in range(dim)])
	else:
		B = nparray([rand_func(range_min,range_max) for i in range(dim*dim)]).reshape(dim,dim)
		
		if symmetric:
			B = symmetrize(B)
			
		return B

def symmetrize(mat):
	return 0.5 * (mat + mat.T)
