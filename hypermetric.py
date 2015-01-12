from matrix_generation import MatrixGenerator
from numpy import diag, dot, identity
from operator import add
import threading
from multiprocessing import cpu_count
import time
import sys
from numpy.linalg import eigh, svd
import matrix_norms

def read_argv():
	if len(sys.argv) < 3:
		print "Give dimension and k as argument"
		return -1,-1
	else:
		try:
			dim = int(sys.argv[1])
			k = int(sys.argv[2])
			
			return dim, k
		except:
			print "The param %s or %s is not an integer" % (sys.argv[1], sys.argv[2])
			return -1,-1

def get_vector_print(v, eigval):
	ret = "("

	for entry in v:
		ret += "%s," % str(entry*eigval)
	
	ret = ret[:-1]
	ret += ")"
	return ret

def get_matrix_print(mat):
	ret = "["
	for row in mat:
		for entry in row:
			ret += "%s," % str(entry)
		ret = ret[:-1]
		ret += ";"
	ret = ret[:-1]
	ret += "]"

	return ret
	
def group_action(g, mat):
	return dot(dot(g, mat), g.T)
	
	
class MultithreadedTest:
	def __init__(self, dim, k, norm, target_trace=None, final_transform=None):
		self.k = k
		self.min_diff = 100000
		self.target_trace = target_trace
		
		self.stop = False
		self.lock = threading.Lock()
		
		#generate starting random testset
		self.he = HypermetricEnvironment(dim, norm=norm)
		self.base_pi, self.base_qi = self.he.generate_testset(self.k, target_trace=self.target_trace)
		
		self.last_improvement = 0
		self.final_transform = final_transform
		
	def k_hypermetric_test(self):
		global min_diff, last_improvement, k, he
		
		while not self.stop:
			if self.last_improvement == 0:
				timediff = 1
			else:
				timediff = time.time() - self.last_improvement
				
			if timediff < 1.0:
				timediff = 1.0
				#print "diff = 1"
			elif timediff >= 5.:
				timediff = 5.0
				#print "diff = 5"

			pi_perturb, qi_perturb = self.he.generate_testset(self.k, perturb=True, dynamic_range=timediff, target_trace=0.0)

			pi = map(add, self.base_pi, pi_perturb)
			qi = map(add, self.base_qi, qi_perturb)
		
			lhs_1, lhs_2 = self.he.compute_lhs(pi, qi)
			lhs = lhs_1 + lhs_2
			rhs = self.he.compute_rhs(pi, qi)
			
			self.lock.acquire()

			diff = rhs - lhs

			if diff < self.min_diff and not self.stop:
				self.min_diff = diff
				self.base_pi, self.base_qi = pi, qi
				
				self.last_improvement = time.time()

				print "found better diff: %.4f - %.4f - %.4f = %.4f - %.4f = %.4f, timediff = %.2f" % (rhs, lhs_1, lhs_2, rhs, lhs, diff, timediff)
			self.lock.release()
			
		
	def run(self):
		for i in range(cpu_count()):
			#print "Now starting thread %d" % i
			t = threading.Thread(target=self.k_hypermetric_test)
			t.start()

		while True and not self.stop:
			try:
				time.sleep(1)

			except KeyboardInterrupt:
				self.stop = True
				time.sleep(0.1)
				#self.he.print_infos(self.base_pi, self.base_qi)
				if self.final_transform is not None:
					self.final_transform(self)
				self.he.print_infos(self.base_pi, self.base_qi)
				exit(0)

	def stop(self):
		self.stop = True

	def alter_testset(self, fun):
		new_pi = list()

		for pi in self.base_pi:
			new_pi.append(fun(pi))

		new_qi = list()

		for qi in self.base_qi:
			new_qi.append(fun(qi))

		[self.base_pi, self.base_qi] = [new_pi, new_qi]


class HypermetricEnvironment:
	def __init__(self, dim, norm="diff", only_int=False, only_diag=False, p_only_diag=False, q_only_diag=False, p_const_orth=None, q_const_orth=None):
		self.dim = dim
		
		self.set_norm_config(norm)
		
		self.only_int = only_int
		
		if only_diag:
			self.p_only_diag = True
			self.q_only_diag = True
			
		self.p_only_diag = p_only_diag
		self.q_only_diag = q_only_diag
		
		self.mg = MatrixGenerator(only_int, only_diag)
		
		self.p_const_orth = p_const_orth
		self.q_const_orth = q_const_orth

	def set_norm_config(self, normstring):
		[self.norm, self.symmetric, self.printstring] = matrix_norms.get_norm_config(normstring)
		
	def const_orth(self, mat, type):
		if type == "p":
			if self.p_const_orth is not None:
				return group_action(self.p_const_orth, mat)
			else:
				return mat	
		elif type == "q":
			if self.q_const_orth is not None:
				return group_action(self.q_const_orth, mat)
			else:
				return mat
		else:
			return None



	def generate_zero_testset(self, k):
		return self.generate_testset(dim, k, zero=True)		

	def generate_testset(self, k, perturb=False, dynamic_range=None, zero=False, target_trace=None):
		dim = self.dim
		
		pi = list()
		for i in range(k):
			pi.append(self.mg.get_random_matrix(dim, perturb, dynamic_range, zero=zero, type="p", target_trace=target_trace, symmetric=self.symmetric))

		qi = list()
		for i in range(k + 1):
			qi.append(self.mg.get_random_matrix(dim, perturb, dynamic_range, zero=zero, type="q", target_trace=target_trace, symmetric=self.symmetric))
		
		return pi, qi


	def compute_lhs(self, pi, qi):
		k = len(pi)

		ret1 = 0.0
		for i in range(k):
			for j in range(i):
				ret1 += self.norm(self.const_orth(pi[i] - pi[j], "p"))

		ret2 = 0.0
		for i in range(k+1):
			for j in range (i):
				ret2 += self.norm(self.const_orth(qi[i] - qi[j], "q"))

		return ret1, ret2


	def compute_rhs(self, pi, qi):
		k = len(pi)
		ret = 0.0

		for i in range(k):
			for j in range(k+1):
				ret += self.norm(self.const_orth(pi[i], "p") - self.const_orth(qi[j], "q"))

		return ret

	def compute_diff(self, pi, qi):
		[lhs1, lhs2] = self.compute_lhs(pi, qi)
		return self.compute_rhs(pi, qi) - (lhs1 + lhs2)
		
	def print_diff(self, mat):
		print mat
	
		no = self.norm(mat)
		if self.symmetric:
			eigvals, eigvectors = eigh(mat)
		else:
			U, eigvals, V = svd(mat)
			
		print "%s vals: %s -> norm = %.2f" % (self.printstring, str(eigvals),no)
		
		return no


	def print_infos(self, pi, qi):
		print "Diff = %.4f" % self.compute_diff(pi, qi)
		pi = map(self.const_orth, pi, ["p" for p in pi])
		qi = map(self.const_orth, qi, ["q" for q in qi])
		
		print "pi:"
		for i in range(len(pi)):
			print "P_%d = %s" % (i, get_matrix_print(pi[i]))

		print "qi:"
		for i in range(len(qi)):
			print "Q_%d = %s" % (i, get_matrix_print(qi[i]))

		k = len(pi)
	
		for i in range(k):
			if self.symmetric:
				print "%s vectors p_%d:" % (self.printstring, i)
				eigvals, eigvectors = eigh(pi[i])
				for j in range(len(eigvectors[:])):
					print "p%d_%d = %s" % (i, j, get_vector_print(eigvectors.T[:][j], eigvals[j]))
			else:
				U, eigvals, V = svd(pi[i])
				
			print "p_%d %s vals: %s" % (i, self.printstring, str(eigvals))
		
		for i in range(k+1):
			if self.symmetric:
				print "%s vectors q_%d:" % (self.printstring, i)
				eigvals, eigvectors = eigh(qi[i])
				for j in range(len(eigvectors[:])):
					print "q%d_%d = %s" % (i, j, get_vector_print(eigvectors.T[:][j], eigvals[j]))
			else:
				U, eigvals, V = svd(qi[i])
				
			print "q_%d %s vals: %s" % (i, self.printstring, str(eigvals))

		norm_sum = 0.0
		print "\npi metrics:"
		for i in range(k):
			for j in range(i):
				mat = pi[i]-pi[j]
				print "p_%d - p_%d:" % (i, j)
				no = self.print_diff(mat)
				norm_sum += no
				
		print "metric sum = %.2f" % norm_sum

		norm_sum = 0.0
		print "\nqi metrics:"
		for i in range(k+1):
			for j in range(i):
				mat = qi[i]-qi[j]
				print "q_%d - q_%d:" % (i, j)
				no = self.print_diff(mat)
				norm_sum += no

		print "metric sum = %.2f" % norm_sum

		norm_sum = 0.0
		print "\nmixed metrics:"
		for i in range(k):
			for j in range(k+1):
				mat = pi[i]-qi[j]
				print "p_%d - q_%d:" % (i, j)
				no = self.print_diff(mat)
				norm_sum += no

		print "metric sum = %.2f" % norm_sum
