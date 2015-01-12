from orthogonal_groups import OrthogonalGroup
from group_actions import orthogonal_symmetric_group_action as osga
from numpy import array as nparray
from math import fabs

def scalar_prod(mat1, mat2):
	ret = 0.0

	for (e1, e2) in zip(mat1, mat2):
		ret += e1 * e2
			
	return ret


class ConvexBody:
	def h(self, B):
		maxval = float("-inf")
	
		for A in self:
			sp = scalar_prod(A, B)
		
			if sp > maxval:
				maxval = sp
				#print "maxval = " + str(maxval)

		return maxval

class Zonotope(ConvexBody):
	def __init__(self, Z, steps):
		self.Z = Z
		self.steps = steps
		self.factors = [1 for i in range(len(self.Z))]

	def h(self, B):
		ret = 0.0

		for zi in self.Z:
			retpart = 0.0
			for (a, b) in zip(zi, B):
				retpart += a * b

			ret += fabs(retpart)

		return ret

	def __iter__(self):
		return self

	def next(self):
		decremented = False

		for i in range(len(self.factors)):
			if not decremented:
				if self.factors[i] > -1 + 1e-2:
					self.factors[i] = self.factors[i] - 2.0 / self.steps
					decremented = True
				else:
					self.factors[i] = 1

					if i == len(self.factors) -1:
						self.factors = [1 for i in range(len(self.Z))]
						raise StopIteration()
	
		return self.get_zonotope_point()

	def reset_iteration(self):
		self.factors = [1 for i in range(len(self.Z))]


#	def zonotope_vertices(Z):
#		factors = [1 for i in range(len(Z))]
#		ended = False
#
#		while not ended:
#			decremented = False
#
#			for i in range(len(factors)):
#				if not decremented:
#					if factors[i] == 1:
#						factors[i] = -1
#						decremented = True
#					else:
#						factors[i] = 1
#
#						if i == len(factors) -1:
#							ended = True
#		
#			return self.get_zonotope_point()


	def get_zonotope_point(self):
		z = [0 for i in range(len(self.Z[0]))]

		for (zi, f) in zip(self.Z, self.factors):
			for i in range(len(zi)):
				z[i] = z[i] + zi[i] * f

		return z

class OrthogonalSymmetricOrbit(ConvexBody):
	def __init__(self, dim, zonotope, steps):
		self.zonotope = zonotope
		self.group = OrthogonalGroup(dim, steps)
		self.z = self.zonotope.next()
		self.g = self.group.next()

	def __iter__(self):
		return self

	def next(self):
		ga = osga(self.g, [[self.z[0], 0], [0, self.z[1]]])
		ret = nparray([ga[0][0], ga[0][1], ga[1][1]])
		try:
			self.z = self.zonotope.next()
		except StopIteration:
			try:
				self.g = self.group.next()
			except StopIteration:
				raise StopIteration()

		return ret

	def reset_iteration(self):
		self.zonotope.reset_iteration()
		self.z = self.zonotope.next()

		self.group.reset_iteration()
		self.g = self.group.next()

	def h(self, B):
		maxval = float("-inf")
	
		for A in self:
			sp = scalar_prod(A, B)
		
			if sp > maxval:
				maxval = sp
				self.maxg = self.g

		return maxval

	def h_fast(self, B):
		return self.zonotope.h(eigvalsh(B))
