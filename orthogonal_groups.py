import math
from numpy import array as nparray, dot

class OrthogonalGroup:
	def __init__(self, dim, steps, special=False):
		self.dim = dim
		self.steps = steps
		self.pos = True
		self.special = special
		
		if dim == 2:
			self.step = 0

		if dim == 3:
			self.angles = [0.0, 0.0, 0.0]

	def __iter__(self):
		return self

	def next(self):
		try:
			if self.dim == 2:
				ret = self.next_two()
			elif self.dim == 3:
				ret = self.next_three()
		except Exception:
			raise StopIteration()

		return ret


	"""
	For a parametrizatio of O(2), see
	http://mathworld.wolfram.com/OrthogonalGroup.html
	"""
	def next_two(self):
		if self.step == self.steps:
			self.step = 0
			raise Exception()

		phi = 2 * math.pi * self.step / self.steps

		if self.pos:
			ret = nparray([[math.cos(phi),-math.sin(phi)],[math.sin(phi),math.cos(phi)]])

			if not self.special:
				self.pos = False
		else:
			ret = nparray([[-math.cos(phi),math.sin(phi)],[math.sin(phi),math.cos(phi)]])

		if self.special or not self.pos:
			self.step += 1

		return ret

	def reset_iteration(self):
		if self.dim == 2:
			self.step = 0
		else:
			raise Exception()

"""
	def O_three():
		for i in range(steps):
			alpha = i * 2 * math.pi / steps
			Ralpha = nparray([[1,0,0],[0,math.cos(alpha),-math.sin(alpha)],[0,math.sin(alpha),math.cos(alpha)]])


			for j in range(steps):
				beta = j * 2 * math.pi / steps
				Rbeta = nparray([[math.cos(beta),0,math.sin(beta)],[0,1,0],[-math.sin(beta),0,math.cos(beta)]])

				for k in range(steps):
					gamma = k * 2 * math.pi / steps
					Rgamma = nparray([[math.cos(gamma),-math.sin(gamma),0],[math.sin(gamma),math.cos(gamma),0],[0,0,1]])

					A = dot(dot(Ralpha,Rbeta),Rgamma)
				
					for mat in [A, -A]:
						yield mat
"""
