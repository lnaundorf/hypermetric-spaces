from hypermetric import read_argv, MultithreadedTest
from numpy.linalg import svd
from numpy import array as nparray
from numpy import dot

def mat_isometry(mat):
	U, svals, V = svd(mat)

	svals_new = [[svals[0] + svals[1], 0], [0, svals[0] - svals[1]]]
	ret = dot(U, svals_new)
	ret = dot(ret, V)

	print "transform:"
	print mat
	print "---->"
	print ret
	print ""

	return ret

def ft(mtt):
	mtt.alter_testset(mat_isometry)
	mtt.he.set_norm_config("operator")


[dim, k, norm] = read_argv() 
if dim == -1:
	exit(0)

target_trace = 0.0

mtt = MultithreadedTest(dim, k, norm)#, final_transform=ft)#, target_trace=target_trace)
mtt.run()


