
import numpy as np 

class Space:

	def __init__(self):
		pass 

	def sample(self):
		exit("sample needs to be overwritten")

	def contains(self):
		exit("contains needs to be overwritten")


class Cube(Space): 

	def __init__(self,lims):
		# input:
		# 	- lims, ndarray, [dim x 2] 

		super(Cube, self).__init__()
		self.dim = lims.shape[0]
		self.lims = lims
		self.df_lim = 10 

	def sample(self):
		# output: 
		# 	- nd array [dim x 1]
		s = np.zeros((self.dim,1))
		for i in range(self.dim):
			if np.any(self.lims[i,:] > 1000):
				s[i] = 2*self.df_lim*np.random.uniform() - self.df_lim
			else: 
				s[i] = self.lims[i,0] + np.random.uniform()*\
					(self.lims[i,1] - self.lims[i,0])
		return s 

	def contains(self,s):
		return (s > self.lims[:,0]).all() and (s < self.lims[:,1]).all()