
import numpy as np 

class Param: 

	def __init__(self):

		# names 
		self.problem_name = "example1" # e.g. example1, ...
		self.solver_name = "PUCT" # e.g. Empty, DARE, MCTS, PUCT, GPUCT...

		# filenames 
		self.curr_plot_fn = "../current/run.pdf"

	def to_dict(self):
		return self.__dict__