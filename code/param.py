
import numpy as np 

class Param: 

	def __init__(self):

		# names 
		self.problem_name = "example2" # e.g. example1, example2, ...
		self.solver_name = "PUCT" # e.g. Empty, DARE, MCTS, PUCT, ...

		# filenames 
		self.curr_plot_fn = "../current/run.pdf"

	def to_dict(self):
		return self.__dict__