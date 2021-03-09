
# standard
from scipy.linalg import solve_discrete_are
from numpy.linalg import pinv 

# custom
from solvers.solver import Solver 


class DARE(Solver):

	def __init__(self):
		pass 

	def policy(self,problem,state):
		F,B,Q,Ru = problem.F, problem.B, problem.Q, problem.Ru
		P = solve_discrete_are(F, B, Q, Ru) #, e=None, s=None, balanced=True)
		K = pinv(Ru + B.T @ P @ B) @ (B.T @ P @ F)
		action = -1*K @ state
		return action 





