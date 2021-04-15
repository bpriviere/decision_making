
# from page 87 of Handbook Of Dynamic Game Theory 2018


# standard
from scipy.linalg import solve_discrete_are
from numpy.linalg import pinv 

# custom
from solvers.solver import Solver 

# state feedback nash equillibrium, infinite horizon solution 
class SFNE(Solver):

	def __init__(self):
		pass 

	def policy(self,problem,state):
		# F,B,Q,Ru = problem.F, problem.B, problem.Q, problem.Ru
		# P = solve_discrete_are(F, B, Q, Ru) #, e=None, s=None, balanced=True)
		# K = pinv(Ru + B.T @ P @ B) @ (B.T @ P @ F)
		# action = -1*K @ state

		def func(Z):
			eqn = np.zeros((Z.shape))
			for robot_j in range(num_robots):
				robot_j_idxs = robot_j * state_dim_per_robot + \
					np.arange(state_dim_per_robot)
				
				tilde_F = 

				eqn
			Z


		Z = np.ones((2,2))

		t = np.linspace(0,-10,)

		exit()

		return action 


