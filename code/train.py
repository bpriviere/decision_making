
# standard 
import numpy as np 
import torch 
import random 
import pickle 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

# custom
import plotter 
from param import Param 
from run import make_instance, run_instance
from solvers.puct import PUCT 
from solvers.policy_solver import PolicySolver
from learning.policy_network import PolicyNetwork
from learning.value_network import ValueNetwork
from util import write_dataset, get_dataset_fn, get_oracle_fn, format_dir

L = 2 
num_D_pi = 10
num_D_v = 10 
num_simulations_expert = 10
num_simulations_learner = 10
learning_rate = 0.001
num_epochs = 10
batch_size = 2
train_test_split = 0.8


class Dataset(torch.utils.data.Dataset):
	
	def __init__(self, src_file, encoding_dim, target_dim, device='cpu'):
		with open(src_file, 'rb') as h:
			datapoints = np.load(h)
		self.X_np, self.target_np = datapoints[:,0:encoding_dim], datapoints[:,encoding_dim:]
		self.X_torch = torch.tensor(self.X_np,dtype=torch.float32,device=device)
		self.target_torch = torch.tensor(self.target_np,dtype=torch.float32,device=device)

	def __len__(self):
		return self.X_torch.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		return self.X_torch[idx,:], self.target_torch[idx,:]

	def to(self,device):
		self.X_torch = self.X_torch.to(device)
		self.target_torch = self.target_torch.to(device)


def make_self_play_states(problem,num_states,policy_oracle,value_oracle):
	self_play_states = [] 
	while len(self_play_states) < num_states: 
		instance = dict()
		instance["problem"] = problem 
		instance["initial_state"] = problem.initialize() 
		instance["solver"] = PUCT(\
			policy_oracle=policy_oracle,
			value_oracle=value_oracle,
			number_simulations=num_simulations_learner)
		sim_result = run_instance(instance,verbose=False)
		self_play_states.extend(sim_result["states"])
	return self_play_states


def make_expert_demonstration_pi(problem,robot,states,policy_oracle,value_oracle):
	datapoints = []
	solver = PUCT(\
		policy_oracle=policy_oracle,
		value_oracle=value_oracle,
		number_simulations=num_simulations_expert)
	action_dim_per_robot = int(problem.action_dim / problem.num_robots)
	robot_action_idx = action_dim_per_robot * robot + np.arange(action_dim_per_robot)
	for state in states: 
		root_node = solver.search(problem,state)
		if root_node is not None:
			actions,num_visits = list(zip(*[(a.squeeze(),c.num_visits) for (c,a) in root_node.edges.items()]))
			encoding = problem.policy_encoding(state,robot).squeeze()
			robot_actions = np.array(actions)[:,robot_action_idx]
			target = np.average(robot_actions, weights=num_visits, axis=0)
			datapoint = np.append(encoding,target)
			datapoints.append(datapoint)
	random.shuffle(datapoints)
	split = int(len(datapoints)*train_test_split)
	train_dataset = datapoints_to_dataset(datapoints[0:split],"train_policy",\
		problem.policy_encoding_dim,action_dim_per_robot,robot=robot)
	test_dataset = datapoints_to_dataset(datapoints[split:],"test_policy",\
		problem.policy_encoding_dim,action_dim_per_robot,robot=robot)
	plotter.plot_policy_dataset(problem,train_dataset,test_dataset)
	plotter.save_figs("../current/models/policy_dataset_l{}_i{}.pdf".format(l,robot))
	return train_dataset, test_dataset


def datapoints_to_dataset(datapoints,oracle_name,encoding_dim,target_dim,robot=0):
	dataset_fn = get_dataset_fn(oracle_name,l,robot=robot)
	datapoints = np.array(datapoints)
	write_dataset(datapoints,dataset_fn)
	dataset = Dataset(dataset_fn,encoding_dim,target_dim)
	return dataset


def make_expert_demonstration_v(problem,states,policy_oracle):
	datapoints = []
	for state in states: 
		instance = dict()
		instance["problem"] = problem 
		instance["initial_state"] = state
		instance["solver"] = PolicySolver(problem,policy_oracle)
		sim_result = run_instance(instance,verbose=False)
		value = calculate_value(problem,sim_result)
		encoding = problem.value_encoding(state).squeeze()
		datapoint = np.append(encoding,value)		
		datapoints.append(datapoint)
	random.shuffle(datapoints) 
	split = int(len(datapoints)*train_test_split)
	train_dataset = datapoints_to_dataset(datapoints[0:split],"train_value",\
		problem.value_encoding_dim,1)
	test_dataset = datapoints_to_dataset(datapoints[split:],"test_value",\
		problem.value_encoding_dim,1)
	plotter.plot_value_dataset(problem,train_dataset,test_dataset)
	plotter.save_figs("../current/models/value_dataset_l{}.pdf".format(l))
	return train_dataset, test_dataset


def calculate_value(problem,sim_result):
	value = np.zeros((1))
	states = sim_result["states"]
	actions = sim_result["actions"]
	for step,(state,action) in enumerate(zip(states,actions)):
		reward = problem.normalized_reward(state,action)
		value += (problem.gamma ** step) * reward 
	return value 


def train_model(problem,train_dataset,test_dataset,l,oracle_name,robot=0):

	# device = "cpu"
	device = "cuda"
	model_fn = get_oracle_fn(oracle_name,l,robot=robot)

	if oracle_name == "policy":
		model = PolicyNetwork(problem,oracle_name,device=device)
	elif oracle_name == "value":
		model = ValueNetwork(problem,oracle_name,device=device)
	model.to(device)

	optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
	scheduler = ReduceLROnPlateau(optimizer, 'min', \
		factor=0.5, patience=50, min_lr=1e-4, verbose=True)

	train_dataset.to(device)
	test_dataset.to(device)
	train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size)
	test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)	

	losses = []
	best_test_loss = np.Inf
	for epoch in range(num_epochs): 
		train_epoch_loss = train(model,optimizer,train_loader)
		test_epoch_loss = test(model,test_loader)
		scheduler.step(test_epoch_loss)
		losses.append((train_epoch_loss,test_epoch_loss))
		if test_epoch_loss < best_test_loss:
			best_test_loss = test_epoch_loss
			torch.save(model.to('cpu').state_dict(),model_fn)
			model.to(device)
	plotter.plot_loss(losses)
	plotter.save_figs("../current/models/losses_{}_l{}_i{}.pdf".format(oracle_name,l,robot))


def train(model,optimizer,loader):
	epoch_loss = 0
	loss_by_components = []
	for step, (x,target) in enumerate(loader):
		mu, logvar = model(x,training=True)
		loss = loss_fnc(mu,logvar,target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		epoch_loss += float(loss)
	return epoch_loss/step


def test(model,loader):
	epoch_loss = 0
	loss_by_components = []
	for step, (x,target) in enumerate(loader):
		mu, logvar = model(x,training=True)
		loss = loss_fnc(mu,logvar,target)
		epoch_loss += float(loss)
	return epoch_loss/step


def loss_fnc(mu,logvar,target):
	criterion = MSELoss(reduction='none')
	loss = torch.sum(criterion(mu, target) / (2 * torch.exp(logvar)) + 1/2 * logvar)
	loss = loss / mu.shape[0]
	return loss 


if __name__ == '__main__':

	param = Param()
	instance = make_instance(param)
	problem = instance["problem"]
	format_dir(clean_dirnames=["data","models"]) 

	# training 
	for l in range(L):
		print('learning iteration: {}/{}'.format(l,L))

		if l == 0:
			policy_oracle = [None for _ in range(problem.num_robots)]
			value_oracle = None 
		else: 
			policy_oracle = [get_oracle_fn("policy",l-1,robot=i) for i in range(problem.num_robots)]
			value_oracle = get_oracle_fn("value",l-1)

		for robot in range(problem.num_robots): 
			print('\t policy training iteration l/L, i/N: {}/{} {}/{}...'.format(\
				l,L,robot,problem.num_robots))
			states_pi = make_self_play_states(problem,num_D_pi,policy_oracle,value_oracle)
			train_dataset_pi, test_dataset_pi = make_expert_demonstration_pi(\
				problem,robot,states_pi,policy_oracle,value_oracle)
			train_model(problem,train_dataset_pi,test_dataset_pi,l,"policy",robot=robot)

		print('\t value training l/L: {}/{}'.format(l,L))
		states_v = make_self_play_states(problem,num_D_v,policy_oracle,value_oracle)
		train_dataset_v, test_dataset_v = make_expert_demonstration_v(problem,states_v,policy_oracle)
		train_model(problem,train_dataset_v,test_dataset_v,l,"value") 

