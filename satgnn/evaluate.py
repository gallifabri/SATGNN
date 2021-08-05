### LOAD MODELS
import argparse
import torch
from reinforce import REINFORCE
from sat_generator import SatGenerator
from env import SatEnv
from torch.distributions import Categorical
import torch.nn.functional as F
import math
from tqdm import tqdm
import pickle
from gnn import SatGNNRecursive
import numpy as np
import io
import os
from dataset import Dataset
import gc




def save_results(avg_solved, avg_initializatios, args, prefix):
	s = f'Avg. Solved: {avg_solved:.3f}'+'\n'+f'Avg. init: {avg_initializatios:.3f}'

	if not os.path.exists(prefix+args.path+'/evaluation'):
		os.makedirs(prefix+args.path+'/evaluation')



	file = open(prefix+args.path+'/evaluation/'+args.file_name+'.txt',"w")
	file.write(s)
	file.close()


def select_action_reinforce(state, policy):
		logits = policy(state)
		probs = F.softmax(logits, dim=1)
		m = Categorical(probs)
		action = m.sample()
		
		return action.item()


def select_action_dqn(state, policy):
	with torch.no_grad():
		logits = policy(state)
		return torch.argmax(logits)



def get_action_func(model_args):
	if model_args.model == 'REINFORCE':
		return select_action_reinforce

	else:
		return select_action_dqn


def evaluate(model, data, args, model_args):
	num_steps  = math.floor(args.n * args.num_steps_coef)
	solved = 0.
	initializations = []
	select_action = get_action_func(model_args)


	for i_sample in tqdm(range(len(data))):
		sample_initializations = 0
		adjecency = data[i_sample]
		env = SatEnv(adjecency, model_args)
		

		for i_episode in tqdm(range(args.num_episodes)):
			sample_initializations += 1
			env.set_random_assignment()

			if env.state['current_sat_value'] == 1.:
				sample_initializations -= 1
				continue

			for t in range(num_steps):
				state = env.get_graph().to(torch.device(args.device))
				action = select_action(state, model).detach()
				r1, r2, done = env.step(action)
				del state
				del action
				gc.collect()
				# print(r1)

				if done:
					break

			if done:
				solved += 1
				initializations.append(sample_initializations)
				break

		del env
		gc.collect()

		 
	avg_initializatios = np.mean(initializations)
	avg_solved = solved/len(data)

	return avg_solved, avg_initializatios

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)


	# SAT hyperparameters
	parser.add_argument('--n', default=20, type=int,
						help='Number of variables')
	parser.add_argument('--coef', default=4, type=int,
						help='Clause/variable coeficient.')
	parser.add_argument('--k', default=3, type=int,
						help='Number of variables per clause.')
	parser.add_argument('--forced_sat', default=True, type=bool,
						help='Generate satisfiable instances only.')
	


	# Dataset parameters
	parser.add_argument('--samples', default=500, type=int,
						help='Number of sample problems to generate.')
	parser.add_argument('--train_coef', default=0.7, type=float,
						help='Coeficient of training samples with respect with total samples.')
	parser.add_argument('--val_coef', default=0.1, type=float,
						help='Coeficient of validation samples with respect with total samples.')
	parser.add_argument('--dataset', default='ksat', type=str,
						help='Data Distribution',
						choices=['ksat', 'sr', 'dimacs'])


	parser.add_argument('--num_steps_coef', default=2., type=float,
						help='Number of steps in episodes with respect to number of variables.')
	parser.add_argument('--num_episodes', default=50, type=int,
						help='Number of episodes per problem.')


	parser.add_argument('--path', default='test', type=str,
						help='Path to store results.')
	parser.add_argument('--file_name', '-f', default='evaluation', type=str,
						help='Path to store results.')
	parser.add_argument('--device', default='cpu', type=str,
						help='Device to use.')
	parser.add_argument('--drop_feats', action='store_true',
						help='Drop special feature.')
	parser.add_argument('--seq_length', default=6, type=int,
						help='Number of episodes per problem.')

	args = parser.parse_args()
	prefix = '../results/'



	with open(prefix+args.path+"/results.pickle", "rb") as f:

		results = CPU_Unpickler(f).load()
		# results = pickle.load(handle)

	model_args = results.args
	model_args.drop_feats = args.drop_feats
	model_args.device = args.device
	del results
	gc.collect()


	# assert args.n == model_args.n
	
	if args.dataset == 'dimacs':
		list_IDs = [i for i in range(100)]
		directory = os.pardir+'/data/dimacs/flat30-60_data/'
		dataset = Dataset(list_IDs,directory)
	else:
		dataset = SatGenerator().get_eval_dataset(args)

	torch.device(args.device)


	model = torch.load(prefix+args.path+'/model', map_location=torch.device('cpu')).to(args.device)
	model.seq_length = args.seq_length
	# model = SatGNNRecursive(model_args.emb_dim, args.n, var_feats=model_args.var_feats, u_feats=model_args.u_feats, seq_length=model_args.seq_length, hidden=model_args.hidden, hidden_dim=2*model_args.emb_dim)
 
	model.eval()

	avg_solved, avg_initializatios = evaluate(model, dataset, args, model_args)
	print(avg_solved, avg_initializatios)
	save_results(avg_solved, avg_initializatios, args, prefix)


	### need to set env options?
	### walk through code
	

#    ______            ______            ______
#   /\_____\          /\_____\          /\_____\          ____
#  _\ \__/_/_         \ \__/_/_         \ \__/_/         /\___\
# /\_\ \_____\        /\ \_____\        /\ \___\        /\ \___\
# \ \ \/ / / /        \ \/ / / /        \ \/ / /        \ \/ / /
#  \ \/ /\/ /          \/_/\/ /          \/_/_/          \/_/_/
#   \/_/\/_/              \/_/



