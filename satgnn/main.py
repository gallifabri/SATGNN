from ecodqn import ECODQN
import argparse
from sat_generator import SatGenerator
from  reinforce import  REINFORCE
from random_model import Random
import os
from utils import *
from time import process_time


def get_model(args):
	if args.model == 'DDQN':
		return ECODQN
	elif args.model == 'DQN':
		return ECODQN
	elif args.model == 'REINFORCE':
		return REINFORCE
	elif args.model == 'RANDOM':
		return Random


def make_results_dir(args):
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# SAT hyperparameters
	parser.add_argument('--n', default=10, type=int,
						help='Number of variables')
	parser.add_argument('--coef', default=4, type=int,
						help='Clause/variable coeficient.')
	parser.add_argument('--k', default=3, type=int,
						help='Number of variables per clause.')
	parser.add_argument('--forced_sat', default=True, type=bool,
						help='Generate satisfiable instances only.')
	


	# Dataset parameters
	parser.add_argument('--samples', default=1000, type=int,
						help='Number of sample problems to generate.')
	parser.add_argument('--train_coef', default=0.7, type=float,
						help='Coeficient of training samples with respect with total samples.')
	parser.add_argument('--val_coef', default=0.1, type=float,
						help='Coeficient of validation samples with respect with total samples.')
	parser.add_argument('--dataset', default='ksat', type=str,
						help='Data Distribution',
						choices=['ksat', 'sr'])


	# GNN hyperparameters
	parser.add_argument('--emb_dim', default=128, type=int,
						help='Embedding dimension for GNN.')
	parser.add_argument('--seq_length', default=6, type=int,
						help='Number of layers for message passing.')
	parser.add_argument('--hidden', default=0, type=int,
						help='Number of layers for fully connected.')
	parser.add_argument('--recursive', '-r', action='store_true',
						help='Model with recursive message layer.')


	# Training hyperparameters
	parser.add_argument('--model', default='REINFORCE', type=str,
						help='Model type to use',
						choices=['DDQN', 'DQN', 'REINFORCE', 'RANDOM'])
	parser.add_argument('--target_update', default=10, type=int,
						help='Target update')
	parser.add_argument('--eps_decay_steps', default=.1, type=float,
						help='Number of steps over which epsilon decays.')
	parser.add_argument('--gamma', default=0.95, type=float,
						help='Discount factor.')
	parser.add_argument('--memory_size', default=5000, type=int,
						help='Size of DQN memory.')
	parser.add_argument('--num_steps_coef', default=1., type=float,
						help='Number of steps in episodes with respect to number of variables.')


	# Evaluation hyperparameters
	parser.add_argument('--eval_freq', default=100, type=int,
						help='Evaluation frequency in number of episodes.')
	parser.add_argument('--num_val_episodes', default=1, type=int,
						help='Number of episodes for each random init.')
	parser.add_argument('--num_eval_episodes', default=50, type=int,
						help='Number of episodes for each random init.')


	# Optimizer hyperparameters
	parser.add_argument('--lr', default=1e-4, type=float,
						help='Learning rate to use')
	parser.add_argument('--batch_size', default=32, type=int,
						help='Minibatch size')


	# Reward hyperparameters
	parser.add_argument('--rew_on_sat', action='store_true',
						help='Assigns reward on satisfiable states.')
	parser.add_argument('--sat_reward_type', default='one', type=str,
						help='Type of sat reward',
						choices=['one', 'num_clause'])
	parser.add_argument('--rew_on_lm', action='store_true',
						help='Assigns reward on local maxima states.')
	parser.add_argument('--lm_reward_type', default='num_clause', type=str,
						help='Type of local maxima reward',
						choices=['num_clause', 'zero'])
	parser.add_argument('--reward_type', default='clause', type=str,
						help='Use variable or clauses to determine reward.',
						choices=['clause', 'variable'])



	# Other hyperparameters
	parser.add_argument('--epochs', default=8, type=int,
						help='Max number of epochs')
	parser.add_argument('--seed', default=42, type=int,
						help='Seed to use for reproducing results')
	parser.add_argument('--verbose', '-v', action='store_true',
						help='Print training details.')
	parser.add_argument('--path', default='test', type=str,
						help='Path to store results.')
	parser.add_argument('--device', default='cpu', type=str,
						help='Device to use.')


	# Env hyperparameters
	parser.add_argument('--lookahead', action='store_true',
						help='lookahead option.')
	parser.add_argument('--var_feats', default=2, type=int,
						help='ignore')
	parser.add_argument('--u_feats', default=4, type=int,
						help='ignore')
	parser.add_argument('--drop_feats', action='store_true',
						help='Drop special feature.')



	args = parser.parse_args()

	if args.lookahead:
		args.var_feats = 4
		args.u_feats = 5
	


	args.save_path = f'../results/{args.path}/'
	args.num_steps  = math.floor(args.n * args.num_steps_coef)
	make_results_dir(args)
	make_args_file(args)

	f = open(args.save_path+"val_results.txt", "w")
	f.close()

	f = open(args.save_path+"best_val.txt", "w")
	f.close()

	dataset = SatGenerator().get_dataset(args)


	
	trainer = get_model(args)

	model = trainer(args)
	model.train(dataset, args)

	results = load_results(args.save_path)
	make_eval_summary(args.save_path, results)
	plot_results(results)


### TODO:


## DIMACS TO DATA SCRIPT

