import pickle
import math
import matplotlib.pyplot as plt
import numpy as np

nl = '\n'

def make_args_file(args):
	s = ''
	s += f'DATA HYPERPARAMETERS'+nl
	s += f'  n: {args.n}'+nl
	s += f'  k: {args.k}'+nl
	s += f'  m: {args.n * args.coef}'+nl
	s += f'  samples: {args.samples}'+nl+nl
	s += f'MODEL TYPE: {args.model}'+nl+nl
	s += f'TRAINING HYPERPARAMETERS:'+nl
	s += f'  lr: {args.lr}'+nl
	s += f'  gamma: {args.gamma:.2f}'+nl
	s += f'  max episode length: {math.floor(args.n * args.num_steps_coef)}'+nl
	s += f'  max episodes: {args.epochs * math.floor(args.samples * args.train_coef)}'+nl+nl

	if args.model != 'REINFORCE':
		s += f'DQN HYPERPARAMETERS'+nl
		s += f'  batch_size: {args.batch_size}'+nl
		s += f'  memory size: {args.memory_size}'+nl
		s += f'  target_update: {args.target_update}'+nl
		s += f'  epsilon decay: {int(args.eps_decay_steps*100)}%'+nl+nl
		

	s += f'SEED: {args.seed}'+nl+nl
	s += f'GNN HYPERPARAMETERS:'+nl
	s += f'  Embedding dimension: {args.emb_dim}'+nl
	s += f'  Message layers: {args.seq_length}'+nl
	s += f'  Fully connected hidden layers: {args.hidden}'+nl+nl
	s += f'EVALUATION HYPERPARAMETERS:'+nl
	s += f'  Evaluation frequency: {args.eval_freq}'+nl
	s += f'  Random initializations per problem: {args.num_eval_episodes}'+nl
	s += f'  Validation set size: {math.floor(args.samples*args.val_coef)}'+nl
	s += f'  Evaluation set size: {math.floor(args.samples*(1-args.val_coef-args.train_coef))}'+nl



	file = open(args.save_path+'args.txt',"w")
	file.write(s)
	file.close()

	# with open(args.save_path+'args.txt', 'wb') as handle:
	# 		pickle.dump(s, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_train_data_view(args):
	pass


def load_results(path):
	with open(path+"results.pickle", "rb") as handle:
		res = pickle.load(handle)

	return res

def make_eval_summary(path, results):
	s = ''
	# s += 'BEST MODEL:'+nl
	# s += f'  eval score: {results.best_model_eval_score[0]:.3f}'+nl
	# s += f'  Episodes solved: {results.best_model_eval_score[2]:.3f}'+nl
	# s += f'  Number of random initializations to solve: {results.best_model_eval_score[1]:.3f}'+nl
	# s += f'  episode: {results.best_model_step}/{len(results.episodes)}'+nl+nl
	# s += 'LAST MODEL:'+nl
	# s += f'  eval score: {results.last_model_eval_score[0]:.3f}'
	# s += f'  Episodes solved: {results.last_model_eval_score[2]:.3f}'+nl
	# s += f'  Number of random initializations to solve: {results.last_model_eval_score[1]:.3f}'+nl+nl
	# s += f'TRAIN TIME: {results.train_time:.3f}'

	file = open(path+'evluation_summary.txt',"w")
	file.write(s)
	file.close()

def smooth(x, N=50):
	cumsum = np.cumsum(np.insert(x, 0, 0)) 
	return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_results(results):
	args = results.args
	model_name = f'{args.model} - {args.n} variables'
	N = 50

	episodes_idx = [i for i in range(len(results.episodes))]
	episodes_scores = [episode.best_value for episode in results.episodes]

	val_idx = [key for key in results.val_socres.keys()]
	val_idx2 = [key-N for key in results.val_socres.keys() if key >= N]
	val_socres = [results.val_socres[key][0] for key in val_idx]
	val_sat_socres = [results.val_socres[key][3] for key in val_idx if key>=N]
	val_len_episodes = [results.val_socres[key][1] for key in val_idx]
	val_solved = [results.val_socres[key][2] for key in val_idx]
	train_solved = [results.solved_avg[key] for key in val_idx]

	# val_idx2 = [i for i in val_idx]
	

	plt.figure()
	plt.plot(smooth(episodes_scores, N=N))
	plt.plot(val_idx2, val_sat_socres)
	plt.title(f'{model_name} - Train Results')
	plt.xlabel('Episode')
	plt.ylabel('Best SAT value')
	plt.savefig(args.save_path+'scores_plot.png')
	plt.close()

	# plt.figure()
	# plt.plot(val_idx, val_socres)
	
	# plt.title(f'{model_name} - Validation Results')
	# plt.xlabel('Iteration')
	# plt.ylabel('Evaluation Score')
	# plt.savefig(args.save_path+'val_plot.png')
	# plt.close()

	# plt.figure()
	# plt.plot(val_idx, val_len_episodes)
	# plt.title(f'{model_name} - Validation: Number of Random Init. to Solve')
	# plt.xlabel('Iteration')
	# plt.ylabel('Num of Episodes')
	# plt.savefig(args.save_path+'val_plot_ep_length.png')
	# plt.close()


	plt.figure()
	plt.plot(val_idx, train_solved)
	plt.plot(val_idx, val_solved)
	
	plt.title(f'{model_name} - Validation: Episodes solved')
	plt.xlabel('Iteration')
	plt.ylabel('% Solved')
	plt.savefig(args.save_path+'solved_plot.png')
	plt.close()






