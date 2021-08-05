from collections import namedtuple
import torch
import random
from termcolor import colored

import torch.optim as optim
from tqdm import tqdm
from env import SatEnv
import pickle
import dgl
import torch.nn.functional as F
import math

from results_manager import TrainingResults
from dataset import *
from gnn import MessageGNN, SatGNN
from time import process_time




class Random():

	def __init__(self, args):
		self.args = args


	def get_color(self, env):
		if env.state['is_local_maxima']:
			if env.state['current_sat_value'] == 1.:
				return 'blue'
			return 'magenta'
		if env.state['is_increasing']:
			if env.state['current_sat_value'] == 1.:
				return 'blue'
			return 'green'
		else:
			return 'red'



	def select_action(self):
		return torch.randint(self.args.n,(1,1))[0][0].to(self.args.device)
				


	def evaluate(self, data, args, eval_type='val'):
		num_steps  = math.floor(args.n * args.num_steps_coef)
		eval_score = 0.
		episodes_run = 0
		solved = 0

		if eval_type == 'eval':
			num_episodes = args.num_eval_episodes
		else:
			num_episodes = args.num_val_episodes

		for i_sample in tqdm(range(len(data))):
			adjecency = data[i_sample]
			env = SatEnv(adjecency, args)
			sample_score = 1.
			episodes_run += 1

			for i_episode in range(num_episodes):
				env.set_random_assignment()
				current_sat_value = env.state['current_sat_value']

				if current_sat_value == 1.:
					continue

				for t in range(num_steps):
					action = self.select_action()
					obs_reward, reward, done = env.step(action)
					if done:
						solved += 1
						break

				if done:
					break
				else:
					sample_score -= 1/num_episodes

			eval_score += sample_score



		return (eval_score/len(data), episodes_run/len(data), solved/len(data))


	def train(self, data, args):
		start_time = process_time() 
		train_data, val_data, eval_data = data
		results = TrainingResults(args)
		torch.manual_seed(args.seed)
		random.seed(args.seed)
		
		
		num_episodes = len(train_data)
		


		## TRAINING LOOP
		it = 0
		episodes_run = 0
		running_reward = 0.
		best_eval_score = 0

		for e in range(args.epochs):
			train_data.permute_samples()
			
			for i_episode in range(num_episodes):
				t1_start = process_time() 
				env = SatEnv(train_data[i_episode], args)
				episodes_run += 1
				env.set_random_assignment()
				initial_value = env.state['current_sat_value']
				best = initial_value
				
				if initial_value == 1.:
					continue
				if args.verbose:
					print(colored(f'Epoch {e+1}/{args.epochs}. Episode {i_episode+1}/{num_episodes}. Initial value: {initial_value:.3f}', 'cyan'))
				
				episode_data = results.make_episode_record(episodes_run, initial_value.item())

				for t in range(args.num_steps):
					it += 1
					action = self.select_action()
					obs_reward, reward, done = env.step(action)
					
					if obs_reward >  best:
						best = obs_reward
					
					if args.verbose:
						print(colored(f'{t:3d}/{args.num_steps}. best: {best:.3f}, obs r: {obs_reward:.3f}, adj r: {reward:6.3f}, action: {action:d}', self.get_color(env)))

					episode_data.add_step(obs_reward.item(), best.item(), reward.item(), action, self.get_color(env))

					if done:
						break

				if running_reward  == 0.:
					running_reward = best.item()
				else:
					running_reward = 0.05 * best.item() + (1 - 0.05) * running_reward
						
				if args.verbose:
					t2_start = process_time()
					print(f'Updates: {it}. Best score: {best.item():.3f}. Average reward: {running_reward:.3f}, Time: {t2_start-t1_start:.5f}')
					print()
				
				
				if not episodes_run % args.eval_freq or episodes_run == 1:
					eval_score = self.evaluate(val_data, args)
					results.add_val_scores(episodes_run, eval_score)
					results.save()

					if eval_score[0] > best_eval_score:
						best_eval_score = eval_score[0]

					print(f'Episodes run: {episodes_run}/{args.epochs*num_episodes}. Evaluation score: {eval_score[0]:.3f}')



		eval_score = self.evaluate(eval_data, args, eval_type='eval')
		results.last_model_eval_score = eval_score
		eval_score = self.evaluate(eval_data, args, eval_type='eval')
		results.best_model_eval_score = eval_score

		end_time = process_time()
		results.train_time = end_time - start_time
		results.save()
					



