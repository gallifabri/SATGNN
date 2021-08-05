from collections import namedtuple
import torch
import random
from termcolor import colored

import torch.optim as optim
from tqdm import tqdm
from env import SatEnv, SatEnvLookahead
import pickle
import dgl
import torch.nn.functional as F
import math
import numpy as np

from results_manager import TrainingResults
from dataset import *
from gnn import MessageGNN, SatGNN, SatGNNRecursive
from time import process_time
from utils import *


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

class ECODQN():

	def __init__(self, args):
		pass


	## MAKE DOUBLE AN OPTION
	def optimize_model(self, env, policy_net, target_net, memory, optimizer, args):
		if len(memory) < args.batch_size:
			return
		transitions = memory.sample(args.batch_size)
		batch = Transition(*zip(*transitions))
		
	   
		
		optimizer.zero_grad()
		
		
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
											  batch.next_state)), dtype=torch.bool)
		
		if len([s for s in batch.next_state if s is not None]) > 0:
			non_final_next_states = [s for s in batch.next_state if s is not None]
			non_final_next_states_graph = dgl.batch_hetero(non_final_next_states).to(torch.device(args.device))
			
		   
		state_batch = dgl.batch_hetero(batch.state).to(torch.device(args.device))
		action_batch = torch.stack(batch.action).reshape(-1,1)
		reward_batch = torch.stack(batch.reward).reshape(-1,1)
		
		
		state_action_values = policy_net(state_batch).gather(1, action_batch)

		
		next_state_values = torch.zeros(args.batch_size).to(torch.device(args.device))
		if len([s for s in batch.next_state if s is not None]) > 0:
			next_state_values[non_final_mask] = target_net(non_final_next_states_graph).max(1)[0].detach()

		
		expected_state_action_values = (next_state_values * args.gamma) + reward_batch.T
		
		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.T)

		
		loss.backward()
		
		for name, param in policy_net.named_parameters():
			if param.grad is not None:
				param.grad.data.clamp_(-1, 1)
			

		optimizer.step()
		
		return loss.item()
		

	def optimize_double_q_learning(self, env, policy_net, target_net, memory, optimizer, args):
		if len(memory) < args.batch_size:
			return

		transitions = memory.sample(args.batch_size)
		batch = Transition(*zip(*transitions))
		optimizer.zero_grad()
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
											  batch.next_state)), dtype=torch.bool)

		if len([s for s in batch.next_state if s is not None]) > 0:
			non_final_next_states = [s for s in batch.next_state if s is not None]
			non_final_next_states_graph = dgl.batch_hetero(non_final_next_states).to(torch.device(args.device))
			
		state_batch = dgl.batch_hetero(batch.state).to(torch.device(args.device))
		action_batch = torch.stack(batch.action).reshape(-1,1)
		reward_batch = torch.stack(batch.reward).reshape(-1,1)
	   
		state_action_values = policy_net(state_batch).gather(1, action_batch)

		next_state_values = torch.zeros(args.batch_size).to(args.device)
		if len([s for s in batch.next_state if s is not None]) > 0:
			next_state_actions = policy_net(non_final_next_states_graph).argmax(1).detach()
			next_state_targets = target_net(non_final_next_states_graph).detach()
			next_state_targets[next_state_targets<0] = 0
			next_state_values[non_final_mask] = next_state_targets[torch.arange(next_state_targets.size(0)), next_state_actions]

			
		expected_state_action_values = (next_state_values * args.gamma) + reward_batch.T
		
		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.T)

		
		loss.backward()
		
		for name, param in policy_net.named_parameters():
			if param.grad is not None:
				param.grad.data.clamp_(-1, 1)
			

		optimizer.step()
		
		return loss.item()


	def get_eps(self, step, t, max_t=100):
		if step == 0:
			return 0.
		elif t > max_t:
			return 0.05
		else:
			return max((max_t - t) / (max_t), 0.05)    
		

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






	def select_action(self, graph, policy_net, args, eps=0.01, store_q_vals=False):
		if random.random() > eps:
			with torch.no_grad():
				logits = policy_net(graph)
				return torch.argmax(logits), torch.max(logits).item()
		else:
			return torch.randint(policy_net.num_var,(1,1))[0][0].to(args.device), None
				

	def save_results(self, data, path):
		with open(path+'.pickle', 'wb') as handle:
			pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


	def evaluate(self, model, data, args, eval_type='val'):
		model.eval()

		num_steps  = math.floor(args.n * args.num_steps_coef)
		eval_score = 0.
		episodes_run = 0
		solved = 0
		scores = []
		EnvClass = SatEnvLookahead if args.lookahead else SatEnv
		skipped=0

	

		num_episodes = args.num_val_episodes

		for i_sample in tqdm(range(len(data))):
			adjecency = data[i_sample]
			env = EnvClass(adjecency, args)
			sample_score = 1.
			episodes_run += 1
			

			for i_episode in range(num_episodes):
				env.set_random_assignment()
				current_sat_value = env.state['current_sat_value']
				max_score = current_sat_value

				if current_sat_value == 1.:
					skipped +=1
					continue


				for t in range(num_steps):
					state = env.get_graph().to(torch.device(args.device))
					action, _ = self.select_action(state, model, args, eps=0.)
					obs_reward, reward, done = env.step(action)
					max_score = max(obs_reward, max_score)

					if done:
						solved += 1
						break

				scores.append(max_score)

				if done:
					break
				else:
					sample_score -= 1/num_episodes

			eval_score += sample_score
			




		model.train()

		eval_mean_score = np.mean(scores).item()

		return (eval_score/len(data), episodes_run/len(data), solved/(len(data)-skipped), eval_mean_score)


	def train(self, data, args):
		start_time = process_time() 
		train_data, val_data, eval_data = data
		results = TrainingResults(args)

		torch.manual_seed(args.seed)
		random.seed(args.seed)
		
		
		num_episodes = len(train_data)
		
		## MODEL
		ModelClass = SatGNNRecursive if args.recursive else SatGNN 
		EnvClass = SatEnvLookahead if args.lookahead else SatEnv

		policy_net = ModelClass(args.emb_dim, args.n, var_feats=args.var_feats, u_feats=args.u_feats, seq_length=args.seq_length, hidden=args.hidden, hidden_dim=2*args.emb_dim)
		target_net = ModelClass(args.emb_dim, args.n, var_feats=args.var_feats, u_feats=args.u_feats, seq_length=args.seq_length, hidden=args.hidden, hidden_dim=2*args.emb_dim)
		target_net.load_state_dict(policy_net.state_dict())
		target_net.eval()

		policy_net.to(args.device)
		target_net.to(args.device)

		optimizer = optim.Adam(policy_net.parameters(),lr=args.lr)
		memory = ReplayMemory(args.memory_size)

		optim_func = self.optimize_double_q_learning if args.model == 'DDQN' else self.optimize_model
		print(optim_func)

		## TRAINING LOOP
		it = 0
		eps_decay = args.epochs * num_episodes * args.eps_decay_steps
		episodes_run = 0
		running_reward = 0.
		best_eval_score = 0.
		best_eval_solved = 0.
		solved = []



		for e in range(args.epochs):
			train_data.permute_samples()
			
			for i_episode in range(num_episodes):
				t1_start = process_time() 
				env = EnvClass(train_data[i_episode], args)
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
					action, max_q_val = self.select_action(env.get_graph().to(torch.device(args.device)), policy_net, args, eps=self.get_eps(t,it,max_t=eps_decay))
					current_graph = env.get_graph_copy()
					obs_reward, reward, done = env.step(action)
					
											
					if not done:
						next_graph = env.get_graph_copy()
					else:
						
						next_graph = None
					
					memory.push(current_graph, action, next_graph, reward)

					loss = optim_func(env, policy_net, target_net, memory, optimizer, args)
				   
					color = self.get_color(env)
					
					if obs_reward >  best:
						best = obs_reward
					
 
					
					
					if args.verbose:
						loss = loss if loss is not None else 0.
						if max_q_val is not None:
							print(colored(f'{t:3d}/{args.num_steps}. best: {best:.3f}, obs r: {obs_reward:.3f}, adj r: {reward:6.3f}, q_val: {max_q_val:6.3f}, loss: {loss:6.5f}, action: {action:d}', color))
						else:
							print(colored(f'{t:3d}/{args.num_steps}. best: {best:.3f}, obs r: {obs_reward:.3f}, adj r: {reward:6.3f}, q_val: {None}  , loss: {loss:6.5f}, action: {action:d}', color))

					
					episode_data.add_step(obs_reward.item(), best.item(), reward.item(), action, self.get_color(env))

					if done:
						break

				solved.append(done)

				if running_reward  == 0.:
					running_reward = best.item()
				else:
					running_reward = 0.05 * best.item() + (1 - 0.05) * running_reward
						
				if args.verbose:
					t2_start = process_time()
					print(f'Updates: {it}. Best score: {best.item():.3f}. Average reward: {running_reward:.3f}, Time: {t2_start-t1_start:.5f}')
					print()
				
				
				if i_episode % args.target_update == 0:
					target_net.load_state_dict(policy_net.state_dict())
				

				if not episodes_run % args.eval_freq or episodes_run == 1:
					start_val_time = process_time()
					eval_score = self.evaluate(policy_net, val_data, args)
					results.add_val_scores(episodes_run, eval_score)
					solved_avg = np.mean(solved)
					results.add_solved_avg(episodes_run, solved_avg)
					solved = []
					results.save()

					if eval_score[3] > best_eval_score:
						best_eval_score = eval_score[3]
						# torch.save(policy_net, args.save_path+'model')

						torch.save(policy_net.to('cpu'), args.save_path+'model')
						policy_net.to(args.device)

					best_eval_solved = eval_score[2] if eval_score[2] > best_eval_solved else best_eval_solved


					with open(args.save_path+'val_results.txt', 'a') as f:
						current_time = process_time() - start_time
						s = f'Val score: {eval_score[3]:.3f}, Val solved: {eval_score[2]:.3f}, Train Solved: {solved_avg:.3f}, it: {episodes_run}, val time: {current_time-start_val_time:.3f}, total time: {current_time:.3f}'+'\n'
						f.write(s)

					plot_results(results)
					print(f'Episodes run: {episodes_run}/{args.epochs*num_episodes}. Evaluation score: {eval_score[0]:.3f}')



		# eval_score = self.evaluate(policy_net, eval_data, args, eval_type='eval')
		# results.last_model_eval_score = eval_score
		# best_model = torch.load(args.save_path+'model')
		# eval_score = self.evaluate(best_model, eval_data, args, eval_type='eval')
		# results.best_model_eval_score = eval_score



		end_time = process_time()
		results.train_time = end_time - start_time
		results.save()

		with open(args.save_path+'best_val.txt', 'w') as f:
						s = f'Best val score: {best_eval_score:.3f}, Best val solved: {best_eval_solved:.3f}, total_time: {end_time - start_time:.3f}'+'\n'
						f.write(s)
					

#    ______            ______            ______
#   /\_____\          /\_____\          /\_____\          ____
#  _\ \__/_/_         \ \__/_/_         \ \__/_/         /\___\
# /\_\ \_____\        /\ \_____\        /\ \___\        /\ \___\
# \ \ \/ / / /        \ \/ / / /        \ \/ / /        \ \/ / /
#  \ \/ /\/ /          \/_/\/ /          \/_/_/          \/_/_/
#   \/_/\/_/              \/_/




