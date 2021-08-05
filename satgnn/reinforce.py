import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from termcolor import colored

from gnn import MessageGNN, SatGNN, SatGNNRecursive
from env import SatEnv, SatEnvLookahead
from results_manager import TrainingResults
from utils import plot_results

from tqdm import tqdm
from time import process_time
from utils import *


import csv




class REINFORCE():
    def __init__(self, args):
        self.eps = np.finfo(np.float32).eps.item()


    def select_action(self, state, log_probs, policy, eval=False):
        logits = policy(state)
        probs = F.softmax(logits, dim=1)
        m = Categorical(probs)
        action = m.sample()
        if not eval:
            log_probs.append(m.log_prob(action))

        return action.item()


    def finish_episode(self, args, log_probs, rewards,  optimizer, policy):
        R = 0
        policy_loss = []
        returns = []

        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum().to(args.device)
        policy_loss.backward()

        # for name, param in policy.named_parameters():
        #     if param.grad is not None:
        #         print(param.grad.device, param.device)

        optimizer.step()


    def get_color(self, env):
        if env.backtrack:
            return 'grey'
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



    def evaluate(self, model, data, args, eval_type='val'):
        model.eval()

        num_steps  = math.floor(args.n * args.num_steps_coef)
        eval_score = 0.
        episodes_run = 0
        solved = 0
        scores = []
        skipped=0

        EnvClass = SatEnvLookahead if args.lookahead else SatEnv

        
        num_episodes = args.num_val_episodes

        for i_sample in tqdm(range(len(data))):
            adjecency = data[i_sample]
            env = EnvClass(adjecency, args)
            sample_score = 1.
            

            for i_episode in range(num_episodes):

                env.set_random_assignment()
                max_score = env.state['current_sat_value']

                if env.state['current_sat_value'] == 1.:
                    skipped +=1
                    continue

                episodes_run += 1


                for t in range(num_steps):
                    state = env.get_graph().to(torch.device(args.device))
                    action = self.select_action(state, None, model, eval=True)
                    obs_reward, reward, done = env.step(action)
                    max_score = max(max_score, obs_reward)
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


    def train(self, dataset, args):
        start_time = process_time() 
        train_data, val_data, eval_data = dataset
        torch.manual_seed(args.seed)
        torch.device(args.device)
        results = TrainingResults(args)

        ModelClass = SatGNNRecursive if args.recursive else SatGNN 

        policy = ModelClass(args.emb_dim, args.n, var_feats=args.var_feats, u_feats=args.u_feats, seq_length=args.seq_length, hidden=args.hidden, hidden_dim=2*args.emb_dim).to(args.device)
        print(next(policy.parameters()).device)

        optimizer = optim.Adam(policy.parameters(), lr=args.lr)

        running_reward = 0.
        num_steps  = math.floor(args.n * args.num_steps_coef)
        num_episodes = len(train_data)
        episodes_run = 0
        best_eval_score = 0.
        best_eval_solved = 0.
        solved = []

        EnvClass = SatEnvLookahead if args.lookahead else SatEnv

        for e in range(args.epochs):
            train_data.permute_samples()

            for i_episode in range(num_episodes):
                t1_start = process_time() 
                episodes_run += 1
                env = EnvClass(train_data[i_episode], args)
                saved_log_probs = []
                rewards = []
                best = 0.
                env.set_random_assignment()
                state, ep_reward = env.get_graph().to(torch.device(args.device)), 0
                initial_value = env.state['current_sat_value']


                if initial_value == 1.:
                    continue

                if args.verbose:
                    # print(colored(f'initial value: {current_sat_value:.3f}', 'cyan'))
                    print(colored(f'Epoch {e+1}/{args.epochs}. Episode {i_episode+1}/{num_episodes}. Initial value: {initial_value:.3f}', 'cyan'))

                episode_data = results.make_episode_record(episodes_run, initial_value.item())

                for t in range(args.num_steps):  
                    action = self.select_action(state, saved_log_probs, policy)
                    obs_reward, reward, done = env.step(action)
                    state = env.get_graph().to(torch.device(args.device))
                    

                    rewards.append(reward)
                    ep_reward += reward

                    if obs_reward > best:
                        best = obs_reward

                    if args.verbose:
                        print(colored(f'{t:3d}/{args.num_steps}. best: {best:.3f}, observed: {obs_reward:.3f}, actual: {reward:.3f}, action: {action}', self.get_color(env)))
                    
                    episode_data.add_step(obs_reward.item(), best.item(), reward.item(), action, self.get_color(env))

                    if done:
                        break

                solved.append(done)

                
                if running_reward  == 0.:
                    running_reward = best.item()
                else:
                    running_reward = 0.05 * best.item() + (1 - 0.05) * running_reward


                self.finish_episode(args, saved_log_probs, rewards,  optimizer, policy)

                if args.verbose:
                    t2_start = process_time() 
                    print(f'Updates: {episodes_run}. Episode score: {best:.3f}. Average reward: {running_reward:.3f}. Time: {t2_start-t1_start:.5f}', '\n')


                

                if not episodes_run % args.eval_freq or episodes_run == 1:
                    start_val_time = process_time()
                    eval_score = self.evaluate(policy, val_data, args)
                    solved_avg = np.mean(solved)
                    results.add_val_scores(episodes_run, eval_score)
                    results.add_solved_avg(episodes_run, solved_avg)
                    results.save()
                    plot_results(results)
                    solved = []



                    if eval_score[3] > best_eval_score:
                        best_eval_score = eval_score[3]


                        torch.save(policy.to('cpu'), args.save_path+'model')
                        policy.to(args.device)

                    best_eval_solved = eval_score[2] if eval_score[2] > best_eval_solved else best_eval_solved

                    with open(args.save_path+'val_results.txt', 'a') as f:
                        current_time = process_time() - start_time
                        s = f'Val score: {eval_score[3]:.3f}, Val solved: {eval_score[2]:.3f}, Train Solved: {solved_avg:.3f}, it: {episodes_run}, val time: {current_time-start_val_time:.3f}, total time: {current_time:.3f}'+'\n'
                        f.write(s)


                    
                    plot_results(results)
                    print(f'Episodes run: {episodes_run}/{args.epochs*num_episodes}. Evaluation score: {eval_score[0]:.3f}')

        
        
        # eval_score = self.evaluate(policy, eval_data, args, eval_type='eval')
        # results.last_model_eval_score = eval_score
        # best_model = torch.load(args.save_path+'model')
        # eval_score = self.evaluate(best_model, eval_data, args, eval_type='eval')
        # results.best_model_eval_score = eval_score

        end_time = process_time()
        results.train_time = end_time - start_time
        total_time = end_time - start_time

        with open(args.save_path+'best_val.txt', 'w') as f:
                        s = f'Best val score: {best_eval_score:.3f}, Best val solved: {best_eval_solved:.3f}, total_time: {total_time:.3f}'+'\n'
                        f.write(s)

        results.save()


#    ______            ______            ______
#   /\_____\          /\_____\          /\_____\          ____
#  _\ \__/_/_         \ \__/_/_         \ \__/_/         /\___\
# /\_\ \_____\        /\ \_____\        /\ \___\        /\ \___\
# \ \ \/ / / /        \ \/ / / /        \ \/ / /        \ \/ / /
#  \ \/ /\/ /          \/_/\/ /          \/_/_/          \/_/_/
#   \/_/\/_/              \/_/



    