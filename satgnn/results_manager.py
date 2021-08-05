from collections import namedtuple
import pickle
from utils import make_eval_summary

Step = namedtuple('Step', ('obs_reward', 'best_reward', 'actual_reward', 'action', 'color', 'qval'))

class TrainingResults():

	def __init__(self, args):
		self.args = args
		self.episodes = []
		self.val_socres = {}
		self.best_model_step = 0
		self.last_model_eval_score = 0
		self.best_model_eval_score = 0
		self.best_val_score = 0
		self.train_time = 0.
		self.solved_avg = {}
	

	class EpisodeData():
		def __init__(self, iteration, initial_value):
			self.iteration = iteration
			self.initial_value = initial_value
			self.best_value = 0
			self.steps = []
			self.length = 0

		def add_step(self, obs_reward, best_reward, actual_reward, action, color, qval=None):
			self.length += 1
			if best_reward > self.best_value:
				self.best_value = best_reward

			self.steps.append(Step(obs_reward, best_reward, actual_reward, action, color, qval))


	def make_episode_record(self, iteration, initial_value):
		episode = self.EpisodeData(iteration, initial_value)
		self.episodes.append(episode)
		return episode


	def add_val_scores(self, iteration, score):
		if score[0] > self.best_val_score:
			self.best_val_score = score[0]
			self.best_model_step = iteration

		self.val_socres[iteration] = score


	def add_solved_avg(self, iteration, score):
		self.solved_avg[iteration] = score


	def save(self):
		with open(self.args.save_path+'results.pickle', 'wb') as handle:
			pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

		





