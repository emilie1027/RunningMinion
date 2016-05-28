import numpy as np


class NaiveAgent():
	"""
		This is our naive agent. It picks actions at random!
	"""
	def __init__(self, allowed_actions):
		self.actions = allowed_actions

	def pickAction(self, reward, obs):
		return self.actions[np.random.randint(0, len(self.actions))]
