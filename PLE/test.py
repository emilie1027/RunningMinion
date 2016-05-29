from ple.games.jumpbird import FlappyBird
from ple.games.waterworld import WaterWorld
from ple import PLE
import numpy as np
import pygame
from pygame.locals import *


class TestAgent():
	def __init__(self, actions):
		self.actions = actions
	def doAction(self,reward,obs):
		#print 'hello'
		for event in pygame.event.get():
			if event.type == KEYDOWN:
				return self.actions[0]
			return None

game = FlappyBird()
#game = WaterWorld()
p = PLE(game, fps=30, display_screen=True)
agent = TestAgent(p.getActionSet())

p.init()
reward = 0.0
nb_frames = 500

for i in range(nb_frames):
	if p.game_over():
		p.reset_game()
	if i%1==0:
		obser = p.getScreenRGB()
		action = agent.doAction(reward,obser)
		reward = p.act(action)