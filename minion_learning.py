from ple.games.minion import FlappyBird
from ple import PLE

# local imports
import agent

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
agent = agent.NaiveAgent(allowed_actions=p.getActionSet())

p.init()
reward = 0.0
nb_frames = 10000

for i in range(nb_frames):
   if p.game_over():
           p.reset_game()

   observation = p.getScreenRGB()
   action = agent.pickAction(reward, observation)
   reward = p.act(action)
