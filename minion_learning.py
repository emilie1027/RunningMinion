import numpy as np
from ple.games.minion import FlappyBird
from ple import PLE

# local imports
import utils
import naive
import agent


def process_state(state):
    return np.array([ state.values() ])


def init_agent(env):
    # agent settings
    batch_size = 32
    num_frames = 4  # number of frames in a 'state'
    frame_skip = 2
    lr = 0.01
    discount = 0.95  # discount factor
    rng = np.random.RandomState(24)

    # myAgent = naive.NaiveAgent(allowed_actions=env.getActionSet())
    myAgent = agent.Agent(env, batch_size, num_frames, frame_skip, lr, discount, rng, optimizer="sgd_nesterov")
    myAgent.build_model()

    return myAgent


def main():
    # training parameters
    num_epochs = 15
    num_steps_train = 15000  # steps per epoch of training
    num_steps_test = 3000
    update_frequency = 4  # step frequency of model training/updates

    epsilon = 0.15  # percentage of time we perform a random action, help exploration.
    epsilon_steps = 30000  # decay steps
    epsilon_min = 0.1
    epsilon_rate = (epsilon - epsilon_min) / epsilon_steps

    # memory settings
    max_memory_size = 100000
    min_memory_size = 1000  # number needed before model training starts


    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=True, state_preprocessor=process_state)
    myAgent = init_agent(env)

    memory = utils.ReplayMemory(max_memory_size, min_memory_size)
    env.init()


    for epoch in range(1, num_epochs + 1):
        steps, num_episodes = 0, 0
        losses, rewards = [], []
        # env.display_screen = False

        # training loop
        while steps < num_steps_train:
            episode_reward = 0.0
            myAgent.start_episode()

            while env.game_over() == False and steps < num_steps_train:
                state = env.getGameState()
                reward, action = myAgent.act(state, epsilon=epsilon)
                memory.add([state, action, reward, env.game_over()])

                if steps % update_frequency == 0:
                    loss = memory.train_agent_batch(myAgent)

                    if loss is not None:
                        losses.append(loss)
                        epsilon = np.max(epsilon_min, epsilon - epsilon_rate)

                episode_reward += reward
                steps += 1

            if num_episodes % 5 == 0:
                print "Episode {:01d}: Reward {:0.1f}".format(num_episodes, episode_reward)

            rewards.append(episode_reward)
            num_episodes += 1
            myAgent.end_episode()

        print "\nTrain Epoch {:02d}: Epsilon {:0.4f} | Avg. Loss {:0.3f} | Avg. Reward {:0.3f}".format(epoch, epsilon, np.mean(losses), np.sum(rewards) / num_episodes)

        steps, num_episodes = 0, 0
        losses, rewards = [], []

        # display the screen
        env.display_screen = True

        # slow it down so we can watch it fail!
        env.force_fps = False

        # testing loop
        while steps < num_steps_test:
            episode_reward = 0.0
            myAgent.start_episode()

            while env.game_over() == False and steps < num_steps_test:
                state = env.getGameState()
                reward, action = myAgent.act(state, epsilon=0.05)

                episode_reward += reward
                steps += 1

                # done watching after 500 steps.
                if steps > 500:
                    env.force_fps = True
                    # env.display_screen = False

            if num_episodes % 5 == 0:
                print "Episode {:01d}: Reward {:0.1f}".format(num_episodes, episode_reward)

            rewards.append(episode_reward)
            num_episodes += 1
            myAgent.end_episode()

        print "Test Epoch {:02d}: Best Reward {:0.3f} | Avg. Reward {:0.3f}".format(epoch, np.max(rewards), np.sum(rewards) / num_episodes)


    # reward = 0.0
    # nb_frames = 10000
    #
    # for i in range(nb_frames):
    #    if env.game_over():
    #        env.reset_game()
    #
    #    state = env.getGameState()
    #    action = myAgent.pickAction(reward, state)
    #    reward = env.act(action)


if __name__ == '__main__':
    main()
