import logging
import numpy as np
import matplotlib.pyplot as plt

# PLE imports
#from ple.games.minion import FlappyBird
from ple.games.jumpbird import FlappyBird
from ple import PLE

# local imports
import utils
import naive
import agent


def process_state(state):
    state = np.array([state.values()])
    # state = state[0, 0:4]

    #max_values = np.array([288.0, 50.0, 288.0, 512.0, 512.0, 288, 512.0, 512.0])
    max_values = np.array([512.0, 50.0, 512.0, 512.0, 512.0])
    state = state / max_values
    # print state

    return state


def init_agent(env):
    # agent settings
    batch_size = 32
    num_frames = 4  # number of frames in a 'state'
    frame_skip = 2
    lr = 0.01
    discount = 0.95  # discount factor
    rng = np.random.RandomState(24)

    # my_agent = naive.NaiveAgent(allowed_actions=env.getActionSet())
    my_agent = agent.Agent(env, batch_size, num_frames, frame_skip, lr, discount, rng, optimizer="sgd_nesterov")
    # my_agent = utils.DQNAgent(env, batch_size, num_frames, frame_skip, lr, discount, rng, optimizer="sgd_nesterov")
    my_agent.build_model()

    return my_agent


def plot_figure(data, label, name):
    plt.plot(data)

    plt.xlabel('episode')
    plt.ylabel(label)
    # plt.title('title')
    # plt.grid(True)
    plt.savefig('../figures/'+name+'.png')
    # plt.show()


def main_naive():
    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=True)
    my_agent = naive.NaiveAgent(allowed_actions=env.getActionSet())

    env.init()
    reward = 0.0
    nb_frames = 10000

    for i in range(nb_frames):
        if env.game_over():
            env.reset_game()

        observation = env.getScreenRGB()
        action = my_agent.pickAction(reward, observation)
        reward = env.act(action)


def main():
    # training parameters
    num_epochs = 5
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
    env = PLE(game, fps=30, display_screen=True, force_fps=True, state_preprocessor=process_state)
    my_agent = init_agent(env)

    memory = utils.ReplayMemory(max_memory_size, min_memory_size)
    env.init()

    # Logging configuration and figure plotting
    logging.basicConfig(filename='../learning.log', filemode='w',
                        level=logging.DEBUG, format='%(levelname)s:%(message)s')
    logging.info('Training started.\n')
    learning_rewards = list()
    testing_rewards = list()

    for epoch in range(1, num_epochs + 1):
        steps, num_episodes = 0, 0
        losses, rewards = [], []
        env.display_screen = False

        # training loop
        while steps < num_steps_train:
            episode_reward = 0.0
            my_agent.start_episode()

            while env.game_over() == False and steps < num_steps_train:
                state = env.getGameState()
                reward, action = my_agent.act(state, epsilon=epsilon)
                memory.add([state, action, reward, env.game_over()])

                if steps % update_frequency == 0:
                    loss = memory.train_agent_batch(my_agent)

                    if loss is not None:
                        losses.append(loss)
                        epsilon = np.max(epsilon_min, epsilon - epsilon_rate)

                episode_reward += reward
                steps += 1
                learning_rewards.append(reward)

            if num_episodes % 5 == 0:
                print "Episode {:01d}: Reward {:0.1f}".format(num_episodes, episode_reward)
                logging.info("Episode {:01d}: Reward {:0.1f}".format(num_episodes, episode_reward))

            rewards.append(episode_reward)
            num_episodes += 1
            my_agent.end_episode()

        print "Train Epoch {:02d}: Epsilon {:0.4f} | Avg. Loss {:0.3f} | Avg. Reward {:0.3f}\n"\
            .format(epoch, epsilon, np.mean(losses), np.sum(rewards) / num_episodes)
        logging.info("Train Epoch {:02d}: Epsilon {:0.4f} | Avg. Loss {:0.3f} | Avg. Reward {:0.3f}\n"
                     .format(epoch, epsilon, np.mean(losses), np.sum(rewards) / num_episodes))

        steps, num_episodes = 0, 0
        losses, rewards = [], []

        # display the screen
        # env.display_screen = True

        # slow it down so we can watch it fail!
        # env.force_fps = True

        # testing loop
        while steps < num_steps_test:
            episode_reward = 0.0
            my_agent.start_episode()

            while env.game_over() == False and steps < num_steps_test:
                state = env.getGameState()
                reward, action = my_agent.act(state, epsilon=0.05)

                episode_reward += reward
                testing_rewards.append(reward)
                steps += 1

                # done watching after 500 steps.
                if steps > 500:
                    env.display_screen = False

            if num_episodes % 5 == 0:
                print "Episode {:01d}: Reward {:0.1f}".format(num_episodes, episode_reward)
                logging.info("Episode {:01d}: Reward {:0.1f}".format(num_episodes, episode_reward))

            rewards.append(episode_reward)
            num_episodes += 1
            my_agent.end_episode()

        print "Test Epoch {:02d}: Best Reward {:0.3f} | Avg. Reward {:0.3f}\n"\
            .format(epoch, np.max(rewards), np.sum(rewards) / num_episodes)
        logging.info("Test Epoch {:02d}: Best Reward {:0.3f} | Avg. Reward {:0.3f}\n"
                     .format(epoch, np.max(rewards), np.sum(rewards) / num_episodes))

    logging.info("\nTraining complete.")
    plot_figure(learning_rewards, 'reward', 'reward_in_training')
    plot_figure(testing_rewards, 'reward', 'reward_in_testing')

    print "\nTraining complete. Will loop forever playing!"
    utils.loop_play_forever(env, my_agent)


if __name__ == '__main__':
    main()
