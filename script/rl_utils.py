from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    #Like DQN, DDPG has a replay buffer stores experience and samples them to train the neural network. 
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

def moving_average(a, window_size):

    '''
    This is a simple function to calculate the
    :param a:
    :param window_size:
    :return:
    '''

    cumulative_sum = np.cumsum(a)
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    return middle



def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):

    '''
    This is the main function to train the algorithm and return action, reward, state list
    :param env: asset allocation environment which includes initial wealth, up probability, etc.
    :param agent: the agent with DDPG algorithm
    :param num_episodes: number of episodes
    :param replay_buffer: replay buffer is used to specify how we want to retrain model
    :param minimal_size: minimum number of samples required
    :param batch_size: number of sample to simulate at each training sample
    :return: action_list, return_list, whole_list, whole_list store :state action reward next_state
    '''

    return_list = []
    action_list = []
    whole_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    next_states=torch.tensor(next_state, dtype=torch.float)
                    rewards=torch.tensor(reward, dtype=torch.float).view(-1, 1)
                    replay_buffer.add(state, action, reward, next_state, done)
                    whole_list.append([state.item(),action.item(),reward,next_state.item(),done])
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                    action_list.append(action)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list,action_list,whole_list