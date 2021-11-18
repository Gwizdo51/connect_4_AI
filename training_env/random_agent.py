import numpy as np


class RandomAgent():

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]


    def choose_action(self, state: np.ndarray, legal_actions=None):
        if legal_actions is None:
            action = np.random.choice(self.action_space)
        else:
            action = np.random.choice(legal_actions)
        return action


    # for compatibility with ddqn agent
    def remember(self, state, action, reward, new_state, done):
        pass

    def learn(self):
        pass
