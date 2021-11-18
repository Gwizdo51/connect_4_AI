import numpy as np


class RandomAgent:

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


class HumanAgent:

    def choose_greedy_action(self, state, legal_actions):

        user_input_is_valid = False
        while not user_input_is_valid:
            user_input = input("column:")
            if not user_input.isdigit():
                print("please input a digit.")
            else:
                user_input = int(user_input)
                if user_input not in legal_actions:
                    print(f"please input a legal action. legal actions are {legal_actions}.")
                else:
                    user_input_is_valid = True

        return user_input
