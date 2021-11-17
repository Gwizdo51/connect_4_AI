from keras.layers import Dense, Activation, Input
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np

from keras import Model, Sequential


class ReplayBuffer:

    def __init__(self, max_size, input_shape, n_actions, discrete=False):

        self.mem_size = max_size
        self.mem_counter = 0
        self.discrete = discrete

        self.state_memory = np.zeros(shape=(self.mem_size, input_shape))
        self.new_state_memory = np.zeros(shape=(self.mem_size, input_shape))
        action_memory_dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros(
            shape=(self.mem_size, n_actions),
            dtype=action_memory_dtype
        )
        self.reward_memory = np.zeros(shape=(self.mem_size,))
        self.terminal_memory = np.zeros(shape=(self.mem_size,), dtype=np.float32)


    def store_transition(self, state, action, reward, new_state, done):

        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state

        if self.discrete:
            actions = np.zeros(shape=(self.action_memory.shape[1],))
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action

        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)

        self.mem_counter += 1


    def sample_buffer(self, batch_size):

        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal


def build_dqn(learning_rate, n_actions, input_dims, fc1_dims, fc2_dims):

    input = Input(shape=(input_dims,))
    main = Dense(fc1_dims, activation="relu")(input)
    main = Dense(fc2_dims, activation="relu")(main)
    output = Dense(n_actions)(main)
    model = Model(input, output)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
    )

    model.summary()
    return model


class DDQNAgent:

    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims,
                 epsilon_dec=0.996, epsilon_min=0.01, mem_size=int(1e6),
                 file_name="ddqn_model.h5", replace_target=100):
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        # n_actions = 5 => action_space = [0, 1, 2, 3, 4]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_file = file_name
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)
        self.q_target = build_dqn(alpha, n_actions, input_dims, 256, 256)


    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)


    def choose_action(self, state: np.ndarray, legal_actions=None):

        # state = state[np.newaxis, :]
        state = state.reshape(1, -1)
        rand = np.random.random()

        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action


    def choose_greedy_action(self, state: np.ndarray, legal_actions=None):
        state = state.reshape(1, -1)
        actions = self.q_eval.predict(state)
        print(actions.shape)
        action = np.argmax(actions)
        print(action.shape)
        return action


    def learn(self):

        if self.memory.mem_counter > self.batch_size:

            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)

            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            self.q_eval.fit(state, q_target, verbose=0)

            self.epsilon = max(self.epsilon*self.epsilon_dec, self.epsilon_min)

            if self.memory.mem_counter % self.replace_target == 0:
                self.update_network_parameters()


    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())


    def save_model(self):
        self.q_eval.save(self.model_file)


    def load_model(self):
        self.q_eval = load_model(self.model_file)
        if self.epsilon <= self.epsilon_min:
            self.update_network_parameters()


if __name__ == "__main__":
    build_dqn(0.0001, 20, 10, 256, 256)
