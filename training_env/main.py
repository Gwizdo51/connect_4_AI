import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import numpy as np

if len(tf.config.list_physical_devices()) == 1:
    print("running on CPU")
else:
    print("running on GPU")
    tf.config.set_logical_device_configuration(
        tf.config.list_physical_devices("GPU")[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
    )

ROOT_DIR_PATH = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR_PATH not in sys.path:
    sys.path.insert(1, ROOT_DIR_PATH)

from training_env.grid_gym_env import Connect4Env
from ddqn.ddqn_keras import DDQNAgent
from training_env.utils_agents import RandomAgent, HumanAgent


"""
Training methods:

1- 2 agents, one versus the other
2- 1 agent versus random agent
3- agent against himself
4- agent against kaggle negamax
5- start with random agent and relace it with copy of agent every X games
"""


def train_method_1(
    n_games,
    alpha=0.0005,
    gamma=0.99,
    epsilon_dec=0.996,
    epsilon_min=0.01,
    batch_size=64,
    replace_target_interval=100,
    verbose=False
    ):

    # Play as yellow, then red, then yellow ...

    env = Connect4Env()
    ddqn_agent = DDQNAgent(
        alpha=alpha,
        gamma=gamma,
        n_actions=7,
        epsilon=1.0,
        epsilon_dec=epsilon_dec,
        epsilon_min=epsilon_min,
        batch_size=batch_size,
        input_dims=42,
        replace_target_interval=replace_target_interval,
        weights_file_name="temp.h5"
    )
    random_agent = RandomAgent(n_actions=7)

    ddqn_scores = []
    eps_history = []

    print("setup complete, begin playing") if verbose else ...

    for game_id in tqdm(range(n_games * 2)):

        done = False
        first_move = True
        new_state, legal_actions = env.reset()

        if game_id % 2 == 0:
            print("ddqn agent is yellow") if verbose else ...
            yellow_player = ddqn_agent
            red_player = random_agent
        else:
            print("ddqn agent is red") if verbose else ...
            yellow_player = random_agent
            red_player = ddqn_agent

        current_player = "yellow"

        while not done:
            if verbose:
                print(f"first move: {first_move}")
                print(f"current player: {current_player}")
                print("current state:")
                env.display()
            # yellow turn
            if current_player == "yellow":
                yellow_state = new_state.copy()
                if not first_move:
                    # remember
                    yellow_player.remember(old_yellow_state, old_yellow_action, reward, yellow_state, False)
                    # learn
                    yellow_player.learn()
                # play
                yellow_action = yellow_player.choose_action(yellow_state, legal_actions)
                print(f"yellow action: {yellow_action}") if verbose else ...
                new_state, reward, done, legal_actions = env.step(1, yellow_action)
                # store old state/action for learning
                old_yellow_state = yellow_state.copy()
                old_yellow_action = yellow_action
                # switch players
                current_player = "red"
            # red turn
            elif current_player == "red":
                red_state = new_state.copy() * -1
                if not first_move:
                    # remember
                    red_player.remember(old_red_state, old_red_action, reward*-1, red_state, False)
                    # learn
                    red_player.learn()
                # play
                red_action = red_player.choose_action(red_state, legal_actions)
                print(f"red action: {red_action}") if verbose else ...
                new_state, reward, done, legal_actions = env.step(-1, red_action)
                # store old state/action for learning
                old_red_state = red_state.copy()
                old_red_action = red_action
                # switch players, set first_move to False
                current_player = "yellow"
                first_move = False

        # after game is done
        print(f"game over, winner: {reward}") if verbose else ...
        yellow_state = new_state.copy()
        yellow_player.remember(old_yellow_state, old_yellow_action, reward, yellow_state, False)
        yellow_player.learn()
        red_state = new_state.copy() * -1
        red_player.remember(old_red_state, old_red_action, reward*-1, red_state, False)
        red_player.learn()

        print("\n" + "#"*100)
        eps_history.append(ddqn_agent.epsilon)
        print("episode:".ljust(15), game_id)
        if game_id % 2 == 0:
            ddqn_scores.append(reward)
            print("score:".ljust(15), reward)
        else:
            ddqn_scores.append(-reward)
            print("score:".ljust(15), -reward)
        avg_score = sum(ddqn_scores[max(0, game_id-100):])
        print("average score:".ljust(15), avg_score)
        print("epsilon".ljust(15), ddqn_agent.epsilon)
        print("#"*100 + "\n")

        if game_id % 50 == 0 and game_id > 0:
            ddqn_agent.save_model()

    ddqn_agent.save_model()


def play_vs_agent(yellow_player_type="human", red_player_type="human", yellow_player_model_file_name=None, red_player_model_file_name=None):

    # default parameters, useless for inference
    alpha = 0.0005
    gamma = 0.99
    epsilon_dec = 0.996
    epsilon_min = 0.01
    batch_size = 64
    replace_target_interval = 100

    # load the players
    if yellow_player_type == "model":
        yellow_player = DDQNAgent(
            alpha=alpha,
            gamma=gamma,
            n_actions=7,
            epsilon=0,
            epsilon_dec=epsilon_dec,
            epsilon_min=epsilon_min,
            batch_size=batch_size,
            input_dims=42,
            replace_target_interval=replace_target_interval,
            weights_file_name=yellow_player_model_file_name
        )
        yellow_player.load_model()
    elif yellow_player_type == "human":
        yellow_player = HumanAgent()
    else:
        raise ValueError("wrong input for yellow_player_type.")

    if red_player_type == "model":
        red_player = DDQNAgent(
            alpha=alpha,
            gamma=gamma,
            n_actions=7,
            epsilon=0,
            epsilon_dec=epsilon_dec,
            epsilon_min=epsilon_min,
            batch_size=batch_size,
            input_dims=42,
            replace_target_interval=replace_target_interval,
            weights_file_name=red_player_model_file_name
        )
        red_player.load_model()
    elif red_player_type == "human":
        red_player = HumanAgent()
    else:
        raise ValueError("wrong input for red_player_type.")

    # begin playing
    env = Connect4Env()
    while True:

        print("new game")
        done = False
        new_state, legal_actions = env.reset()
        env.display()

        current_player = "yellow"

        while not done:
            if current_player == "yellow":
                yellow_state = new_state.copy()
                # play
                yellow_action = yellow_player.choose_greedy_action(yellow_state, legal_actions)
                print(f"yellow action: {yellow_action}")
                new_state, reward, done, legal_actions = env.step(1, yellow_action)
                # switch players
                current_player = "red"
            # red turn
            elif current_player == "red":
                red_state = new_state.copy() * -1
                # play
                red_action = red_player.choose_greedy_action(red_state, legal_actions)
                print(f"red action: {red_action}")
                new_state, reward, done, legal_actions = env.step(-1, red_action)
                # switch players
                current_player = "yellow"
            env.display()

        if reward == 1:
            print("YELLOW WINS")
        elif reward == -1:
            print("RED WINS")



if __name__ == "__main__":

    # train_method_1(
    #     1000,
    #     epsilon_dec=(1 - np.exp(-7)),
    #     epsilon_min=0.05
    # )
    play_vs_agent(
        yellow_player_type="model",
        red_player_type="human",
        yellow_player_model_file_name="ddqn_connect4_method1.h5"
    )
