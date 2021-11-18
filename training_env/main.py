import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
from training_env.random_agent import RandomAgent


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
    batch_size=64
    ):

    # Play as yellow, then red, then yellow ...

    env = Connect4Env()
    ddqn_agent = DDQNAgent(
        alpha=alpha,
        gamma=gamma,
        n_actions=7,
        epsilon=1.0,
        epsilon_dec=epsilon_dec,
        batch_size=batch_size,
        input_dims=42,
        weights_file_name="ddqn_connect4_method1.h5"
    )
    random_agent = RandomAgent(n_actions=7)

    ddqn_scores = []
    eps_history = []

    print("setup complete, begin playing")

    for game_id in tqdm(range(n_games * 2)):

        done = False
        first_move = True
        new_state, legal_actions = env.reset()

        if game_id % 2 == 0:
            print("ddqn agent is yellow")
            yellow_player = ddqn_agent
            red_player = random_agent
        else:
            print("ddqn agent is red")
            yellow_player = random_agent
            red_player = ddqn_agent

        current_player = "yellow"

        while not done:
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
                print(f"yellow action: {yellow_action}")
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
                print(f"red action: {red_action}")
                new_state, reward, done, legal_actions = env.step(-1, red_action)
                # store old state/action for learning
                old_red_state = red_state.copy()
                old_red_action = red_action
                # switch players, set first_move to False
                current_player = "yellow"
                first_move = False

        # after game is done
        print(f"game over, winner: {reward}")
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
        print("overall score last 100 games:".ljust(15), avg_score)
        print("#"*100 + "\n")

        if game_id % 10 == 0 and game_id > 0:
            ddqn_agent.save_model()


def play_vs_agent():
    pass


if __name__ == "__main__":
    train_method_1(1)
