import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
if len(tf.config.list_physical_devices()) == 1:
    print("running on CPU")
else:
    print("running on GPU")
    tf.config.set_logical_device_configuration(
        tf.config.list_physical_devices("GPU")[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
    )

import gym
from gym import wrappers
import numpy as np
from tqdm import tqdm
import copy
import sys
from pathlib import Path

ROOT_DIR_PATH = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR_PATH not in sys.path:
    sys.path.insert(1, ROOT_DIR_PATH)

from ddqn.ddqn_keras import DDQNAgent


if __name__ == "__main__":

    env = gym.make("LunarLander-v2")
    ddqn_agent = DDQNAgent(
        alpha=0.0005,
        gamma=0.99,
        n_actions=4,
        epsilon=1.0,
        batch_size=64,
        input_dims=8
    )
    # ddqn_agent_2 = copy.deepcopy(ddqn_agent)
    # print(vars(ddqn_agent))
    # print()
    # print(vars(ddqn_agent_2))

    n_games = 2
    ddqn_scores = []
    eps_history = []

    env = wrappers.Monitor(env, "./vids/", video_callable=lambda episode_id: True, force=True)

    for game_id in tqdm(range(n_games)):
        done = False
        score = 0
        state = env.reset()
        # print(state)
        # print(state.dtype)
        # break
        state_id = 0
        while not done:
            state_id += 1
            if state_id % 5 == 0:
                print("|"*(state_id // 5), end="\r")
            action = ddqn_agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            score += reward
            ddqn_agent.remember(state, action, reward, new_state, done)
            state = new_state
            ddqn_agent.learn()
            # print("epsilon:".ljust(15), ddqn_agent.epsilon)
        print()

        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0, game_id-100):])
        # print(f"episode {game_id}\nscore {round(score, 2)}\naverage_score {round(avg_score, 2)}")
        print("episode:".ljust(15), game_id)
        print("score:".ljust(15), round(score, 2))
        print("average score:".ljust(15), round(avg_score, 2))

        if game_id % 10 == 0 and game_id > 0:
            ddqn_agent.save_model(file_name="connect_4_model_vs_random.h5")

    # for env_spec in gym.envs.registry.all():
    #     print(env_spec)
