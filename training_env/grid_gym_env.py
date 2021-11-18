from pathlib import Path
import sys
import numpy as np

ROOT_DIR_PATH = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR_PATH not in sys.path:
    sys.path.insert(1, ROOT_DIR_PATH)

from game_content.load import WINNING_GRIDS


class Connect4Env:
    """
    Yellow:  1
    Red:    -1

    Atributes
    ---------
    done: bool
    grid_array: np.ndarray

    Methods
    -------
    reset()             -> state: np.ndarray
    step(action: int)   -> next_state: np.ndarray, reward: float, done: bool, legal_actions: list[int]
    display()           -> None
    """

    def __init__(self):

        # the grid is stored as yellow sees it:
        # 0  are empty spaces
        # 1  are yellow coins
        # -1 are red coins
        self.grid_array = np.zeros((6,7), dtype=np.int8)
        self.done = False
        self.winner = 0
        self.history = []


    def reset(self):
        self.grid_array = np.zeros((6,7), dtype=np.int8)
        self.done = False
        self.winner = 0
        self.history = []
        return self.grid_array.flatten(), self._get_legal_actions()


    def _get_legal_actions(self):
        """
        Checks whether the last line has empty
        places or not
        """
        last_line = self.grid_array[0, :]
        empty_places = list(last_line == 0)
        legal_actions = [index for index, item in enumerate(empty_places) if empty_places[index]]
        return legal_actions


    def _add_coin(self, coin, column_number):
        # coin = 1 or -1
        # isolate the column where to coin is to be added
        column = self.grid_array[:, column_number]
        # get the next empty space for that column
        try:
            next_empty_space_index = max(np.where(column == 0)[0])
        except ValueError as e:
            raise type(e)("Tried to add a coin to a full column.")
        # add the coin there
        column[next_empty_space_index] = coin
        # update game status
        self._check_winner()


    def _check_winner(self):
        for win_grid in WINNING_GRIDS:
            if np.array_equal((self.grid_array == 1) & win_grid, win_grid):
                # print("YELLOW WINS")
                self.winner = 1
                self.done = True
                break
            elif np.array_equal((self.grid_array == -1) & win_grid, win_grid):
                # print("RED WINS")
                self.winner = -1
                self.done = True
                break


    def step(self, player: int, action: int):
        """
        Returns the state as yellow sees it. The only reward so far is
        1 if yellow wins, -1 if yellow loses, 0 if game is still playing.
        """

        if not self.done:
            # add the coin + update status
            self._add_coin(player, action)
            self.history.append(self.grid_array.copy())

        next_state = self.grid_array.flatten()
        reward = self.winner
        legal_actions = self._get_legal_actions()

        # if the board is full, set self.done to True
        if len(legal_actions) == 0:
            self.done = True

        return next_state, reward, self.done, legal_actions


    def display(self):
        print(self.grid_array)


    def display_game_history(self):
        for step in self.history:
            print(step)


if __name__ == "__main__":

    env = Connect4Env()
    while True:
        print(
            "\n"
            + ("#"*50)
            + "\nNEW GAME\n"
            + ("#"*50)
            + "\n"
        )
        env.reset()
        env.display()
        player = 1
        while not env.done:
            _, _, _, legal_actions = env.step(player, int(input("column: ")))
            print(legal_actions)
            env.display()
            player *= -1
        print("game history:")
        env.display_game_history()
