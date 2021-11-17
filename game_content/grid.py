import pyglet.shapes as shapes
import numpy as np
from pathlib import Path
import sys

ROOT_DIR_PATH = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR_PATH not in sys.path:
    sys.path.insert(1, ROOT_DIR_PATH)

from game_content.grid_column import Column
from game_content.load import BRIGHT_COLOR, YELLOW_COIN, YELLOW_HOVER, \
                              RED_COIN, RED_HOVER, WINNING_GRIDS


# color : (0,0,0) is black


class Grid():

    def __init__(self, window_width, window_height, batch):

        self.grid_array = np.zeros((6,7), dtype=np.int8)
        # print(self.grid_array)

        self.window_width, self.window_height = window_width, window_height

        self.mouse_x, self.mouse_y = 350, 0

        # self.mouse_state = False
        # self.old_mouse_state = False

        # background
        self.background = shapes.Rectangle(x=0, y=0, color=(60,60,65), width=self.window_width, height=self.window_height, batch=batch)

        # create column object
        # on hover, highlight the slot that will be filled
        # on click, add a colored coin on the column and modify self.grid_array
        # verify that self.grid_array doesn't have a winning pattern
        # if self.grid_array has a winning pattern, stop the game and highlight the
        # 4 connected coins
        self.grid_columns = [Column(col, batch) for col in range(7)]

        # tests
        # self.grid_columns[2].slots_circles[3].color = (0,0,0)

        # True: yellow, False: red
        self.next_coin = True

        # 0: no win, 1: yellow won, 2: red won
        self.winner = 0

        # timer
        self.timer = 0.


    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_x, self.mouse_y = x, y


    def on_mouse_press(self, x, y, button, modifiers):

        if self.winner == 0:

            # add a coin
            old_coin = self.next_coin
            self.next_coin, self.grid_array = self.grid_columns[int(x / 100)].add_coin(self.next_coin, self.grid_array)

            # check if there is a win, only if a coin was added
            if self.next_coin != old_coin:
                for win_grid in WINNING_GRIDS:
                    if np.array_equal((self.grid_array == 1) & win_grid, win_grid):
                        print("YELLOW WINS")
                        self.winner = 1
                        self.winner_positions = win_grid
                        break
                    elif np.array_equal((self.grid_array == -1) & win_grid, win_grid):
                        print("RED WINS")
                        self.winner = 2
                        self.winner_positions = win_grid
                        break


    # def on_mouse_release(self, x, y, button, modifiers):
    #     self.mouse_state = False


    def on_resize(self, width, height):
        # print("resizing")
        # self.background.width = width
        # self.background.height = height
        pass


    def highlight_winner(self, dt):
        self.timer += dt
        if self.timer < .5:
            # dim the winning positions
            for line in range(self.winner_positions.shape[0]):
                for col in range(self.winner_positions.shape[1]):
                    if self.winner_positions[line, col]:
                        self.grid_columns[col].slots_circles[line].color = YELLOW_HOVER if self.winner == 1 else RED_HOVER
        elif self.timer < 1:
            # highlight the winning positions
            for line in range(self.winner_positions.shape[0]):
                for col in range(self.winner_positions.shape[1]):
                    if self.winner_positions[line, col]:
                        self.grid_columns[col].slots_circles[line].color = YELLOW_COIN if self.winner == 1 else RED_COIN
        else:
            self.timer = 0


    def update(self, dt):

        if self.winner == 0:
            # highlight next coin position
            for column in self.grid_columns:
                column.mouse_hover(self.mouse_x, self.next_coin)
        else:
            # highlight the winning connect 4
            self.highlight_winner(dt)


if __name__ == "__main__":
    print("hello world!")
