import pyglet.shapes as shapes
from pathlib import Path
import sys

ROOT_DIR_PATH = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR_PATH not in sys.path:
    sys.path.insert(1, ROOT_DIR_PATH)

from game_content.load import BRIGHT_COLOR, YELLOW_COIN, YELLOW_HOVER, \
                              RED_COIN, RED_HOVER

class Column():

    def __init__(self, column_number, batch):

        self.column_number = column_number

        # create a list of circles to represent the slots
        # 5: bottom
        # 0: top
        self.slots_circles = [shapes.Circle(column_number*100 + 50, row*100 + 50, 40, color=BRIGHT_COLOR, batch=batch) for row in range(6)][::-1]

        self.next_empty = 5


    def mouse_hover(self, mouse_x, next_coin):
        # if mouse_x is hovering this column, highlight the next empty slot with
        # the next color ; otherwise, reset to default color
        if self.next_empty >= 0:
            if self.column_number * 100 <= mouse_x < (self.column_number + 1) * 100:
                self.slots_circles[self.next_empty].color = YELLOW_HOVER if next_coin else RED_HOVER
            else:
                self.slots_circles[self.next_empty].color = BRIGHT_COLOR


    def add_coin(self, next_coin, grid_array):
        # print(f"added coin to column {self.column_number}")
        if self.next_empty >= 0:
            # 1: yellow, -1: red
            grid_array[self.next_empty, self.column_number] = 1 if next_coin else -1
            self.slots_circles[self.next_empty].color = YELLOW_COIN if next_coin else RED_COIN
            self.next_empty -= 1
            next_coin = not next_coin
        return next_coin, grid_array
