import pyglet
import numpy as np

# same but not using update

class Grid():
    
    def __init__(self, window_width, window_height, batch):

        self.grid_array = np.zeros((6,7), dtype=np.uint8)

        self.window_width, self.window_height = window_width, window_height

        self.background = pyglet.shapes.Rectangle(x=0, y=0, color=(60,60,65), width=self.window_width, height=self.window_height, batch=batch)
        self.middle_square = pyglet.shapes.Rectangle(x=self.window_width // 2, y=self.window_height // 2, color=(150,0,0), width=100, height=100, batch=batch)
        self.middle_square.anchor_position = (50, 50)
        self.middle_square_borders = {
            "left": self.window_width // 2 - self.middle_square.width // 2,
            "right": self.window_width // 2 + self.middle_square.width // 2,
            "bottom": self.window_height // 2 - self.middle_square.height // 2,
            "top": self.window_height // 2 + self.middle_square.height // 2
        }

        self.middle_square_state = False

    def on_mouse_motion(self, x, y, dx, dy):
        if self.middle_square_borders["left"] <= x <= self.middle_square_borders["right"] and self.middle_square_borders["bottom"] <= y <= self.middle_square_borders["top"]:
            if not self.middle_square_state:
                self.middle_square.color = (200,0,0)
            else:
                self.middle_square.color = (0,200,0)
        else:
            if not self.middle_square_state:
                self.middle_square.color = (150,0,0)
            else:
                self.middle_square.color = (0,150,0)

    def on_mouse_press(self, x, y, button, modifiers):
        if self.middle_square_borders["left"] <= x <= self.middle_square_borders["right"] and self.middle_square_borders["bottom"] <= y <= self.middle_square_borders["top"]:
            self.middle_square_state = not self.middle_square_state
            if not self.middle_square_state:
                self.middle_square.color = (200,0,0)
            else:
                self.middle_square.color = (0,200,0)

    def on_mouse_release(self, x, y, button, modifiers):
        pass

    def update(self, dt):
        pass