import pyglet
import numpy as np

# using update

class Grid():
    
    def __init__(self, window_width, window_height, batch):

        self.grid_array = np.zeros((6,7), dtype=np.uint8)

        self.window_width, self.window_height = window_width, window_height

        self.mouse_x, self.mouse_y = 0, 0

        self.mouse_state = False
        self.old_mouse_state = False

        self.background = pyglet.shapes.Rectangle(x=0, y=0, color=(60,60,65), width=self.window_width, height=self.window_height, batch=batch)
        self.middle_square = pyglet.shapes.Rectangle(x=self.window_width // 2, y=self.window_height // 2, color=(100,0,0), width=100, height=100, batch=batch)
        self.middle_square.anchor_position = (50, 50)
        self.middle_square_borders = {
            "left": self.window_width // 2 - self.middle_square.width // 2,
            "right": self.window_width // 2 + self.middle_square.width // 2,
            "bottom": self.window_height // 2 - self.middle_square.height // 2,
            "top": self.window_height // 2 + self.middle_square.height // 2
        }

        self.middle_square_state = False

    # def render_grid(self, batch=None):

    #     self.background = pyglet.shapes.Rectangle(x=0, y=0, color=(60,60,65), width=self.window_width, height=self.window_heigth, batch=batch)
    #     self.middle_square = pyglet.shapes.Rectangle(x=350, y=300, color=(100,0,0), width=100, height=100, batch=batch)
    #     self.middle_square.anchor_position = 50, 50

    #     stuff_to_draw = [background, self.middle_square]
    #     return stuff_to_draw

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_x, self.mouse_y = x, y

    def on_mouse_press(self, x, y, button, modifiers):
        self.mouse_state = True

    def on_mouse_release(self, x, y, button, modifiers):
        self.mouse_state = False

    def update(self, dt):
        
        if self.middle_square_borders["left"] <= self.mouse_x <= self.middle_square_borders["right"] and self.middle_square_borders["bottom"] <= self.mouse_y <= self.middle_square_borders["top"]:
            if self.mouse_state != self.old_mouse_state:
                if self.mouse_state:
                    self.middle_square_state = not self.middle_square_state
                self.old_mouse_state = self.mouse_state
            if not self.middle_square_state:
                self.middle_square.color = (200,0,0)
            else:
                self.middle_square.color = (0,200,0)
        else:
            if not self.middle_square_state:
                self.middle_square.color = (150,0,0)
            else:
                self.middle_square.color = (0,150,0)

        if self.mouse_state != self.old_mouse_state:
            if self.mouse_state:
                print("click")
            else:
                print("release")
            self.old_mouse_state = self.mouse_state