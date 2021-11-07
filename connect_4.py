import pyglet
from pyglet.window import key

from game_content.grid import Grid


class MyWindow(pyglet.window.Window):

    def __init__(self, *args, **kwargs):

        super(MyWindow, self).__init__(*args, **kwargs)

        self.grid_batch = pyglet.graphics.Batch()
        self.grid = Grid(self.width, self.height, self.grid_batch)
        self.push_handlers(self.grid)

    def on_key_press(self, symbol, modifiers):

        if symbol == key.R:

            self.grid_batch = pyglet.graphics.Batch()
            self.grid = Grid(self.width, self.height, self.grid_batch)
            self.push_handlers(self.grid)

        elif symbol == key.ESCAPE:
            # exit()
            self.close()

    # def on_key_release(self, symbol, modifiers):
    #     pass

    # def on_mouse_motion(self, x, y, dx, dy):
    #     pass

    # def on_mouse_press(self, x, y, button, modifiers):
    #     pass

    # def on_mouse_release(self, x, y, button, modifiers):
    #     pass

    def on_draw(self):
        self.clear()
        self.grid_batch.draw()

    def update(self, dt):
        self.grid.update(dt)


if __name__ == "__main__":
    frame_rate = 30
    window = MyWindow(700, 600, "Connect 4", resizable=False)
    # print(window.grid.grid_array)
    pyglet.clock.schedule_interval(window.update, 1/frame_rate)
    pyglet.app.run()
