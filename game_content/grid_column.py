import pyglet
import pyglet.shapes as shapes

class Column():

    def __init__(self, column_number, batch, bright_color, yellow_hover, yellow_coin, red_hover, red_coin):

        self.column_number = column_number

        self.bright_color = bright_color
        self.yellow_hover = yellow_hover
        self.yellow_coin = yellow_coin
        self.red_hover = red_hover
        self.red_coin = red_coin

        # create a list of circles to represent the slots
        # 5: bottom
        # 0: top
        self.slots_circles = [shapes.Circle(column_number*100 + 50, row*100 + 50, 40, color=self.bright_color, batch=batch) for row in range(6)][::-1]

        self.next_empty = 5

    def mouse_hover(self, mouse_x, next_coin):
        # if mouse_x is hovering this column, highlight the next empty slot with
        # the next color ; otherwise, reset to default color
        if self.next_empty >= 0:
            if self.column_number * 100 <= mouse_x < (self.column_number + 1) * 100:
                self.slots_circles[self.next_empty].color = self.yellow_hover if next_coin else self.red_hover
            else:
                self.slots_circles[self.next_empty].color = self.bright_color

    def add_coin(self, next_coin, grid_array):
        # print(f"added coin to column {self.column_number}")
        if self.next_empty >= 0:
            # 1: yellow, 2: red
            grid_array[self.next_empty, self.column_number] = 1 if next_coin else 2
            self.slots_circles[self.next_empty].color = self.yellow_coin if next_coin else self.red_coin
            self.next_empty -= 1
            next_coin = not next_coin
        return next_coin, grid_array
