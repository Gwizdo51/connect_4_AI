import pyglet

window = pyglet.window.Window(width=700, height=700)

label = pyglet.text.Label(
    text = "This is text",
    font_name = "Helvetica",
    font_size = 18,
    x = window.width // 2,
    y = window.height // 2,
    anchor_x = "center",
    anchor_y = "center"
)

@window.event
def on_draw():
    window.clear()
    label.draw()

pyglet.app.run()