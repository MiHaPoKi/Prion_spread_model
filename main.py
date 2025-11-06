import pygame as graphic
import numpy as data
import os

# Example file showing a circle moving on screen

# graphic setup
graphic.init()
screen = graphic.display.set_mode((1280, 720))
clock = graphic.time.Clock()
running = True
dt = 0
spread = 1

player_pos = graphic.Vector2(screen.get_width() / 2, screen.get_height() / 2)

while running:
    # poll for events
    # graphic.QUIT event means the user clicked X to close your window
    for event in graphic.event.get():
        if event.type == graphic.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill(graphic.Color("#FFFFFF"))


    armavir = graphic.image.load(os.path.join("stuff", "Armavir_in_Armenia.png"))
    fon = screen.blit(armavir, (70, -220))
    graphic.draw.circle(screen, "red", player_pos, spread)

    keys = graphic.key.get_pressed()
    if keys[graphic.K_w]:
        spread *= 1.1 #* dt
    if keys[graphic.K_s]:
        spread /= 1.1 #* dt


    # flip() the display to put your work on screen
    graphic.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) #/ 1000

graphic.quit()