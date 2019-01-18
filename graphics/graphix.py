import pygame as pg
import random
width = 640
height = 1240
screen = pg.display.set_mode((width, height))
circlesize = 300

running = 1

while running:
    event = pg.event.poll()
    if event.type == pg.QUIT:
         running = 0
    screen.fill((0, 0, 0))    
    w = circlesize + random.randint(-5, 5)
    h = circlesize + random.randint(-5, 5)
    pg.draw.ellipse(screen, (0,0,250), [width / 2 - w / 2, height / 2 - h / 2, w, h], 0)
    
    pg.display.flip()