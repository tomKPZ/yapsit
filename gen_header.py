#!/usr/bin/env python3

import collections
import os
import re
import sys

import PIL.Image

URL = "https://pokemondb.net/sprites"
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SPRITES_DIR = os.path.join(
    SCRIPT_DIR,
    "sprites",
    "sprites",
    "pokemon",
    "versions",
    "generation-iii",
    "emerald",
)
SHINY_DIR = os.path.join(SPRITES_DIR, "shiny")
FNAME_PAT = re.compile(r"\d{1,3}.png")


def pixel(sprite, x, y):
    r, g, b, a = sprite.getpixel((x, y))
    if not a:
        return 0
    return (a >> 7) << 15 | (r >> 3) << 10 | (g >> 3) << 5 | (b >> 3)


def create_colormap(sprite, shiny):
    colormap = collections.OrderedDict({(0, 0): 0})
    n, m = sprite.size
    for y in range(m):
        for x in range(n):
            color = (pixel(sprite, x, y), pixel(shiny, x, y))
            if color not in colormap:
                colormap[color] = len(colormap)
    return colormap


for p in os.listdir(SPRITES_DIR):
    fname = os.path.join(SPRITES_DIR, p)
    if not FNAME_PAT.match(p):
        continue
    sprite = PIL.Image.open(fname).convert("RGBA")
    shiny = PIL.Image.open(os.path.join(SHINY_DIR, p)).convert("RGBA")

    colormap = create_colormap(sprite, shiny)
    if len(colormap) > 16:
        print("Excess colors in sprite", fname, file=sys.stderr)
        continue

    xl, yl, xh, yh = sprite.getbbox()
    print("{%d,%d,(uint8_t[]){" % (xh - xl, yh - yl), end="")
    startb = True
    for y in range(yl, yh):
        for x in range(xl, xh):
            color = (pixel(sprite, x, y), pixel(shiny, x, y))
            color = colormap[color]
            print(("0x%X" if startb else "%X,") % color, end="")
            startb = not startb
    print("},{")
    for (color, _) in colormap:
        print("0x%04X," % color, end="")
    print("},{")
    for (_, color) in colormap:
        print("0x%04X," % color, end="")
    print("}},")
