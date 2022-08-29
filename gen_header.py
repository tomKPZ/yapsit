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


def create_colormap(sprite):
    colormap = collections.OrderedDict()
    n, m = sprite.size
    for y in range(m):
        for x in range(n):
            r, g, b, a = sprite.getpixel((x, y))
            if a != 0 and (r, g, b) not in colormap:
                colormap[(r, g, b)] = len(colormap) + 1
    return colormap


for p in os.listdir(SPRITES_DIR):
    fname = os.path.join(SPRITES_DIR, p)
    if not FNAME_PAT.match(p):
        continue
    sprite = PIL.Image.open(fname).convert("RGBA")
    shiny = PIL.Image.open(os.path.join(SHINY_DIR, p)).convert("RGBA")

    colormap = create_colormap(sprite)
    if len(colormap) > 15:
        print("Excess colors in sprite " + fname, file=sys.stderr)
        continue

    xl, yl, xh, yh = sprite.getbbox()
    print("{%d,%d,(uint8_t[]){" % (xh - xl, yh - yl), end="")
    startb = True
    for y in range(yl, yh):
        for x in range(xl, xh):
            r, g, b, a = sprite.getpixel((x, y))
            if a == 0:
                color = 0
            else:
                color = colormap[(r, g, b)]
            if startb:
                print("0x%X" % color, end="")
            else:
                print("%X," % color, end="")
            startb = not startb
    print("},{")
    for color in colormap:
        print("{%s}," % ",".join("0x%02X" % c for c in color), end="")
    print("},{")
    for color in create_colormap(shiny):
        print("{%s}," % ",".join("0x%02X" % c for c in color), end="")
    print("}},")
