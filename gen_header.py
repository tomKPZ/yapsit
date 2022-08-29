#!/usr/bin/env python3

import collections
import os
import re
import sys

from matplotlib import image

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
FNAME_PAT = re.compile(r"\d{1,3}.png")

for p in os.listdir(SPRITES_DIR):
    fname = os.path.join(SPRITES_DIR, p)
    if not FNAME_PAT.match(os.path.basename(fname)):
        continue
    sprite = image.imread(fname)

    n, m, _ = sprite.shape
    inf = float("inf")
    xl, yl, xh, yh = inf, inf, -inf, -inf
    for y in range(n):
        for x in range(m):
            if sprite[y][x][3]:
                xl = min(xl, x)
                yl = min(yl, y)
                xh = max(xh, x)
                yh = max(yh, y)

    colormap = collections.OrderedDict()
    w = xh - xl + 1
    h = yh - yl + 1
    print("{%d,%d,(uint8_t[]){" % (w, h), end="")
    startb = True
    for y in range(yl, yh + 1):
        for x in range(xl, xh + 1):
            r, g, b, a = (round(v * 255) for v in sprite[y][x])
            if a == 0:
                color = 0
            elif (r, g, b) in colormap:
                color = colormap[(r, g, b)]
            else:
                colormap[(r, g, b)] = len(colormap) + 1
                color = colormap[(r, g, b)]
            if color > 15:
                print("Invalid color in sprite " + fname, file=sys.stderr)
                color = 0
            if startb:
                print("0x%X" % color, end="")
            else:
                print("%X," % color, end="")
            startb = not startb
    print("},{")
    for channel in range(3):
        print("(uint8_t[]){")
        for color in colormap:
            print("0x%02X," % color[channel], end="")
        print("},")
    print("}},")
