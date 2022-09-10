#!/usr/bin/env python3

import collections
import os
import sys

import PIL.Image

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
VERSIONS_DIR = os.path.join(
    SCRIPT_DIR,
    "sprites",
    "sprites",
    "pokemon",
    "versions",
)
SPRITES = [
    # ("generation-iii", "emerald", 386, True),
    ("generation-iii", "firered-leafgreen", 151, True),
    # ("generation-iii", "ruby-sapphire", 386, True),
    # ("generation-iv", "diamond-pearl", 493, True),
    # ("generation-iv", "heartgold-soulsilver", 493, True),
    # ("generation-iv", "platinum", 493, True),
    # ("generation-v", "black-white", 650, True),
    # ("generation-vii", "icons", 807, False),
]

encoded = []


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


def lz77(data):
    ans = []
    i = 0
    n = len(data)
    while i < n:
        run = (0, 0)
        for j in range(i):
            for k in range(j, min(i, n - i + j)):
                if data[k] != data[i + k - j]:
                    break
                run = max(run, (k - j + 1, j - i))
        if not run[0]:
            ans.append((0, 0, data[i]))
            i += 1
            continue

        runlen, delta = run
        nxt = data[i + runlen] if i + runlen < n else 0
        ans.append((-delta, runlen, nxt))
        i += 1 + runlen
    return zip(*ans)


for gen, game, max_id, has_shiny in SPRITES:
    sprites_dir = os.path.join(VERSIONS_DIR, gen, game)
    shiny_dir = os.path.join(sprites_dir, "shiny")
    for id in range(max_id):
        basename = "%d.png" % (id + 1)
        path = os.path.join(sprites_dir, basename)
        sprite = PIL.Image.open(path).convert("RGBA")
        if has_shiny:
            shiny_path = os.path.join(shiny_dir, basename)
            shiny = PIL.Image.open(shiny_path).convert("RGBA")
        else:
            shiny = sprite

        colormap = create_colormap(sprite, shiny)
        if len(colormap) > 16:
            print("Excess colors in sprite", path, file=sys.stderr)
            continue

        image = []
        xl, yl, xh, yh = sprite.getbbox()
        for y in range(yl, yh):
            for x in range(xl, xh):
                color = (pixel(sprite, x, y), pixel(shiny, x, y))
                image.append(colormap[color])

        delta, runlen, color = list(lz77(image))
        for d in runlen:
            print(d)
