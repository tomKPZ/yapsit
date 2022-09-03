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
    ("generation-iii", "emerald", 386, True),
    # ("generation-iii", "firered-leafgreen", 151, True),
    # ("generation-iii", "ruby-sapphire", 386, True),
    # ("generation-iv", "diamond-pearl", 493, True),
    # ("generation-iv", "heartgold-soulsilver", 493, True),
    # ("generation-iv", "platinum", 493, True),
    # ("generation-v", "black-white", 650, True),
    # ("generation-vii", "icons", 807, False),
]


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


pokedex = collections.defaultdict(list)
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
        w, h = sprite.size
        for y in range(h):
            for x in range(w):
                color = (pixel(sprite, x, y), pixel(shiny, x, y))
                image.append(colormap[color])

        pokedex[id].append(((w, h), sprite.getbbox(), image, tuple(colormap)))


# def delta(s1, s2):
#     def aux(s1, s2):
#         d = collections.defaultdict(collections.Counter)
#         for c1, c2 in zip(s1, s2):
#             d[c1][c2] += 1
#         return sum(sum(vs.values()) - max(vs.values()) for vs in d.values())
#
#     return min(aux(s1, s2), aux(s2, s1))
#
#
# for i, pokemon in pokedex.items():
#     dedupe = []
#     for _, _, sprite, colormap in pokemon:
#         image = [colormap[color][0] for color in sprite]
#         for d in dedupe:
#             if len(d) == len(image):
#                 print(delta(d, image), i + 1, file=sys.stderr)
#         dedupe.append(image)

for pokemon in pokedex.values():
    for (sw, sh), (xl, yl, xh, yh), sprite, colormap in pokemon:
        w = xh - xl
        h = yh - yl
        print("{%d,%d,(uint8_t[]){" % (w, h), end="")
        startb = True
        for y in range(h):
            for x in range(w):
                color = sprite[(y + yl) * sw + (x + xl)]
                print(("0x%X" if startb else "%X,") % color, end="")
                startb = not startb
        print("},{")
        for (color, _) in colormap:
            print("0x%04X," % color, end="")
        print("},{")
        for (_, color) in colormap:
            print("0x%04X," % color, end="")
        print("}},")
