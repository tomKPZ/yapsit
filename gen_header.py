#!/usr/bin/env python3

import collections
import heapq
import itertools
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
    ("generation-iv", "platinum", 493, True),
    ("generation-v", "black-white", 650, True),
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


def rle(data):
    counts = []
    runs = []
    for x, g in itertools.groupby(data):
        count = len(list(g))
        while count > 16:
            counts.append(15)
            runs.append(x)
            count -= 16
        counts.append(count - 1)
        runs.append(x)
    return (counts, runs)


def he(data):
    counter = collections.Counter(data)
    heap = [(counter[i], i, i) for i in range(16)]
    heapq.heapify(heap)
    while len(heap) > 1:
        c1, _, v1 = heapq.heappop(heap)
        c2, _, v2 = heapq.heappop(heap)
        heapq.heappush(heap, (c1 + c2, -len(heap), (v1, v2)))
    tree = heap[0][2]

    data2bits = {}
    acc = []
    form = []
    perm = []

    def dfs(node):
        if type(node) == int:
            form.append(1)
            data2bits[node] = acc[::]
            perm.append(node)
            return
        form.append(0)
        l, r = node
        acc.append(0)
        dfs(l)
        acc.pop()
        acc.append(1)
        dfs(r)
        acc.pop()

    dfs(tree)
    bits = [y for x in data for y in data2bits[x]]
    return (bits, form, perm)


def bit_encode(bits):
    encoded = 0
    for bit in bits:
        encoded *= 2
        encoded += bit
    return encoded


def byte_encode(bits):
    while len(bits) % 8 != 0:
        bits.append(0)
    print("{")
    for i in range(0, len(bits), 8):
        print("0x%02X," % bit_encode(bits[i : i + 8]), end="")
    print("},")


def output_huffman(data):
    bits, form, perm = he(data)
    print("{")
    byte_encode(list(form))
    print("{")
    startb = True
    for x in perm:
        print(("0x%X" if startb else "%X,") % x, end="")
        startb = not startb
    print("},(uint8_t[])")
    byte_encode(bits)
    print("},")


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

        print("{%d,%d,{" % (xh - xl, yh - yl), end="")
        for (color, _) in colormap:
            print("0x%04X," % color, end="")
        print("},{")
        for (_, color) in colormap:
            print("0x%04X," % color, end="")
        print("},")
        counts, runs = rle(image)
        output_huffman(counts)
        output_huffman(runs)
        print("},")
