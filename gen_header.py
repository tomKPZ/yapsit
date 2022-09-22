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
    counter = collections.Counter()
    n, m = sprite.size
    for y in range(m):
        for x in range(n):
            color = (pixel(sprite, x, y), pixel(shiny, x, y))
            counter[color] -= 1
    del counter[(0, 0)]
    colormap = collections.OrderedDict({(0, 0): 0})
    for count, color in sorted(zip(counter.values(), counter.keys())):
        colormap[color] = len(colormap)
    return colormap


def rle(data):
    counts = []
    values = []
    for x, g in itertools.groupby(data):
        count = len(list(g))
        while count > 16:
            counts.append(15)
            values.append(x)
            count -= 16
        counts.append(count - 1)
        values.append(x)
    return (counts, values)


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
    return (bits, form, perm, data2bits)


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
    print("}")


def output_huffman(form, perm):
    print("{")
    byte_encode(list(form))
    print(",{")
    startb = True
    for x in perm:
        print(("0x%X" if startb else "%X,") % x, end="")
        startb = not startb
    print("}};")


images = []
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
        bbox = sprite.getbbox()
        xl, yl, xh, yh = bbox
        for y in range(yl, yh):
            for x in range(xl, xh):
                color = (pixel(sprite, x, y), pixel(shiny, x, y))
                image.append(colormap[color])
        counts, values = rle(image)
        images.append(((xh - xl, yh - yl), colormap, counts, values))

all_counts = []
all_values = []
for _, _, counts, values in images:
    all_counts.extend(counts)
    all_values.extend(values)
count_bits, count_form, count_perm, count_data2bits = he(all_counts)
value_bits, value_form, value_perm, value_data2bits = he(all_values)

print('#include "types.h"')
print("const Sprite sprites[] = {")
for size, colormap, counts, values in images:
    print("{%d,%d,{" % size, end="")
    for (color, _) in list(colormap)[1:]:
        print("0x%04X," % color, end="")
    print("},{")
    for (_, color) in list(colormap)[1:]:
        print("0x%04X," % color, end="")
    print("},")
    count_size = sum(len(count_data2bits[x]) for x in counts)
    value_size = sum(len(value_data2bits[x]) for x in values)
    print("%d,%d," % (count_size, value_size))
    print("},")
print("};")
print("const size_t n_sprites = %s;" % len(images))
print("const uint8_t count_bits[] =")
byte_encode(count_bits)
print(";")
print("const uint8_t value_bits[] =")
byte_encode(value_bits)
print(";")
print("const HuffmanHeader count_header =")
output_huffman(count_form, count_perm)
print("const HuffmanHeader value_header =")
output_huffman(value_form, value_perm)
