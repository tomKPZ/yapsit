#!/usr/bin/env python3

import heapq
import os
import sys
from collections import Counter, OrderedDict, defaultdict, namedtuple

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

Huffman = namedtuple("Huffman", ["bits", "form", "perm", "data2bits"])
Lz77 = namedtuple("Lz77", ["deltas", "runlen", "values"])


def pixel(sprite, x, y):
    r, g, b, a = sprite.getpixel((x, y))
    if not a:
        return 0
    return (a >> 7) << 15 | (r >> 3) << 10 | (g >> 3) << 5 | (b >> 3)


def create_colormap(sprite, shiny):
    counter = Counter()
    n, m = sprite.size
    for y in range(m):
        for x in range(n):
            color = (pixel(sprite, x, y), pixel(shiny, x, y))
            counter[color] -= 1
    del counter[(0, 0)]
    colormap = OrderedDict({(0, 0): 0})
    for count, color in sorted(zip(counter.values(), counter.keys())):
        colormap[color] = len(colormap)
    return colormap


def lz77(data):
    n = len(data)
    dp = [0] * n
    for i in reversed(range(n)):
        size, lst = dp[i + 1] if i + 1 < n else (0, None)
        ans = (size + 1, ((0, 0, data[i]), lst))
        for j in range(max(0, i - 15), i):
            for k in range(j, min(i, n - i + j)):
                if data[k] != data[i + k - j]:
                    break
                runlen = k - j + 1
                delta = i - j
                index = i + runlen + 1
                lstlen, lst = dp[index] if index < n else (0, None)
                nxt = data[i + runlen] if i + runlen < n else None
                ans = min(ans, (1 + lstlen, ((delta, runlen, nxt), lst)))
        dp[i] = ans

    node = dp[0][1]
    ans = []
    while node is not None:
        first, rest = node
        ans.append(first)
        node = rest
    return Lz77(*zip(*ans))


def he(data):
    counter = Counter(data)
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
    return Huffman(bits, form, perm, data2bits)


def byte_encode(bits):
    while len(bits) % 8 != 0:
        bits.append(0)
    print("{")
    for i in range(0, len(bits), 8):
        encoded = 0
        for bit in bits[i : i + 8]:
            encoded *= 2
            encoded += bit
        print("0x%02X," % encoded, end="")
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
        images.append(((xh - xl, yh - yl), colormap, lz77(image)))

all_streams = defaultdict(list)
for _, _, streams in images:
    for i, stream in enumerate(streams):
        all_streams[i].extend(list(stream))
lz = Lz77(*(he(stream) for stream in all_streams.values()))

print('#include "types.h"')
print("const Sprite sprites[] = {")
for size, colormap, streams in images:
    print("{%d,%d,{" % size, end="")
    for (color, _) in list(colormap)[1:]:
        print("0x%04X," % color, end="")
    print("},{")
    for (_, color) in list(colormap)[1:]:
        print("0x%04X," % color, end="")
    print("},")
    for stream, huffman in zip(streams, lz):
        size = sum(len(huffman.data2bits[x]) for x in stream)
        print("%d," % size)
    print("},")
print("};")
print("const size_t n_sprites = %s;" % len(images))
print("const uint8_t deltas_bits[] =")
byte_encode(lz.deltas.bits)
print(";")
print("const uint8_t runlen_bits[] =")
byte_encode(lz.runlen.bits)
print(";")
print("const uint8_t values_bits[] =")
byte_encode(lz.values.bits)
print(";")
print("const HuffmanHeader deltas_header =")
output_huffman(lz.deltas.form, lz.deltas.perm)
print("const HuffmanHeader runlen_header =")
output_huffman(lz.runlen.form, lz.runlen.perm)
print("const HuffmanHeader values_header =")
output_huffman(lz.values.form, lz.values.perm)
