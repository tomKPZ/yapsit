#!/usr/bin/env python3

import heapq
import os
import sys
from collections import Counter, OrderedDict, defaultdict, namedtuple
from functools import partial
from multiprocessing import Pool

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
LZ77_FIELDS = ["dys", "dxs", "runlen", "values"]

Huffman = namedtuple("Huffman", ["bits", "form", "perm", "data2bits"])
Lz77 = namedtuple("Lz77", LZ77_FIELDS)


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


def lz77(data, width, data2bits):
    def nbits(output):
        return sum(d2b[out] for out, d2b in zip(output, data2bits))

    n = len(data)
    dp = [0] * n
    for i in reversed(range(n)):
        j = i
        while j + 1 < n and data[j + 1] == data[i]:
            j += 1
        size, lst = dp[j + 1] if j + 1 < n else (0, None)
        out = (0, 128, j - i + 1, data[i])
        ans = (size + nbits(out), (out, lst))
        for j in range(i):
            for k in range(j, min(i, n - i + j)):
                if data[k] != data[i + k - j]:
                    break
                runlen = k - j + 1
                y1, x1 = divmod(j, width)
                y2, x2 = divmod(i, width)
                dy, dx = y2 - y1, x2 - x1 + 128
                if not (0 <= dx < 256 and 0 <= dy < 256):
                    continue
                index = i + runlen + 1
                size, lst = dp[index] if index < n else (0, None)
                # TODO: don't output nxt if i + runlen == n
                nxt = data[i + runlen] if i + runlen < n else 0
                out = (dy, dx, runlen, nxt)
                ans = min(ans, (size + nbits(out), (out, lst)))
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
    heap = [(counter[i], i, i) for i in range(256)]
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

    # TODO: only output non-zero counted values
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
    for x in perm:
        print("0x%02X," % x, end="")
    print("}};")


def read_image(sprites_dir, shiny_dir, id):
    basename = "%d.png" % (id + 1)
    path = os.path.join(sprites_dir, basename)
    sprite = PIL.Image.open(path).convert("RGBA")
    if shiny_dir:
        shiny_path = os.path.join(shiny_dir, basename)
        shiny = PIL.Image.open(shiny_path).convert("RGBA")
    else:
        shiny = sprite

    colormap = create_colormap(sprite, shiny)
    if len(colormap) > 16:
        print("Excess colors in sprite", path, file=sys.stderr)
        # TODO: error handling

    image = []
    bbox = sprite.getbbox()
    xl, yl, xh, yh = bbox
    for y in range(yl, yh):
        for x in range(xl, xh):
            color = (pixel(sprite, x, y), pixel(shiny, x, y))
            image.append(colormap[color])
    width = xh - xl
    return ((width, yh - yl), colormap, image)


pool = Pool()

uncompressed_images = []
for gen, game, max_id, has_shiny in SPRITES:
    sprites_dir = os.path.join(VERSIONS_DIR, gen, game)
    shiny_dir = os.path.join(sprites_dir, "shiny") if has_shiny else None
    read = partial(read_image, sprites_dir, shiny_dir)
    uncompressed_images.extend(pool.map(read, range(max_id)))

d2bs = [[1] * 256] * len(LZ77_FIELDS)
for _ in range(3):

    def compress(uncompressed_image):
        (w, h), colormap, uncompressed = uncompressed_image
        return ((w, h), colormap, lz77(uncompressed, w, d2bs))

    pool = Pool()
    images = pool.map(compress, uncompressed_images)

    all_streams = defaultdict(list)
    for _, _, streams in images:
        for i, stream in enumerate(streams):
            all_streams[i].extend(list(stream))
    lz = Lz77(*pool.map(he, all_streams.values()))
    d2bs = []
    size = 0
    for huffman in lz:
        size += len(huffman.bits)
        d2bs.append([len(huffman.data2bits[d]) for d in range(256)])
    print("%.3fKB" % ((size + 7) // 8 / 1000), file=sys.stderr)

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
print("const uint8_t dys_bits[] =")
byte_encode(lz.dys.bits)
print(";")
print("const uint8_t dxs_bits[] =")
byte_encode(lz.dxs.bits)
print(";")
print("const uint8_t runlen_bits[] =")
byte_encode(lz.runlen.bits)
print(";")
print("const uint8_t values_bits[] =")
byte_encode(lz.values.bits)
print(";")
print("const HuffmanHeader dys_header =")
output_huffman(lz.dys.form, lz.dys.perm)
print("const HuffmanHeader dxs_header =")
output_huffman(lz.dxs.form, lz.dxs.perm)
print("const HuffmanHeader runlen_header =")
output_huffman(lz.runlen.form, lz.runlen.perm)
print("const HuffmanHeader values_header =")
output_huffman(lz.values.form, lz.values.perm)
