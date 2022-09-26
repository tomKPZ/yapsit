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
    ("generation-iii", "firered-leafgreen", 10, True),
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
        return None
    return (r // 8, g // 8, b // 8)


def create_colormap(sprite, shiny):
    counter = Counter()
    n, m = sprite.size
    for y in range(m):
        for x in range(n):
            color = (pixel(sprite, x, y), pixel(shiny, x, y))
            counter[color] -= 1
    del counter[(None, None)]
    colormap = OrderedDict({(None, None): 0})
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


def output_huffman(form, perm, bits):
    print("{")
    byte_encode(list(form))
    print(",{")
    for x in perm:
        print("0x%02X," % x, end="")
    print("}, %s}" % bits)


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


def read_images(pool):
    images = []
    for gen, game, max_id, has_shiny in SPRITES:
        sprites_dir = os.path.join(VERSIONS_DIR, gen, game)
        shiny_dir = os.path.join(sprites_dir, "shiny") if has_shiny else None
        read = partial(read_image, sprites_dir, shiny_dir)
        images.extend(pool.map(read, range(max_id)))
    return images


def compress_image(d2bs, input):
    uncompressed_image, color_sizes = input
    (w, h), _, uncompressed = uncompressed_image
    return ((w, h), color_sizes, lz77(uncompressed, w, d2bs))


def compress_images(uncompressed_images, pool):
    colormaps = []
    for _, colormap, _ in uncompressed_images:
        regular_colormap = []
        shiny_colormap = []
        for regular, shiny in list(colormap)[1:]:
            regular_colormap.extend(regular)
            shiny_colormap.extend(shiny)
        colormaps.append((regular_colormap, shiny_colormap))
    colors = he([x for pairs in colormaps for colormap in pairs for x in colormap])
    color_sizes = [
        [sum(len(colors.data2bits[x]) for x in cmap) for cmap in pair]
        for pair in colormaps
    ]
    uncompressed = list(zip(uncompressed_images, color_sizes))

    d2bs = [[1] * 256] * len(LZ77_FIELDS)
    for _ in range(3):
        images = pool.map(partial(compress_image, d2bs), uncompressed)

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
    return images, colors, lz


def output(images, colors, lz):
    print('#include "types.h"')
    print("const Sprite sprites[] = {")
    for size, color_sizes, streams in images:
        print("{%d,%d," % size)
        for color_size in color_sizes:
            print("%d," % color_size)
        for stream, huffman in zip(streams, lz):
            size = sum(len(huffman.data2bits[x]) for x in stream)
            print("%d," % size)
        print("},")
    print("};")
    print("const size_t n_sprites = %s;" % len(images))
    for name, field in zip(LZ77_FIELDS, lz):
        print("static const uint8_t %s_bits[] =" % name)
        byte_encode(field.bits)
        print(";")
    print("const Lz77Header lz77 = {")
    for name, field in zip(LZ77_FIELDS, lz):
        output_huffman(field.form, field.perm, "%s_bits" % name)
        print(",")
    print("};")

    print("static const uint8_t colormap_bits[] =")
    byte_encode(colors.bits)
    print(";")
    print("const HuffmanHeader colormaps =")
    output_huffman(colors.form, colors.perm, "colormap_bits")
    print(";")


def main():
    pool = Pool()
    uncompressed_images = read_images(pool)
    images, colors, lz = compress_images(uncompressed_images, pool)
    output(images, colors, lz)


if __name__ == "__main__":
    main()
