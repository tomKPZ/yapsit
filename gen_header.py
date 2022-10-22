#!/usr/bin/env python3

from collections import Counter, OrderedDict, namedtuple
from functools import partial
from heapq import heapify, heappop, heappush
from math import ceil, log2
from multiprocessing import Pool
from os import path
from sys import stderr

import PIL.Image

SCRIPT_DIR = path.dirname(path.realpath(__file__))
VERSIONS_DIR = path.join(
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

Huffman = namedtuple("Huffman", ["form", "perm", "data2bits"])


def pixel(sprite, shiny, x, y):
    r1, g1, b1, a1 = sprite.getpixel((x, y))
    r2, g2, b2, a2 = shiny.getpixel((x, y))
    if not a1 and not a2:
        return None, None
    # Some gen 3 sprites have inaccurate alpha channels.
    if not a1 or not a2:
        raise Exception("Bad alpha channel")
    return (r1 // 8, g1 // 8, b1 // 8), (r2 // 8, g2 // 8, b2 // 8)


def create_palette(sprite, shiny):
    counter = Counter()
    n, m = sprite.size
    for y in range(m):
        for x in range(n):
            counter[pixel(sprite, shiny, x, y)] -= 1
    if len(counter) > 16:
        raise Exception("Excess colors in palette")
    del counter[(None, None)]
    palette = OrderedDict({(None, None): 0})
    for _, color in sorted(zip(counter.values(), counter.keys())):
        palette[color] = len(palette)
    return palette


def lz77(data, width, data2bits):
    # TODO: multiple representations of (dy, dx) for the same delta.
    def nbits(output):
        return sum((d2b[out] if out >= 0 else 0) for out, d2b in zip(output, data2bits))

    n = len(data)
    dp: list[tuple[int, tuple[int, int, int, int], int]] = [(0, (-1,) * 4, -1)] * n
    for i in reversed(range(n)):
        size, tail = (dp[i + 1][0], i + 1) if i + 1 < n else (0, -1)
        out = (-1 if i < width else 0, -1 if i == 0 else 128, -1, data[i])
        ans = (size + nbits(out), out, tail)
        for j in range(i):
            for k in range(j, n - i + j):
                if data[k] != data[k + i - j]:
                    break
                runlen = k - j + 1
                y1, x1 = divmod(j, width)
                y2, x2 = divmod(i, width)
                dy, dx = y2 - y1, x2 - x1 + 128
                if not (0 <= dx < 256 and 0 <= dy < 256):
                    continue
                index = i + runlen + 1
                size, tail = (dp[index][0], index) if index < n else (0, -1)
                out = (
                    dy if y2 else -1,
                    dx if y2 or x2 else -1,
                    runlen if dy or dx else -1,
                    data[i + runlen] if i + runlen < n else -1,
                )
                ans = min(ans, (size + nbits(out), out, tail))
        dp[i] = ans

    node = 0
    ans = []
    while node >= 0:
        _, first, rest = dp[node]
        ans.append(first)
        node = rest
    return ans


def he(data):
    counter = Counter(data)
    heap = [(counter[i], i, i) for i in range(256)]
    heapify(heap)
    while len(heap) > 1:
        c1, _, v1 = heappop(heap)
        c2, _, v2 = heappop(heap)
        heappush(heap, (c1 + c2, -len(heap), (v1, v2)))
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

    total = sum(counter.values())
    shannon = total * log2(total)
    bitlen = 0
    for x, count in counter.items():
        shannon -= count * log2(count)
        bitlen += count * len(data2bits[x])
    print(
        "%d/%d (+%.1fB) (+%.2f%%)"
        % (
            bitlen,
            ceil(shannon),
            (bitlen - shannon) / 8,
            100 * (bitlen / shannon - 1),
        ),
        file=stderr,
    )
    return Huffman(form, perm, data2bits)


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
    print("}}")


def read_image(sprites_dir, shiny_dir, id):
    basename = "%d.png" % (id + 1)
    pathname = path.join(sprites_dir, basename)
    sprite = PIL.Image.open(pathname).convert("RGBA")
    if shiny_dir:
        shiny_path = path.join(shiny_dir, basename)
        shiny = PIL.Image.open(shiny_path).convert("RGBA")
    else:
        shiny = sprite

    try:
        palette = create_palette(sprite, shiny)
    except Exception as e:
        print(e, "in", pathname, file=stderr)
        return None

    image = []
    xl, yl, xh, yh = sprite.getbbox()
    for y in range(yl, yh):
        for x in range(xl, xh):
            color = pixel(sprite, shiny, x, y)
            image.append(palette[color])
    width = xh - xl
    return ((width, yh - yl), palette, image)


def read_images(pool):
    images = []
    for gen, game, max_id, has_shiny in SPRITES:
        sprites_dir = path.join(VERSIONS_DIR, gen, game)
        shiny_dir = path.join(sprites_dir, "shiny") if has_shiny else None
        read = partial(read_image, sprites_dir, shiny_dir)
        images.extend(x for x in pool.map(read, range(max_id)) if x)
    return images


def compress_image(d2bs, input):
    (w, h), _, uncompressed = input
    return ((w, h), lz77(uncompressed, w, d2bs))


def compress_images(uncompressed, pool):
    palettes = []
    for _, palette, _ in uncompressed:
        regular_palette = []
        shiny_palette = []
        for regular, shiny in list(palette)[1:]:
            regular_palette.extend(regular)
            shiny_palette.extend(shiny)
        palettes.append((regular_palette, shiny_palette))
    colors = he([x for pairs in palettes for palette in pairs for x in palette])

    LZ77_LEN = 4
    d2bs = [[1] * 256] * LZ77_LEN
    for _ in range(3):
        sizes, streams = zip(*pool.map(partial(compress_image, d2bs), uncompressed))
        # TODO: repaletteize based on value stream.

        all_streams = [[] for _ in range(LZ77_LEN)]
        for stream in streams:
            for t in stream:
                for i, x in enumerate(t):
                    if x >= 0:
                        all_streams[i].append(x)
        lz = tuple(pool.map(he, all_streams))

        d2bs = [[len(huffman.data2bits[d]) for d in range(256)] for huffman in lz]
        bitstreams = []
        bitlens = []
        for stream, palette_pair in zip(streams, palettes):
            bitstream = []
            for t in stream:
                for huffman, x in zip(lz, t):
                    if x >= 0:
                        bitstream.extend(huffman.data2bits[x])
            for palette in palette_pair:
                for value in palette:
                    bitstream.extend(colors.data2bits[value])
            bitlens.append(len(bitstream))
            bitstreams.extend(bitstream)
        print("%.3fKB" % ((len(bitstreams) + 7) // 8 / 1000), file=stderr)
    return sizes, colors, bitstreams, bitlens, lz


def output(sizes, colors, bitstream, bitlens, lz):
    print('#include "types.h"')
    print("static const Sprite sprite_data[] = {")
    for (w, h), bitlen in zip(sizes, bitlens):
        print("{%d,%d,%d}," % (w, h, bitlen))
    print("};")
    print("static const uint8_t bitstream[] =")
    byte_encode(bitstream)
    print(";")
    print("const Sprites sprites = {")
    print("sprite_data,")
    print("%d," % len(sizes))
    print("{")
    for field in lz:
        output_huffman(field.form, field.perm)
        print(",")
    print("},")
    output_huffman(colors.form, colors.perm)
    print(",bitstream};")


def main():
    pool = Pool()
    uncompressed_images = read_images(pool)
    output(*compress_images(uncompressed_images, pool))


if __name__ == "__main__":
    main()
