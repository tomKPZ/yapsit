#!/usr/bin/env python3

from collections import Counter, namedtuple
from functools import partial
from heapq import heapify, heappop, heappush
from json import load
from math import ceil, log2
from multiprocessing import Pool
from os import path
from sys import stderr

import PIL.Image
from cffi import FFI

SCRIPT_DIR = path.dirname(path.realpath(__file__))
ASSETS_DIR = path.join(SCRIPT_DIR, "assets")
MONTAGES = set(
    [
        "ruby",
        "firered",
        "emerald",
        "diamond",
        "platinum",
        "heartgold",
        "black",
        "old",
        "icons",
    ]
)
FRAMES = [1, 2, 2, 1, 1, 2]
SHINY = [1, 1, 1, 0, 0, 1]

Huffman = namedtuple("Huffman", ["form", "perm", "data2bits"])
Images = namedtuple(
    "Images", ["images", "variants", "limits", "groups", "ids", "frames"]
)
Compressed = namedtuple("Compressed", ["sizes", "colors", "bitstream", "bitlens", "lz"])


def create_palette(sprite, shiny):
    counter = Counter()
    for pair in zip(sprite, shiny):
        counter[pair] -= 1
    if len(counter) > 16:
        raise Exception("Excess colors in palette")
    transparent = ((-1, -1, -1), (-1, -1, -1))
    del counter[transparent]
    palette = {transparent: 0}
    for _, color in sorted(zip(counter.values(), counter.keys())):
        palette[color] = len(palette)
    return palette


def lz3d(data, size, data2bits):
    n = len(data) + 1
    dp = ffibuilder.new("int[%d][7]" % n, [[0] + [-1] * 6] * n)
    window = 10000 if len(data) > 80000 else len(data)
    lib.lz3d(*size, window, data, data2bits, dp)  # type: ignore

    node = 0
    ans = []
    while node < n - 1:
        ans.append(list(dp[node])[2:])
        node = dp[node][1]
    return ans


def huffman_encode(data):
    counter = Counter(data)
    heap = list(zip(counter.values(), counter.keys()))
    heapify(heap)
    nodes: list[tuple[int, int, int]] = [(i, -1, -1) for i in range(256)]
    while len(heap) > 1:
        c1, v1 = heappop(heap)
        c2, v2 = heappop(heap)
        heappush(heap, (c1 + c2, len(nodes)))
        nodes.append((-1, v1, v2))

    data2bits = {}
    acc = []
    form = []
    perm = []

    def dfs(node: tuple[int, int, int]):
        val = node[0]
        if val >= 0:
            form.append(1)
            data2bits[val] = acc[::]
            perm.append(val)
            return
        form.append(0)
        _, l, r = node
        acc.append(0)
        dfs(nodes[l])
        acc.pop()
        acc.append(1)
        dfs(nodes[r])
        acc.pop()

    dfs(nodes[-1])
    huffman_info(counter, data2bits)
    return Huffman(form, perm, data2bits)


def huffman_info(counter, data2bits):
    total = sum(counter.values())
    if not total:
        return
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


def pixel(montage, x, y):
    r, g, b, a = montage.getpixel((x, y))
    return (r // 8, g // 8, b // 8) if a else (-1, -1, -1)


def get_metadata():
    metadata = [
        (size, [g for g in group if g[0] in MONTAGES])
        for size, group in load(open(path.join(ASSETS_DIR, "metadata.json")))
    ]
    return [(size, group) for size, group in metadata if group]


def read_images():
    cmm = {(i, j): 0 for i in range(16) for j in range(16)}
    images = []
    variants = []
    frames = []
    metadata = get_metadata()
    for (w, h), group in metadata:
        spritess = []
        for name, variants_id, variant_counts in group:
            frames.append(FRAMES[variants_id])
            montage = PIL.Image.open(path.join(ASSETS_DIR, name + ".png")).convert(
                "RGBA"
            )
            row = 0
            spritess.append([])
            variants.extend(variant_counts)
            for variant_count in variant_counts:
                for _ in range(variant_count):
                    for frame in range(FRAMES[variants_id]):
                        data = []
                        shiny = []
                        for y in range(h):
                            for x in range(w):
                                xp = 2 * frame * w + x
                                yp = h * row + y
                                data.append(pixel(montage, xp, yp))
                                # TODO: Don't output shiny palette if not necessary.
                                if SHINY[variants_id]:
                                    shiny.append(pixel(montage, w + xp, yp))
                                else:
                                    shiny.append(pixel(montage, xp, yp))

                        palette = create_palette(data, shiny)
                        sprite = [palette[colors] for colors in zip(data, shiny)]
                        spritess[-1].append((sprite, palette))

                    row += 1
        for sprites in zip(*spritess):
            xl = yl = 255
            xh = yh = 0
            for sprite, _ in sprites:
                for y in range(h):
                    for x in range(w):
                        if sprite[y * w + x]:
                            xl = min(xl, x)
                            yl = min(yl, y)
                            xh = max(xh, x)
                            yh = max(yh, y)
            for sprite, _ in sprites:
                sprite[:] = [
                    sprite[y * w + x]
                    for y in range(yl, yh + 1)
                    for x in range(xl, xh + 1)
                ]
            n = len(sprites)
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    for (c1, c2), count in (
                        cmm | Counter(zip(sprites[i][0], sprites[j][0]))
                    ).items():
                        edges.append((-count, i, j, c1, c2))
            edges.sort()
            covered_from = set()
            covered_to = set()
            msf = []
            for _, i, j, c1, c2 in edges:
                if (i, j, c1) in covered_from or (i, j, c2) in covered_to:
                    continue
                covered_from.add((i, j, c1))
                covered_to.add((i, j, c2))
                msf.append((i, j, c1, c2))
            perms = [list(range(16))] + [[16] * 16 for _ in range(n - 1)]
            for i, j, c1, c2 in sorted(msf):
                perms[j][c2] = perms[i][c1]
            image_stream = []
            palettes = []
            for perm, (sprite, palette) in zip(perms, sprites):
                inv = dict(zip(perm, range(16)))
                image = [inv[c] for c in sprite]
                image_stream.extend(image)
                p = list(palette.keys())
                p += [((0, 0, 0), (0, 0, 0))] * (16 - len(p))
                for key, c in palette.items():
                    p[inv[c]] = key
                palettes.append((p, list(set(image) - {-1})))
            size = (xh - xl + 1, yh - yl + 1, n)
            images.append((size, image_stream, palettes))
    limits = [len(group[0]) for _, group in metadata]
    groups = [len(group) for _, group in metadata]
    ids = max(limits)
    return Images(images, variants, limits, groups, ids, frames)


def compress_image(d2bs, input):
    i, (size, uncompressed, _) = input
    return i, lz3d(uncompressed, size, d2bs)


def compress_images(uncompressed):
    sizes = [size for size, _, _ in uncompressed]
    images = [image for _, image, _ in uncompressed]
    palettess = [palettes for _, _, palettes in uncompressed]

    global ffibuilder, lib
    ffibuilder = FFI()
    ffibuilder.cdef(
        """
void lz3d(uint8_t width, uint8_t height, uint8_t depth, unsigned int window,
          const uint8_t data[], const uint8_t data2bits[5][256], int dp[][7]);
"""
    )
    lib = ffibuilder.dlopen(path.join(SCRIPT_DIR, "liblz77.so"))

    LZ77_LEN = 5
    d2bs = [[1] * 256] * LZ77_LEN
    for _ in range(5):
        pool = Pool()
        perm = sorted(enumerate(uncompressed), key=lambda x: -len(x[1][1]))
        _, streams = zip(
            *sorted(pool.map(partial(compress_image, d2bs), perm, chunksize=1))
        )

        # Repaletteize based on value stream.
        for palettes, stream, image in zip(palettess, streams, images):
            counter = Counter(s[-1] for s in stream)
            counter[0] = len(stream)
            del counter[-1]
            perm = sorted(counter.keys(), key=lambda key: -counter[key])
            while len(perm) < 16:
                perm.append(len(perm))
            inv = dict(zip(perm, range(16)))
            for i, (p, c) in enumerate(palettes):
                palettes[i] = [p[j] for j in inv], [inv[v] for v in c]
            for l in stream:
                l[-1] = inv.get(l[-1], -1)
            image[:] = [inv[v] for v in image]
        palettes = [
            [x for p, m in ps for pair in p[1 : max(m) + 1] for c in pair for x in c]
            for ps in palettess
        ]
        colors = huffman_encode([c for palette in palettes for c in palette])

        all_streams = [[] for _ in range(LZ77_LEN)]
        for stream in streams:
            for t in stream:
                for i, x in enumerate(t):
                    if x >= 0:
                        all_streams[i].append(x)
        lz = tuple(pool.map(huffman_encode, all_streams, chunksize=1))
        pool.close()

        d2bs = [
            [len(huffman.data2bits.get(d, [])) for d in range(256)] for huffman in lz
        ]
        bitstreams = []
        bitlens = []
        for stream, palette in zip(streams, palettes):
            bitstream = []
            for t in stream:
                for huffman, x in zip(lz, t):
                    if x >= 0:
                        bitstream.extend(huffman.data2bits[x])
            for c in palette:
                bitstream.extend(colors.data2bits[c])
            bitlens.append(len(bitstream))
            bitstreams.extend(bitstream)
        print("%.3fKB" % ((len(bitstreams) + 7) // 8 / 1000), file=stderr)
    ffibuilder.dlclose(lib)
    return Compressed(sizes, colors, bitstreams, bitlens, lz)  # type: ignore


def output_bits(bits):
    while len(bits) % 8 != 0:
        bits.append(0)
    print("{")
    for i in range(0, len(bits), 8):
        encoded = 0
        for bit in bits[i : i + 8]:
            encoded *= 2
            encoded += bit
        print("0x%02X," % encoded, end="")
    print("},")


# TODO: Move huffman headers to bitstream.
def output_huffman(form, perm):
    print("{")
    output_bits(list(form))
    print("{")
    for x in perm:
        print("0x%02X," % x, end="")
    print("}},")


def output_array(name, arr):
    print("// " + name)
    print("{")
    print(", ".join(str(x) for x in arr))
    print("},")


def output(compressed, images):
    print('#include "types.h"')
    print("const Sprites sprites = {")
    print("{")
    for (w, h, d), bitlen in zip(compressed.sizes, compressed.bitlens):
        print("{%d,%d,%d,%d}," % (w, h, d, bitlen))
    print("},")
    print("{")
    for field in compressed.lz:
        output_huffman(field.form, field.perm)
    print("},")
    output_huffman(compressed.colors.form, compressed.colors.perm)
    output_bits(compressed.bitstream)
    output_array("variants", images.variants)
    output_array("limits", images.limits)
    output_array("groups", images.groups)
    output_array("frames", images.frames)
    print("};")

    with open(path.join(SCRIPT_DIR, "constants.h"), "w") as f:
        bitcount = (len(compressed.bitstream) + 7) // 8
        print("#define SHEET_COUNT %d" % len(images.frames), file=f)
        print("#define GROUP_COUNT %d" % len(images.groups), file=f)
        print("#define VARIANT_COUNT %d" % len(images.variants), file=f)
        print("#define BITSTREAM_LEN %d" % bitcount, file=f)
        print("#define SPRITE_COUNT %d" % len(images.images), file=f)
        print("#define ID_COUNT %d" % images.ids, file=f)


def main():
    images = read_images()
    compressed = compress_images(images.images)
    output(compressed, images)


if __name__ == "__main__":
    main()
