#!/usr/bin/env python3

from collections import Counter, namedtuple
from itertools import accumulate
from json import load
from os import path
from typing import Iterable

import ctypes

import PIL.Image

SCRIPT_DIR = path.dirname(path.realpath(__file__))
ASSETS_DIR = path.join(SCRIPT_DIR, "assets")
SHEETS = set(
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
PALETTES = [2, 2, 2, 1, 1, 2]

Images = namedtuple(
    "Images",
    ["images", "variants", "limits", "groups", "ids", "frames", "palette_counts"],
)
Compressed = namedtuple(
    "Compressed", ["sizes", "offsets", "compressed", "dictionary", "decode_buffer"]
)


def create_palette(zipped_sprites):
    counter = Counter()
    for colors in zipped_sprites:
        counter[colors] -= 1
    if len(counter) > 16:
        raise Exception("Excess colors in palette")
    transparent = ((-1, -1, -1),) * len(zipped_sprites[0])
    del counter[transparent]
    palette = {transparent: 0}
    for _, color in sorted(zip(counter.values(), counter.keys())):
        palette[color] = len(palette)
    return palette


DICT_SIZE = 65536


def get_metadata():
    metadata = [
        (size, [g for g in group if g[0] in SHEETS])
        for size, group in load(open(path.join(ASSETS_DIR, "metadata.json")))
    ]
    return [(size, group) for size, group in metadata if group]


def trim_sprites(sprites, w, h):
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
            sprite[y * w + x] for y in range(yl, yh + 1) for x in range(xl, xh + 1)
        ]
    return (xh - xl + 1, yh - yl + 1, len(sprites))


def optimize_sprites_palettes(sprites):
    n = len(sprites)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            for (c1, c2), count in (
                {(i, j): 0 for i in range(16) for j in range(16)}
                | Counter(zip(sprites[i][0], sprites[j][0]))
            ).items():
                edges.append((-count, i, j, c1, c2))
    edges.sort()
    covered_from = set()
    covered_to = set()
    msf = []  # maximum spanning forest
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
        p += [((0, 0, 0),) * len(next(iter(palette)))] * (16 - len(p))
        for key, c in palette.items():
            p[inv[c]] = key
        palettes.append((p, list(set(image) - {-1})))
    return image_stream, palettes


def pixel(sheet, x, y):
    r, g, b, a = sheet.getpixel((x, y))
    return (r // 8, g // 8, b // 8) if a else (-1, -1, -1)


def read_sprite(sheet, row, col, w, h):
    return [
        pixel(
            sheet,
            w * col + x,
            h * row + y,
        )
        for y in range(h)
        for x in range(w)
    ]


def read_sprite_with_palettes(sheet, row, col, w, h, palettes):
    sprites = [read_sprite(sheet, row, col + p, w, h) for p in range(palettes)]
    pixels = list(zip(*sprites))
    palette = create_palette(pixels)
    sprite = [palette[colors] for colors in pixels]
    return sprite, palette


def read_sprite_sheet(w, h, metadata):
    name, vid, variants = metadata
    palettes = PALETTES[vid]

    filename = path.join(ASSETS_DIR, name + ".png")
    sheet = PIL.Image.open(filename).convert("RGBA")

    return [
        [
            read_sprite_with_palettes(
                sheet, row - vs + v, palettes * frame, w, h, palettes
            )
            for v in range(vs)
            for frame in range(FRAMES[vid])
        ]
        for vs, row in zip(variants, accumulate(variants))
    ]


def read_images():
    images = []
    variants = []
    frames = []
    palette_counts = []
    metadata = get_metadata()
    for (w, h), group in metadata:
        frames.extend(FRAMES[v] for _, v, _ in group)

        palette_count = set(PALETTES[v] for _, v, _ in group)
        assert len(palette_count) == 1
        palette_counts.append(next(iter(palette_count)))

        variants.extend(x for lst in zip(*(vc for _, _, vc in group)) for x in lst)

        for sprites in zip(*[read_sprite_sheet(w, h, g) for g in group]):
            sprites = sum(sprites, start=[])
            size = trim_sprites(sprites, w, h)
            image_stream, palettes = optimize_sprites_palettes(sprites)
            images.append((size, image_stream, palettes))
    limits = [len(group[0][2]) for _, group in metadata]
    groups = [len(group) for _, group in metadata]
    ids = max(limits)
    return Images(images, variants, limits, groups, ids, frames, palette_counts)


def load_zstd():
    lib = ctypes.CDLL("libzstd.so")
    lib.ZDICT_trainFromBuffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_uint,
    ]
    lib.ZDICT_trainFromBuffer.restype = ctypes.c_size_t
    lib.ZDICT_isError.argtypes = [ctypes.c_size_t]
    lib.ZDICT_isError.restype = ctypes.c_uint
    lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
    lib.ZSTD_getErrorName.restype = ctypes.c_char_p
    lib.ZSTD_createCCtx.argtypes = []
    lib.ZSTD_createCCtx.restype = ctypes.c_void_p
    lib.ZSTD_freeCCtx.argtypes = [ctypes.c_void_p]
    lib.ZSTD_freeCCtx.restype = ctypes.c_size_t
    lib.ZSTD_compress_usingDict.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.ZSTD_compress_usingDict.restype = ctypes.c_size_t
    lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
    lib.ZSTD_compressBound.restype = ctypes.c_size_t
    lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
    lib.ZSTD_isError.restype = ctypes.c_uint
    return lib


def scale_color(channel: int) -> int:
    if channel < 0:
        return 0
    return channel * 8 * 255 // 248


def group_palette_bytes(palette_entry, palette_count: int) -> Iterable[int]:
    for palette_index in range(16):
        for palette_id in range(palette_count):
            r, g, b = palette_entry[palette_index][palette_id]
            yield scale_color(r)
            yield scale_color(g)
            yield scale_color(b)


def group_bytes(size, image_stream, palettes, palette_count):
    image_bytes = bytes(image_stream)
    palette_bytes = bytearray()
    for palette_entry, _ in palettes:
        palette_bytes.extend(group_palette_bytes(palette_entry, palette_count))
    return image_bytes + palette_bytes


def group_palette_counts(limits, palette_counts):
    gid = 0
    remaining = limits[0]
    for _ in range(sum(limits)):
        while remaining == 0:
            gid += 1
            remaining = limits[gid]
        remaining -= 1
        yield palette_counts[gid]


def compress_images(uncompressed, limits, palette_counts):
    sizes = [size for size, _, _ in uncompressed]
    palette_count_iter = group_palette_counts(limits, palette_counts)
    raw_groups = [
        group_bytes(size, image_stream, palettes, palette_count)
        for (size, image_stream, palettes), palette_count in zip(
            uncompressed, palette_count_iter
        )
    ]

    decode_buffer = 0
    for (w, h, d), palette_count in zip(sizes, group_palette_counts(limits, palette_counts)):
        decode_buffer = max(
            decode_buffer, w * h * d + d * palette_count * 16 * 3
        )

    lib = load_zstd()
    dict_buffer = ctypes.create_string_buffer(DICT_SIZE)

    samples_buffer = ctypes.create_string_buffer(b"".join(raw_groups))
    sizes_arr = (ctypes.c_size_t * len(raw_groups))(
        *[len(sample) for sample in raw_groups]
    )
    dict_size = lib.ZDICT_trainFromBuffer(
        dict_buffer,
        DICT_SIZE,
        samples_buffer,
        sizes_arr,
        len(raw_groups),
    )
    if lib.ZDICT_isError(dict_size):
        raise RuntimeError(lib.ZSTD_getErrorName(dict_size).decode("utf-8"))
    dictionary = dict_buffer.raw[:dict_size]
    dict_data = ctypes.create_string_buffer(dictionary)

    ctx = lib.ZSTD_createCCtx()
    compressed = bytearray()
    offsets = []
    for sample in raw_groups:
        offsets.append((len(compressed), len(sample)))
        bound = lib.ZSTD_compressBound(len(sample))
        out_buffer = ctypes.create_string_buffer(bound)
        sample_buffer = ctypes.create_string_buffer(sample)
        result = lib.ZSTD_compress_usingDict(
            ctx,
            out_buffer,
            bound,
            sample_buffer,
            len(sample),
            dict_data,
            len(dictionary),
            3,
        )
        if lib.ZSTD_isError(result):
            raise RuntimeError(lib.ZSTD_getErrorName(result).decode("utf-8"))
        compressed.extend(out_buffer.raw[:result])
        offsets[-1] = (offsets[-1][0], result)
    lib.ZSTD_freeCCtx(ctx)

    print("%.3fKB" % (len(compressed) / 1000))
    return Compressed(sizes, offsets, compressed, dictionary, decode_buffer)


def output_array(name, arr, f):
    print("// " + name, file=f)
    print("{", file=f)
    print(", ".join(str(x) for x in arr), file=f)
    print("},", file=f)


def output(compressed: Compressed, images: Images):
    with open(path.join(SCRIPT_DIR, "sprites.c"), "w") as f:
        print('#include "types.h"', file=f)
        print("const Sprites sprites = {", file=f)
        output_array("limits", images.limits, f)
        output_array("variants", images.variants, f)
        output_array("groups", images.groups, f)
        output_array("frames", images.frames, f)
        output_array("palette_counts", images.palette_counts, f)
        print("{", file=f)
        for size in compressed.sizes:
            print("{%d,%d,%d}," % size, file=f)
        print("},", file=f)
        output_array("offsets", [o for o, _ in compressed.offsets], f)
        output_array("sizes", [s for _, s in compressed.offsets], f)
        output_array("dictionary", list(compressed.dictionary), f)
        output_array("compressed", list(compressed.compressed), f)
        print("};", file=f)

    with open(path.join(SCRIPT_DIR, "constants.h"), "w") as f:
        print("#define SHEET_COUNT %d" % len(images.frames), file=f)
        print("#define GROUP_COUNT %d" % len(images.groups), file=f)
        print("#define VARIANT_COUNT %d" % len(images.variants), file=f)
        print("#define SPRITE_COUNT %d" % len(images.images), file=f)
        print("#define ID_COUNT %d" % images.ids, file=f)
        print("#define DICT_LEN %d" % len(compressed.dictionary), file=f)
        print("#define COMPRESSED_LEN %d" % len(compressed.compressed), file=f)
        print("#define DECOMPRESS_BUFFER %d" % compressed.decode_buffer, file=f)
        print("#define MAX_PALETTES %d" % max(images.palette_counts), file=f)


def main():
    images = read_images()
    compressed = compress_images(images.images, images.limits, images.palette_counts)
    output(compressed, images)


if __name__ == "__main__":
    main()
