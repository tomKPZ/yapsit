#!/usr/bin/env python3

from collections import Counter, namedtuple
from itertools import accumulate
from json import load
from os import path
import ctypes
import struct
import zlib

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
    [
        "images",
        "variants",
        "limits",
        "groups",
        "ids",
        "frames",
        "palette_counts",
        "gids",
    ],
)
Compressed = namedtuple(
    "Compressed",
    [
        "raw_data",
        "dict_data",
        "compressed_data",
        "image_data_len",
        "palette_data_len",
    ],
)
Sheet = namedtuple("Sheet", ["width", "height", "data"])


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


def paeth_predictor(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_scanlines(raw, width, height, bpp):
    stride = width * bpp
    out = bytearray(height * stride)
    offset = 0
    prev = bytearray(stride)
    for y in range(height):
        filter_type = raw[offset]
        offset += 1
        row = bytearray(raw[offset : offset + stride])
        offset += stride
        if filter_type == 1:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + left) & 0xFF
        elif filter_type == 2:
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                up = prev[i]
                row[i] = (row[i] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                up = prev[i]
                up_left = prev[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + paeth_predictor(left, up, up_left)) & 0xFF
        elif filter_type != 0:
            raise ValueError("Unsupported PNG filter: %d" % filter_type)
        out[y * stride : (y + 1) * stride] = row
        prev = row
    return out


def load_png_rgba(filename):
    with open(filename, "rb") as f:
        signature = f.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Invalid PNG signature")
        width = height = None
        color_type = None
        bit_depth = None
        palette = None
        transparency = None
        chunks = []
        while True:
            length_bytes = f.read(4)
            if not length_bytes:
                break
            length = struct.unpack(">I", length_bytes)[0]
            ctype = f.read(4)
            data = f.read(length)
            f.read(4)  # CRC
            if ctype == b"IHDR":
                width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                    ">IIBBBBB", data
                )
                if bit_depth != 8 or color_type not in (3, 6) or interlace != 0:
                    raise ValueError("Unsupported PNG format")
            elif ctype == b"IDAT":
                chunks.append(data)
            elif ctype == b"PLTE":
                palette = [
                    tuple(data[i : i + 3]) for i in range(0, len(data), 3)
                ]
            elif ctype == b"tRNS":
                transparency = list(data)
            elif ctype == b"IEND":
                break
        if width is None or height is None:
            raise ValueError("Missing IHDR")
        if color_type == 3 and palette is None:
            raise ValueError("Missing PLTE for indexed PNG")
        raw = zlib.decompress(b"".join(chunks))
        if color_type == 6:
            decoded = decode_scanlines(raw, width, height, 4)
            return Sheet(width=width, height=height, data=decoded)

        decoded = decode_scanlines(raw, width, height, 1)
        if transparency is None:
            transparency = [255] * len(palette)
        palette_rgba = [
            (r, g, b, transparency[i] if i < len(transparency) else 255)
            for i, (r, g, b) in enumerate(palette)
        ]
        rgba = bytearray(width * height * 4)
        for i, idx in enumerate(decoded):
            r, g, b, a = palette_rgba[idx]
            offset = i * 4
            rgba[offset : offset + 4] = bytes((r, g, b, a))
        return Sheet(width=width, height=height, data=rgba)


def pixel(sheet, x, y):
    idx = (y * sheet.width + x) * 4
    r, g, b, a = sheet.data[idx : idx + 4]
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
    sheet = load_png_rgba(filename)

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
    gids = []
    metadata = get_metadata()
    for gid, ((w, h), group) in enumerate(metadata):
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
            gids.append(gid)
    limits = [len(group[0][2]) for _, group in metadata]
    groups = [len(group) for _, group in metadata]
    ids = max(limits)
    return Images(images, variants, limits, groups, ids, frames, palette_counts, gids)


def scale_color(value):
    if value < 0:
        return 0
    return value * 8 * 255 // 248


def build_sprite_data(images: Images):
    image_data = bytearray()
    palette_data = bytearray()
    sprite_entries = []
    samples = []

    for (size, image_stream, palettes), gid in zip(images.images, images.gids):
        w, h, d = size
        image_offset = len(image_data)
        image_bytes = bytes([v if v >= 0 else 0 for v in image_stream])
        image_data.extend(image_bytes)

        palette_offset = len(palette_data)
        palette_count = images.palette_counts[gid]
        palette_bytes = bytearray()
        for palette, _ in palettes:
            for variant in range(palette_count):
                for entry in palette:
                    r, g, b = entry[variant]
                    palette_bytes.extend(
                        [scale_color(r), scale_color(g), scale_color(b)]
                    )
        palette_data.extend(palette_bytes)
        samples.append(image_bytes + palette_bytes)

        sprite_entries.append((w, h, d, image_offset, palette_offset))

    return sprite_entries, image_data, palette_data, samples


def align_blob(blob: bytearray, alignment: int):
    padding = (-len(blob)) % alignment
    if padding:
        blob.extend(b"\0" * padding)


def serialize_sprite_blob(images: Images, sprite_entries, image_data, palette_data):
    blob = bytearray()
    align_blob(blob, 2)
    for value in images.limits:
        blob.extend(struct.pack("<H", value))

    blob.extend(bytes(images.variants))
    blob.extend(bytes(images.groups))
    blob.extend(bytes(images.frames))
    blob.extend(bytes(images.palette_counts))

    align_blob(blob, 4)
    for entry in sprite_entries:
        blob.extend(struct.pack("<3B1xII", *entry))

    blob.extend(image_data)
    blob.extend(palette_data)
    return blob


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
    lib.ZDICT_getErrorName.argtypes = [ctypes.c_size_t]
    lib.ZDICT_getErrorName.restype = ctypes.c_char_p

    lib.ZSTD_createCCtx.restype = ctypes.c_void_p
    lib.ZSTD_freeCCtx.argtypes = [ctypes.c_void_p]
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
    lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
    lib.ZSTD_getErrorName.restype = ctypes.c_char_p
    return lib


def train_zstd_dict(lib, samples, dict_size):
    sample_buffer = b"".join(samples)
    sizes = (ctypes.c_size_t * len(samples))(*[len(s) for s in samples])
    dict_buffer = ctypes.create_string_buffer(dict_size)
    sample_buf = ctypes.create_string_buffer(sample_buffer)
    result = lib.ZDICT_trainFromBuffer(
        dict_buffer, dict_size, sample_buf, sizes, len(samples)
    )
    if lib.ZDICT_isError(result):
        raise RuntimeError(lib.ZDICT_getErrorName(result).decode("utf-8"))
    return dict_buffer.raw[:result]


def zstd_compress(lib, data, dict_data, level=19):
    cctx = lib.ZSTD_createCCtx()
    if not cctx:
        raise RuntimeError("Failed to create zstd context")

    src = ctypes.create_string_buffer(bytes(data))
    dict_buf = ctypes.create_string_buffer(dict_data)
    max_out = lib.ZSTD_compressBound(len(data))
    out = ctypes.create_string_buffer(max_out)
    result = lib.ZSTD_compress_usingDict(
        cctx,
        out,
        max_out,
        src,
        len(data),
        dict_buf,
        len(dict_data),
        level,
    )
    lib.ZSTD_freeCCtx(cctx)
    if lib.ZSTD_isError(result):
        raise RuntimeError(lib.ZSTD_getErrorName(result).decode("utf-8"))
    return out.raw[:result]


def compress_images(images: Images):
    sprite_entries, image_data, palette_data, samples = build_sprite_data(images)
    raw_data = serialize_sprite_blob(images, sprite_entries, image_data, palette_data)
    lib = load_zstd()
    dict_data = train_zstd_dict(lib, samples, dict_size=32768)
    compressed_data = zstd_compress(lib, raw_data, dict_data)
    print("%.3fKB" % (len(compressed_data) / 1000))
    return Compressed(
        raw_data=raw_data,
        dict_data=dict_data,
        compressed_data=compressed_data,
        image_data_len=len(image_data),
        palette_data_len=len(palette_data),
    )


def output_bytes(name, data, f):
    print("const uint8_t %s[] = {" % name, file=f)
    for i, byte in enumerate(data):
        if i % 12 == 0:
            print("\n", end="", file=f)
        print("0x%02X, " % byte, end="", file=f)
    print("\n};", file=f)


def output(compressed: Compressed, images: Images):
    with open(path.join(SCRIPT_DIR, "sprites.c"), "w") as f:
        print("#include <stddef.h>", file=f)
        print('#include "types.h"', file=f)
        print("const size_t sprites_zstd_dict_len = %d;" % len(compressed.dict_data), file=f)
        output_bytes("sprites_zstd_dict", compressed.dict_data, f)
        print("const size_t sprites_zstd_data_len = %d;" % len(compressed.compressed_data), file=f)
        output_bytes("sprites_zstd_data", compressed.compressed_data, f)

    with open(path.join(SCRIPT_DIR, "constants.h"), "w") as f:
        print("#define SHEET_COUNT %d" % len(images.frames), file=f)
        print("#define GROUP_COUNT %d" % len(images.groups), file=f)
        print("#define VARIANT_COUNT %d" % len(images.variants), file=f)
        print("#define SPRITE_COUNT %d" % len(images.images), file=f)
        print("#define ID_COUNT %d" % images.ids, file=f)
        print("#define MAX_PALETTES %d" % max(images.palette_counts), file=f)
        print("#define SPRITES_IMAGE_DATA_LEN %d" % compressed.image_data_len, file=f)
        print("#define SPRITES_PALETTE_DATA_LEN %d" % compressed.palette_data_len, file=f)
        print("#define SPRITES_RAW_LEN %d" % len(compressed.raw_data), file=f)


def main():
    images = read_images()
    compressed = compress_images(images)
    output(compressed, images)


if __name__ == "__main__":
    main()
