#include <assert.h>
#include <getopt.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "constants.h"
#include "types.h"

#define SHINY_NUMERATOR 1
#define SHINY_DENOMINATOR 16
#define RANGE_ARGS 6
#define DRAW_BUFFER 44294

static uint8_t max(uint8_t a, uint8_t b) { return a > b ? a : b; }

static bool read_bit(BitstreamContext *bitstream) {
  uint8_t byte = bitstream->bits[bitstream->offset / 8];
  bool bit = byte & (1 << (7 - bitstream->offset % 8));
  bitstream->offset += 1;
  return !!bit;
}

static uint8_t read_int(BitstreamContext *bitstream, size_t length) {
  uint8_t v = 0;
  for (size_t i = 0; i < length; i++)
    v = 2 * v + read_bit(bitstream);
  return v;
}

static uint8_t decode_node(BitstreamContext *bits, HuffmanNode *nodes,
                           uint8_t i, size_t *j, HuffmanBranch *parent) {
  parent->is_leaf = read_bit(bits);
  if (parent->is_leaf) {
    // TODO: interlace form and perm to avoid shuffling.
    parent->value = (*j)++;
    return 0;
  }
  parent->value = i;
  uint8_t l = decode_node(bits, nodes, i + 1, j, &nodes[i].l);
  uint8_t r = decode_node(bits, nodes, i + 1 + l, j, &nodes[i].r);
  return l + r + 1;
}

static void huffman_init(HuffmanContext *context, BitstreamContext *bitstream) {
  uint8_t bitlen = read_int(bitstream, 3) + 1;
  HuffmanBranch dummy;
  size_t perm_len = 0;
  decode_node(bitstream, context->nodes, 0, &perm_len, &dummy);
  uint8_t perm[256];
  for (size_t i = 0; i < perm_len; i++)
    perm[i] = read_int(bitstream, bitlen);
  for (HuffmanBranch *branch = &context->nodes[0].l; perm_len; branch++) {
    if (branch->is_leaf) {
      branch->value = perm[branch->value];
      perm_len--;
    }
  }
}

static uint8_t huffman_decode(const HuffmanContext *context,
                              BitstreamContext *bitstream) {
  const HuffmanNode *node = context->nodes;
  while (true) {
    if (read_bit(bitstream)) {
      if (node->r.is_leaf)
        return node->r.value;
      node = &context->nodes[node->r.value];
    } else {
      if (node->l.is_leaf)
        return node->l.value;
      node = &context->nodes[node->l.value];
    }
  }
}

static inline bool in_range(size_t x, const Range *range) {
  return x >= range->lo && x <= range->hi;
}

static bool any_variants_in_range(const uint8_t *variants, uint8_t count,
                                  const Range *range) {
  int max_v = 0;
  for (uint8_t i = 0; i < count; i++)
    max_v = max(max_v, variants[i]);
  return range->lo < max_v;
}

static uint32_t image_bitlen(const Sprite *sprite) {
  uint16_t bitlen = sprite->bitlen_h * 256 + sprite->bitlen_l;
  return bitlen < LARGE_LENS_COUNT ? sprites.large_lens[bitlen] : bitlen;
}

static const Sprite *choose_sprite(const Arguments *args, size_t *offset_out,
                                   uint8_t *z_out) {
  // TODO: Clean up this mess.
  size_t n = 0;
  const Sprite *sprite = NULL;
  size_t offset = 0;
  const Sprite *image = sprites.images;
  uint8_t sheet = 0;
  const uint8_t *variants = sprites.variants;
  size_t sprite_gid;
  uint8_t sprite_sheet;
  const uint8_t *sprite_variants;
  for (size_t gid = 0; gid < GROUP_COUNT; gid++) {
    bool sheet_in_range = false;
    bool frame_in_range = false;
    for (uint8_t s = 0; s < sprites.groups[gid]; s++) {
      sheet_in_range |= in_range(sheet + s, &args->sheet);
      frame_in_range |= args->frame.lo < sprites.frames[sheet + s];
    }
    for (size_t id = 0; id < sprites.limits[gid];
         id++, offset += image_bitlen(image), image++) {
      if (sheet_in_range && frame_in_range && in_range(id, &args->id) &&
          in_range(image->w - 1, &args->width) &&
          in_range(image->h - 1, &args->height) &&
          any_variants_in_range(variants, sprites.groups[gid],
                                &args->variants) &&
          rand() % ++n == 0) {
        sprite = image;
        *offset_out = offset;
        sprite_gid = gid;
        sprite_sheet = sheet;
        sprite_variants = variants;
      }
      variants += sprites.groups[gid];
    }
    sheet += sprites.groups[gid];
  }
  if (!sprite)
    return NULL;

  n = 0;
  image = NULL;
  for (size_t g = 0, z = 0; g < sprites.groups[sprite_gid];
       g++, sprite_variants++, sprite_sheet++) {
    for (size_t v = 0; v < *sprite_variants; v++) {
      for (uint8_t f = 0; f < sprites.frames[sprite_sheet]; f++, z++) {
        if (in_range(sprite_sheet, &args->sheet) &&
            in_range(v, &args->variants) && in_range(f, &args->frame) &&
            rand() % ++n == 0) {
          *z_out = z;
          image = sprite;
        }
      }
    }
  }
  return image;
}

static void decompress_palette(BitstreamContext *bitstream,
                               HuffmanContext *color_context,
                               uint8_t palette_max,
                               uint8_t palettes[2][16][3]) {
  for (size_t i = 1; i <= palette_max; i++) {
    for (size_t k = 0; k < 2; k++) {
      for (size_t j = 0; j < 3; j++) {
        palettes[k][i][j] =
            huffman_decode(color_context, bitstream) * 8 * 255 / 248;
      }
    }
  }
}

static void choose_palette(const Arguments *args, BitstreamContext *bitstream,
                           HuffmanContext *color_context, uint8_t palette_max,
                           uint8_t palette[16][3]) {
  uint8_t palettes[2][16][3];
  decompress_palette(bitstream, color_context, palette_max, palettes);
  bool shiny = rand() % args->denominator < args->numerator;
  memcpy(palette, palettes[shiny], sizeof(palettes[0]));
}

static void decompress_image(uint8_t *buf, uint8_t w, uint8_t h, uint8_t d,
                             BitstreamContext *bitstream,
                             const HuffmanContext contexts[5]) {
  size_t size = w * h * d;
  assert(size <= DECODE_BUFFER);
  uint8_t *image = buf;
  while (buf < image + size) {
    size_t offset = buf - image;
    uint8_t z = offset / (w * h);
    uint8_t y = offset % (w * h) / w;
    uint8_t x = offset % w;

    uint8_t dz = z ? huffman_decode(&contexts[0], bitstream) : 0;
    uint8_t dy = y || z ? huffman_decode(&contexts[1], bitstream) : 128;
    uint8_t dx = x || y || z ? huffman_decode(&contexts[2], bitstream) : 128;
    int8_t dxi = dx - 128;
    int8_t dyi = dy - 128;
    uint16_t delta = w * h * dz + w * dyi + dxi;
    size_t runlen = delta ? 1 + huffman_decode(&contexts[3], bitstream) : 0;

    // Manual copy instead of memcpy/memmove to handle overlapping ranges.
    for (size_t i = 0; i < runlen; i++)
      buf[i] = buf[i - delta];
    buf += runlen;

    if (buf < image + size)
      *(buf++) = huffman_decode(&contexts[4], bitstream);
  }
}

static uint8_t palette_max(uint8_t *image, uint8_t w, uint8_t h) {
  uint8_t res = 0;
  for (size_t i = 0; i < w * h; i++)
    res = max(res, image[i]);
  return res;
}

static char *out;
static uint8_t pfg = 0;
static uint8_t pbg = 0;

static void output_u8(uint8_t i) {
  out[2] = i % 10 + '0';
  i /= 10;
  out[1] = i % 10 + '0';
  i /= 10;
  out[0] = i + '0';
  out += 3;
}

static void output_color(const uint8_t color[3]) {
  for (size_t i = 0; i < 3; i++) {
    *out++ = ';';
    output_u8(color[i]);
  }
}

static void output_str(const char *str) {
  size_t len = strlen(str);
  memcpy(out, str, len);
  out += len;
}

static void fg(uint8_t i, const uint8_t palette[16][3]) {
  if (pfg == i)
    return;
  pfg = i;

  output_str("\033[38;2");
  output_color(palette[i]);
  *out++ = 'm';
}

static void bg(uint8_t i, const uint8_t palette[16][3]) {
  if (pbg == i)
    return;
  pbg = i;

  output_str("\033[48;2");
  output_color(palette[i]);
  *out++ = 'm';
}

static void reset() {
  if (pfg || pbg)
    output_str("\033[m");
  pfg = 0;
  pbg = 0;
}

static void draw(uint8_t w, uint8_t h, const uint8_t *image,
                 const uint8_t palette[16][3]) {
  char buf[DRAW_BUFFER];
  out = buf;
  for (size_t y = 0; y < h; y += 2) {
    for (size_t x = 0; x < w; x++) {
      uint8_t u = image[y * w + x];
      uint8_t l = 0;
      if (y + 1 < h)
        l = image[(y + 1) * w + x];
      if (u && l) {
        if ((pbg == u) + (pfg == l) > (pfg == u) + (pbg == l)) {
          fg(l, palette);
          bg(u, palette);
          output_str("\u2584");
        } else {
          fg(u, palette);
          bg(l, palette);
          output_str("\u2580");
        }
      } else if (u) {
        if (pbg)
          reset();
        fg(u, palette);
        output_str("\u2580");
      } else if (l) {
        if (pbg)
          reset();
        fg(l, palette);
        output_str("\u2584");
      } else {
        reset();
        *out++ = ' ';
      }
    }
    reset();
    *out++ = '\n';
  }
  *out++ = 0;
  assert(out - buf <= DRAW_BUFFER);
  puts(buf);
}

static bool parse_u32(const char *s, uint32_t *i) {
  *i = 0;
  for (; *s; s++) {
    if (*s > '9' || *s < '0')
      return false;
    uint64_t j = *i * 10 + *s - '0';
    if (j > 0xffffffff)
      return false;
    *i = j;
  }
  return true;
}

static bool parse_range_endpoint(const char *s, uint16_t *i) {
  uint32_t j;
  if (!parse_u32(s, &j))
    return false;
  *i = j - 1;
  return j != 0 && j <= 0xffff;
}

static bool parse_range(char *arg, Range *range) {
  char *upper = strchr(arg, '-');
  if (upper)
    *upper++ = '\0';
  else
    upper = arg;
  return parse_range_endpoint(arg, &range->lo) &&
         parse_range_endpoint(upper, &range->hi);
}

static void init_range(Range *range) {
  range->lo = 0;
  range->hi = 0xffff;
}

static const char usage[] =
    "Usage: yapsit [OPTION...]\n"
    "Show a random pokemon sprite.\n"
    "\n"
    "  -i, --id=ID[-ID]           Filter by ID\n"
    "  -s, --sheet=SID[-SID]      Filter by sprite sheet\n"
    "  -v, --variants=VID[-VID]   Filter by variant ID\n"
    "  -f, --frame=FID[-FID]      Filter by frame number\n"
    "  -W, --width=W[-MAX]        Filter by sprite width\n"
    "  -H, --height=H[-MAX]       Filter by sprite height\n"
    "  -n, --numerator=N          Shiny chance numerator\n"
    "  -d, --denominator=D        Shiny chance denominator\n"
    "  -t, --test                 Output all sprites\n"
    "  -h, --help                 Give this help list\n";

static struct option options[] = {
    {"id", required_argument, 0, 'i'},
    {"sheet", required_argument, 0, 's'},
    {"variants", required_argument, 0, 'v'},
    {"frame", required_argument, 0, 'f'},
    {"width", required_argument, 0, 'W'},
    {"height", required_argument, 0, 'H'},
    {"numerator", required_argument, 0, 'n'},
    {"denominator", required_argument, 0, 'd'},
    {"test", no_argument, 0, 't'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0},
};

static void init_args(Arguments *args, int argc, char *argv[]) {
  for (size_t i = 0; i < RANGE_ARGS; i++)
    init_range(&args->id + i);
  args->numerator = SHINY_NUMERATOR;
  args->denominator = SHINY_DENOMINATOR;
  args->test = false;
  while (true) {
    switch (getopt_long(argc, argv, "i:s:v:f:W:H:n:d:th", options, NULL)) {
    case -1:
      return;
    case 'i':
      if (!parse_range(optarg, &args->id))
        exit(EXIT_FAILURE);
      break;
    case 's':
      if (!parse_range(optarg, &args->sheet))
        exit(EXIT_FAILURE);
      break;
    case 'v':
      if (!parse_range(optarg, &args->variants))
        exit(EXIT_FAILURE);
      break;
    case 'f':
      if (!parse_range(optarg, &args->frame))
        exit(EXIT_FAILURE);
      break;
    case 'W':
      if (!parse_range(optarg, &args->width))
        exit(EXIT_FAILURE);
      break;
    case 'H':
      if (!parse_range(optarg, &args->height))
        exit(EXIT_FAILURE);
      break;
    case 'n':
      if (!parse_u32(optarg, &args->numerator))
        exit(EXIT_FAILURE);
      break;
    case 'd':
      if (!parse_u32(optarg, &args->denominator))
        exit(EXIT_FAILURE);
      break;
    case 't':
      args->test = true;
      break;
    case 'h':
      puts(usage);
      exit(EXIT_SUCCESS);
    case '?':
    default:
      fputs(usage, stderr);
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char *argv[]) {
  Arguments args;
  init_args(&args, argc, argv);

  BitstreamContext bitstream = {sprites.bitstream, 0};
  HuffmanContext color_context;
  huffman_init(&color_context, &bitstream);
  HuffmanContext contexts[5];
  for (size_t i = 0; i < sizeof(contexts) / sizeof(contexts[0]); i++)
    huffman_init(&contexts[i], &bitstream);
  uint8_t image[DECODE_BUFFER];

  if (args.test) {
    const Sprite *images = sprites.images;
    for (size_t i = 0; i < SPRITE_COUNT; i++) {
      uint8_t w = images[i].w, h = images[i].h, d = images[i].d;
      decompress_image(image, w, h, d, &bitstream, contexts);
      uint8_t palettes[2][16][3];
      for (size_t z = 0; z < d; z++) {
        uint8_t *frame = image + w * h * z;
        decompress_palette(&bitstream, &color_context, palette_max(frame, w, h),
                           palettes);
        for (int j = 0; j < 2; j++)
          draw(w, h, frame, palettes[j]);
      }
    }
  } else {
    srand(time(NULL) ^ getpid());

    size_t offset;
    uint8_t z;
    const Sprite *sprite = choose_sprite(&args, &offset, &z);
    if (sprite == NULL)
      return EXIT_FAILURE;
    uint8_t w = sprite->w, h = sprite->h, d = sprite->d;
    bitstream.offset += offset;

    decompress_image(image, w, h, d, &bitstream, contexts);

    uint8_t palette[16][3];
    for (size_t p = 0; p <= z; p++) {
      choose_palette(&args, &bitstream, &color_context,
                     palette_max(image + w * h * p, w, h), palette);
    }

    draw(w, h, image + w * h * z, palette);
  }

  return EXIT_SUCCESS;
}
