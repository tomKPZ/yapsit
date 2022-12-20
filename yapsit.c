#include <argp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include "constants.h"
#include "types.h"

extern const Sprites sprites;

static void *checked_malloc(size_t size) {
  void *mem = malloc(size);
  if (!mem)
    exit(1);
  return mem;
}

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
  HuffmanBranch dummy;
  size_t perm_len = 0;
  decode_node(bitstream, context->nodes, 0, &perm_len, &dummy);
  uint8_t perm[256];
  for (size_t i = 0; i < perm_len; i++)
    perm[i] = read_int(bitstream, 8);
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

static const Sprite *choose_sprite(const Arguments *args, int max_w, int max_h,
                                   size_t *offset_out, uint8_t *z_out) {
  size_t n = 0;
  const Sprite *sprite = NULL;
  size_t offset = 0;
  const Sprite *image = sprites.images;
  const uint8_t *variants = sprites.variants;
  uint8_t sheet = 0;
  for (size_t gid = 0; gid < GROUP_COUNT; sheet += sprites.groups[gid], gid++) {
    for (size_t id = 0; id < sprites.limits[gid];
         id++, offset += image->bitlen, image++) {
      for (size_t g = 0, z = 0, s = sheet; g < sprites.groups[gid];
           g++, variants++, s++) {
        for (size_t v = 0; v < *variants; v++) {
          for (size_t f = 0; f < sprites.frames[s]; f++, z++) {
            if (in_range(id, &args->id) && in_range(s, &args->sheet) &&
                in_range(v, &args->variants) && in_range(f, &args->frame) &&
                image->w <= max_w && (image->h + 1) / 2 + 2 <= max_h &&
                rand() % (++n) == 0) {
              sprite = image;
              *offset_out = offset;
              *z_out = z;
            }
          }
        }
      }
    }
  }
  return sprite;
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

static void choose_palette(BitstreamContext *bitstream,
                           HuffmanContext *color_context, uint8_t palette_max,
                           uint8_t palette[16][3]) {
  uint8_t palettes[2][16][3];
  decompress_palette(bitstream, color_context, palette_max, palettes);
  memcpy(palette, palettes[rand() % 16 == 0], sizeof(palettes[0]));
}

static uint8_t *decompress_image(uint8_t w, uint8_t h, uint8_t d,
                                 BitstreamContext *bitstream,
                                 const HuffmanContext contexts[5]) {
  size_t size = w * h * d;
  uint8_t *buf = checked_malloc(size);
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
  return image;
}

static uint8_t max(uint8_t a, uint8_t b) { return a > b ? a : b; }

static uint8_t palette_max(uint8_t *image, uint8_t w, uint8_t h) {
  uint8_t res = 0;
  for (size_t i = 0; i < w * h; i++)
    res = max(res, image[i]);
  return res;
}

static char *out;
static uint8_t pfg = 0;
static uint8_t pbg = 0;

static void itoa(uint8_t i) {
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
    itoa(color[i]);
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
  size_t size = (h + 1) / 2 * (w * 44 + 1) + 1;
  char *buf = out = checked_malloc(size);
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
  printf("%s", buf);
  free(buf);
}

static struct argp_option options[] = {
    {"id", 'i', "ID[-ID]", 0, "Filter by ID", 0},
    {"sheet", 's', "SID[-SID]", 0, "Filter by sprite sheet", 0},
    {"variants", 'v', "VID[-VID]", 0, "Filter by variant ID", 0},
    {"frame", 'f', "FID[-FID]", 0, "Filter by frame number", 0},
    {"test", 't', 0, 0, "Output all sprites", 1},
    {0},
};

static bool parse_u16(const char *s, uint16_t *i) {
  unsigned long len = strlen(s);
  if (len <= 0 || len > 5)
    return false;
  for (const char *t = s; *t; t++) {
    if (*t < '0' || *t > '9')
      return false;
  }
  int j = atoi(s);
  if (j <= 0 || j > 65536)
    return false;
  *i = j - 1;
  return true;
}

static bool parse_range(char *arg, Range *range) {
  char *upper = strchr(arg, '-');
  if (upper)
    *upper++ = '\0';
  else
    upper = arg;
  return parse_u16(arg, &range->lo) && parse_u16(upper, &range->hi);
}

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  Arguments *args = state->input;
  switch (key) {
  case 'i':
    if (!parse_range(arg, &args->id))
      return ERANGE;
    break;
  case 's':
    if (!parse_range(arg, &args->sheet))
      return ERANGE;
    break;
  case 'v':
    if (!parse_range(arg, &args->variants))
      return ERANGE;
    break;
  case 'f':
    if (!parse_range(arg, &args->frame))
      return ERANGE;
    break;
  case 't':
    args->test = true;
    break;
  case ARGP_KEY_ARG:
    return 0;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {
    options, parse_opt, 0, "Show a random pokemon sprite.", 0, 0, 0};

static void init_range(Range *range) {
  range->lo = 0;
  range->hi = 0xffff;
}

int main(int argc, char *argv[]) {
  Arguments args;
  init_range(&args.id);
  init_range(&args.sheet);
  init_range(&args.variants);
  init_range(&args.frame);
  args.test = false;
  if (argp_parse(&argp, argc, argv, 0, 0, &args))
    return 1;

  BitstreamContext bitstream = {sprites.bitstream, 0};
  HuffmanContext color_context;
  huffman_init(&color_context, &bitstream);
  HuffmanContext contexts[5];
  for (size_t i = 0; i < sizeof(contexts) / sizeof(contexts[0]); i++)
    huffman_init(&contexts[i], &bitstream);

  if (args.test) {
    const Sprite *images = sprites.images;
    for (size_t i = 0; i < SPRITE_COUNT; i++) {
      uint8_t w = images[i].w, h = images[i].h, d = images[i].d;
      uint8_t *image = decompress_image(w, h, d, &bitstream, contexts);
      uint8_t palettes[2][16][3];
      for (size_t z = 0; z < d; z++) {
        uint8_t *frame = image + w * h * z;
        decompress_palette(&bitstream, &color_context, palette_max(frame, w, h),
                           palettes);
        for (int j = 0; j < 2; j++)
          draw(w, h, frame, palettes[j]);
      }
      free(image);
    }
  } else {
    srand(time(NULL) ^ getpid());

    // TODO: Replace this with command line args.
    struct winsize term_size;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &term_size);

    size_t offset;
    uint8_t z;
    const Sprite *sprite =
        choose_sprite(&args, term_size.ws_col, term_size.ws_row, &offset, &z);
    if (sprite == NULL)
      return 1;
    uint8_t w = sprite->w, h = sprite->h, d = sprite->d;
    bitstream.offset += offset;

    uint8_t *image = decompress_image(w, h, d, &bitstream, contexts);

    uint8_t palette[16][3];
    for (size_t p = 0; p <= z; p++) {
      choose_palette(&bitstream, &color_context,
                     palette_max(image + w * h * p, w, h), palette);
    }

    draw(w, h, image + w * h * z, palette);
    free(image);
  }

  return 0;
}
