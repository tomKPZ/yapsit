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

static uint8_t decode_node(BitstreamContext *bits, HuffmanNode *nodes,
                           uint8_t i, const uint8_t **perm,
                           HuffmanBranch *parent) {
  if (read_bit(bits)) {
    parent->is_leaf = true;
    parent->value = **perm;
    (*perm)++;
    return 0;
  }
  parent->is_leaf = false;
  parent->value = i;
  uint8_t l = decode_node(bits, nodes, i + 1, perm, &nodes[i].l);
  uint8_t r = decode_node(bits, nodes, i + 1 + l, perm, &nodes[i].r);
  return l + r + 1;
}

static void huffman_init(HuffmanContext *context, const HuffmanHeader *header) {
  BitstreamContext bitstream = {header->form, 0};
  const uint8_t *perm = header->perm;
  HuffmanBranch dummy;
  decode_node(&bitstream, context->nodes, 0, &perm, &dummy);
}

static uint8_t huffman_decode(HuffmanContext *context,
                              BitstreamContext *bitstream) {
  HuffmanNode *node = context->nodes;
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

static const Sprite *choose_sprite(int max_w, int max_h, size_t *bit_offset) {
  size_t n = 0;
  const Sprite *sprite = NULL;
  size_t offset = 0;
  const Sprite *images = sprites.images;
  for (size_t i = 0; i < sprites.count; i++) {
    if (images[i].w <= max_w && (images[i].h + 1) / 2 + 2 <= max_h &&
        rand() % (++n) == 0) {
      sprite = &images[i];
      *bit_offset = offset;
    }
    offset += images[i].bitlen;
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

static void choose_palette(BitstreamContext *bitstream, uint8_t palette_max,
                           uint8_t palette[16][3]) {
  HuffmanContext color_context;
  huffman_init(&color_context, &sprites.palettes);
  uint8_t palettes[2][16][3];
  decompress_palette(bitstream, &color_context, palette_max, palettes);
  memcpy(palette, palettes[rand() % 16 == 0], sizeof(palettes[0]));
}

static uint8_t *decompress_image(uint8_t w, uint8_t h, uint8_t d,
                                 BitstreamContext *bitstream) {
  size_t size = w * h * d;
  uint8_t *buf = checked_malloc(size);
  uint8_t *image = buf;
  HuffmanContext contexts[5];
  const HuffmanHeader *headers = &sprites.lz77.dzs;
  for (size_t i = 0; i < sizeof(contexts) / sizeof(contexts[0]); i++)
    huffman_init(&contexts[i], &headers[i]);
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
    uint8_t runlen = delta ? huffman_decode(&contexts[3], bitstream) : 0;

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
    {"test", 't', 0, 0, "Output all sprites.", 0},
    {0},
};

struct arguments {
  bool test;
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  (void)arg;
  struct arguments *arguments = state->input;
  switch (key) {
  case 't':
    arguments->test = true;
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

int main(int argc, char *argv[]) {
  struct arguments arguments;
  arguments.test = false;
  if (argp_parse(&argp, argc, argv, 0, 0, &arguments))
    return 1;

  if (arguments.test) {
    const Sprite *images = sprites.images;
    BitstreamContext bitstream = {sprites.bitstream, 0};
    HuffmanContext color_context;
    huffman_init(&color_context, &sprites.palettes);
    for (size_t i = 0; i < sprites.count; i++) {
      uint8_t w = images[i].w, h = images[i].h, d = images[i].d;
      uint8_t *image = decompress_image(w, h, d, &bitstream);
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

    struct winsize term_size;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &term_size);

    size_t offset;
    const Sprite *sprite =
        choose_sprite(term_size.ws_col, term_size.ws_row, &offset);
    if (sprite == NULL)
      return 1;
    uint8_t w = sprite->w, h = sprite->h, d = sprite->d;
    BitstreamContext bitstream = {sprites.bitstream, offset};

    uint8_t *image = decompress_image(w, h, d, &bitstream);

    uint8_t palette[16][3];
    choose_palette(&bitstream, palette_max(image, w, h), palette);

    draw(w, h, image, palette);
    free(image);
  }

  return 0;
}
