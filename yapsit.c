#include <argp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "types.h"

extern const Sprites sprites;

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
                               uint8_t palette_max, uint8_t palette[16][3]) {
  memset(palette[0], 0, sizeof(palette[0]));
  for (size_t i = 1; i <= palette_max; i++) {
    for (size_t j = 0; j < 3; j++)
      palette[i][j] = huffman_decode(color_context, bitstream) * 8 * 255 / 248;
  }
}

static void choose_palette(BitstreamContext *bitstream, uint8_t palette_max,
                           uint8_t palette[16][3]) {
  HuffmanContext color_context;
  huffman_init(&color_context, &sprites.palettes);
  size_t palettes = rand() % 16 == 0 ? 2 : 1;
  for (size_t cmap = 0; cmap < palettes; cmap++)
    decompress_palette(bitstream, &color_context, palette_max, palette);
}

static uint8_t *decompress_image(const Sprite *sprite,
                                 BitstreamContext *bitstream,
                                 uint8_t *palette_max) {
  size_t size = sprite->w * sprite->h;
  uint8_t *buf = malloc(size);
  if (!buf)
    return NULL;
  uint8_t *image = buf;
  HuffmanContext contexts[4];
  const HuffmanHeader *headers = &sprites.lz77.dys;
  for (size_t i = 0; i < 4; i++)
    huffman_init(&contexts[i], &headers[i]);
  *palette_max = 0;
  while (buf < image + size) {
    uint8_t dy = huffman_decode(&contexts[0], bitstream);
    uint8_t dx = huffman_decode(&contexts[1], bitstream);
    uint8_t runlen = huffman_decode(&contexts[2], bitstream);
    uint8_t value = huffman_decode(&contexts[3], bitstream);

    if (value > *palette_max)
      *palette_max = value;

    uint16_t delta = (sprite->w * dy) + dx - 128;
    if (delta == 0) {
      *(buf++) = value;
      continue;
    }
    // Manual copy instead of memcpy/memmove to handle overlapping ranges.
    for (size_t i = 0; i < runlen; i++)
      buf[i] = buf[i - delta];
    buf += runlen;
    if (buf < image + size)
      *(buf++) = value;
  }
  return image;
}

static char *out;

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

static void draw(const Sprite *sprite, const uint8_t *image,
                 const uint8_t palette[16][3]) {
  size_t size = (sprite->h + 1) / 2 * (sprite->w * 44 + 1) + 1;
  // TODO: check malloc.
  char *buf = out = malloc(size);
  for (size_t y = 0; y < sprite->h; y += 2) {
    for (size_t x = 0; x < sprite->w; x++) {
      uint8_t hi = image[y * sprite->w + x];
      uint8_t li = 0;
      if (y + 1 < sprite->h)
        li = image[(y + 1) * sprite->w + x];
      const uint8_t *h = palette[hi];
      const uint8_t *l = palette[li];
      if (hi && li) {
        output_str("\033[48;2");
        output_color(h);
        output_str("m\033[38;2");
        output_color(l);
        output_str("m\u2584");
      } else if (hi) {
        output_str("\033[38;2");
        output_color(h);
        output_str("m\u2580");
      } else if (li) {
        output_str("\033[38;2");
        output_color(l);
        output_str("m\u2584");
      } else {
        *out++ = ' ';
      }
      output_str("\033[m");
    }
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
      uint8_t palette_max;
      uint8_t *image = decompress_image(&images[i], &bitstream, &palette_max);
      if (!image)
        return 1;
      uint8_t palette[16][3];
      for (int j = 0; j < 2; j++) {
        decompress_palette(&bitstream, &color_context, palette_max, palette);
        draw(&images[i], image, palette);
      }
      free(image);
    }
  } else {
    srand(*(unsigned int *)getauxval(AT_RANDOM));

    struct winsize term_size;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &term_size);

    size_t offset;
    const Sprite *sprite =
        choose_sprite(term_size.ws_col, term_size.ws_row, &offset);
    if (sprite == NULL)
      return 1;
    BitstreamContext bitstream = {sprites.bitstream, offset};

    uint8_t palette_max;
    uint8_t *image = decompress_image(sprite, &bitstream, &palette_max);
    if (!image)
      return 1;

    uint8_t palette[16][3];
    choose_palette(&bitstream, palette_max, palette);

    draw(sprite, image, palette);
    free(image);
  }

  return 0;
}
