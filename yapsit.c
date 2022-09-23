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

extern const Sprite sprites[];
extern const size_t n_sprites;
extern const uint8_t deltas_bits[];
extern const uint8_t runlen_bits[];
extern const uint8_t values_bits[];
extern const HuffmanHeader deltas_header;
extern const HuffmanHeader runlen_header;
extern const HuffmanHeader values_header;

#define FG "\033[38;2;%d;%d;%dm"
#define BG "\033[48;2;%d;%d;%dm"

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

static void huffman_init(HuffmanContext *context, const HuffmanHeader *header,
                         const uint8_t *bits, size_t offset) {
  BitstreamContext bitstream = {header->form, 0};
  const uint8_t *perm = header->perm;
  HuffmanBranch dummy;
  uint8_t i = decode_node(&bitstream, context->nodes, 0, &perm, &dummy);
  context->bits.bits = bits;
  context->bits.offset = offset;
}

static uint8_t huffman_decode(HuffmanContext *context) {
  HuffmanNode *node = context->nodes;
  while (true) {
    if (read_bit(&context->bits)) {
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

static bool A(uint16_t c) { return c >> 15; }
static uint8_t R(uint16_t c) { return ((c >> 10) & 0b11111) * 8 * 255 / 248; }
static uint8_t G(uint16_t c) { return ((c >> 5) & 0b11111) * 8 * 255 / 248; }
static uint8_t B(uint16_t c) { return (c & 0b11111) * 8 * 255 / 248; }

static uint16_t color(const uint16_t *colormap, uint8_t i) {
  return i ? colormap[i - 1] : 0;
}

int main() {
  srand(*(unsigned int *)getauxval(AT_RANDOM));

  struct winsize w;
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

  size_t n = 0;
  size_t deltas_offset = 0;
  size_t runlen_offset = 0;
  size_t values_offset = 0;
  const Sprite *sprite = NULL;
  size_t sprite_deltas_offset = 0;
  size_t sprite_runlen_offset = 0;
  size_t sprite_values_offset = 0;
  for (size_t i = 0; i < n_sprites; i++) {
    if (sprites[i].w <= w.ws_col && (sprites[i].h + 1) / 2 + 2 <= w.ws_row &&
        rand() % (++n) == 0) {
      sprite = &sprites[i];
      sprite_deltas_offset = deltas_offset;
      sprite_runlen_offset = runlen_offset;
      sprite_values_offset = values_offset;
    }
    deltas_offset += sprites[i].deltas_size;
    runlen_offset += sprites[i].runlen_size;
    values_offset += sprites[i].values_size;
  }
  if (!sprite)
    return 1;

  const uint16_t *colormap =
      rand() % 16 == 0 ? sprite->shiny : sprite->colormap;

  size_t size = sprite->w * sprite->h;
  uint8_t *buf = malloc(size);
  if (!buf)
    return 1;
  const uint8_t *image = buf;
  HuffmanContext deltas_context;
  huffman_init(&deltas_context, &deltas_header, deltas_bits,
               sprite_deltas_offset);
  HuffmanContext runlen_context;
  huffman_init(&runlen_context, &runlen_header, runlen_bits,
               sprite_runlen_offset);
  HuffmanContext values_context;
  huffman_init(&values_context, &values_header, values_bits,
               sprite_values_offset);
  while (buf - image < size) {
    uint8_t delta = huffman_decode(&deltas_context);
    uint8_t runlen = huffman_decode(&runlen_context);
    uint8_t value = huffman_decode(&values_context);
    if (delta == 0 && runlen == 0) {
      *(buf++) = value;
      continue;
    }
    memcpy(buf, buf - delta, runlen);
    buf += runlen;
    if (buf - image < size)
      *(buf++) = value;
  }

  for (size_t y = 0; y < sprite->h; y += 2) {
    for (size_t x = 0; x < sprite->w; x++) {
      uint16_t h = color(colormap, image[y * sprite->w + x]);
      uint16_t l = 0;
      if (y + 1 < sprite->h)
        l = color(colormap, image[(y + 1) * sprite->w + x]);
      if (A(h) && A(l))
        printf(BG FG "▄", R(h), G(h), B(h), R(l), G(l), B(l));
      else if (A(h))
        printf(FG "▀", R(h), G(h), B(h));
      else if (A(l))
        printf(FG "▄", R(l), G(l), B(l));
      else
        printf(" ");
      printf("\033[m");
    }
    printf("\n");
  }
  return 0;
}
