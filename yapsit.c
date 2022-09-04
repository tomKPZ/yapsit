#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/auxv.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "types.h"

extern const Sprite sprites[];
extern const size_t n_sprites;
extern const uint8_t encoded[];

#define FG "\033[38;2;%d;%d;%dm"
#define BG "\033[48;2;%d;%d;%dm"

static bool read_bit(BitstreamContext *bitstream) {
  uint8_t byte = bitstream->bits[bitstream->offset / 8];
  bool bit = byte & (1 << (7 - bitstream->offset % 8));
  bitstream->offset += 1;
  return !!bit;
}

static uint8_t decode_node(BitstreamContext *bits, HuffmanNode *nodes,
                           uint8_t i, uint8_t **perm) {
  if (read_bit(bits)) {
    nodes[i].is_leaf = true;
    nodes[i].data.val = **perm;
    (*perm)++;
    return i;
  }
  uint8_t l = decode_node(bits, nodes, i, perm);
  uint8_t r = decode_node(bits, nodes, l + 1, perm);
  i = r + 1;
  nodes[i].is_leaf = false;
  nodes[i].data.node.l = l;
  nodes[i].data.node.r = r;
  return i;
}

static void huffman_init(HuffmanContext *context, const HuffmanHeader *header,
                         size_t offset) {
  uint8_t perms[16];
  for (int i = 0; i < 8; i++) {
    perms[2 * i] = header->perm[i] >> 4;
    perms[2 * i + 1] = header->perm[i] & 0x0F;
  }
  BitstreamContext bits = {header->form, 0};
  uint8_t *perm = perms;
  decode_node(&bits, context->nodes, 0, &perm);
  context->bits.bits = encoded;
  context->bits.offset = offset;
}

static uint8_t huffman_decode(HuffmanContext *context) {
  HuffmanNode *node = &context->nodes[30];
  while (!node->is_leaf) {
    if (read_bit(&context->bits))
      node = &context->nodes[node->data.node.r];
    else
      node = &context->nodes[node->data.node.l];
  }
  return node->data.val;
}

static void runlength_init(RunlengthContext *context,
                           const HuffmanHeader *counts,
                           const HuffmanHeader *values, size_t offset) {
  context->count = 0;
  context->value = 0;
  huffman_init(&context->counts, counts, offset);
  huffman_init(&context->values, values, offset + counts->size);
}

static uint8_t runlength_decode(RunlengthContext *context) {
  if (!context->count) {
    context->count = huffman_decode(&context->counts) + 1;
    context->value = huffman_decode(&context->values);
  }
  context->count--;
  return context->value;
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
  size_t offset = 0;
  const Sprite *sprite = NULL;
  size_t sprite_offset = 0;
  for (size_t i = 0; i < n_sprites; i++) {
    if (sprites[i].w <= w.ws_col && (sprites[i].h + 1) / 2 + 2 <= w.ws_row &&
        rand() % (++n) == 0) {
      sprite = &sprites[i];
      sprite_offset = offset;
    }
    offset += sprites[i].counts.size + sprites[i].values.size;
  }
  if (!sprite)
    return 0;

  const uint16_t *colormap =
      rand() % 16 == 0 ? sprite->shiny : sprite->colormap;

  RunlengthContext t;
  runlength_init(&t, &sprite->counts, &sprite->values, sprite_offset);
  for (size_t y = 0; y < sprite->h; y += 2) {
    RunlengthContext b = t;
    for (size_t x = 0; x < sprite->w; x++)
      runlength_decode(&b);
    for (size_t x = 0; x < sprite->w; x++) {
      uint16_t h = color(colormap, runlength_decode(&t));
      uint16_t l = 0;
      if (y + 1 < sprite->h)
        l = color(colormap, runlength_decode(&b));
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
    t = b;
    printf("\n");
  }
  return 0;
}
