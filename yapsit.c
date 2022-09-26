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
extern const Lz77Header lz77;
extern const HuffmanHeader colormaps;

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
                         size_t offset) {
  BitstreamContext bitstream = {header->form, 0};
  const uint8_t *perm = header->perm;
  HuffmanBranch dummy;
  uint8_t i = decode_node(&bitstream, context->nodes, 0, &perm, &dummy);
  context->bits.bits = header->bits;
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

int main() {
  srand(*(unsigned int *)getauxval(AT_RANDOM));

  struct winsize w;
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

  size_t n = 0;
  const Sprite *sprite = NULL;
  size_t offsets[4] = {0};
  size_t color_offset = 0;
  size_t sprite_offsets[4] = {0};
  size_t sprite_color_offset = 0;
  size_t sprite_shiny_offset = 0;
  for (size_t i = 0; i < n_sprites; i++) {
    if (sprites[i].w <= w.ws_col && (sprites[i].h + 1) / 2 + 2 <= w.ws_row &&
        rand() % (++n) == 0) {
      sprite = &sprites[i];
      memcpy(sprite_offsets, offsets, sizeof(offsets));
      sprite_color_offset = color_offset;
      sprite_shiny_offset = color_offset + sprites[i].colormap_size;
    }
    const uint16_t *sizes = &sprites[i].dys_size;
    for (size_t j = 0; j < 4; j++)
      offsets[j] += sizes[j];
    color_offset += sprites[i].colormap_size + sprites[i].shiny_size;
  }
  if (!sprite)
    return 1;

  size_t colormap_offset =
      rand() % 16 == 0 ? sprite_shiny_offset : sprite_color_offset;
  uint8_t colormap[16][3];
  memset(colormap[0], 0, sizeof(colormap[0]));
  HuffmanContext color_context;
  huffman_init(&color_context, &colormaps, colormap_offset);
  for (size_t i = 1; i < 16; i++) {
    for (size_t j = 0; j < 3; j++)
      colormap[i][j] = huffman_decode(&color_context) * 8 * 255 / 248;
  }

  size_t size = sprite->w * sprite->h;
  uint8_t *buf = malloc(size);
  if (!buf)
    return 1;
  const uint8_t *image = buf;
  HuffmanContext contexts[4];
  const HuffmanHeader *headers = &lz77.dys;
  for (size_t i = 0; i < 4; i++)
    huffman_init(&contexts[i], &headers[i], sprite_offsets[i]);
  while (buf - image < size) {
    uint8_t decoded[4];
    for (size_t i = 0; i < 4; i++)
      decoded[i] = huffman_decode(&contexts[i]);
    uint8_t dy = decoded[0], dx = decoded[1], runlen = decoded[2],
            value = decoded[3];

    uint16_t delta = (sprite->w * dy) + dx - 128;
    if (delta == 0) {
      memset(buf, value, runlen);
      buf += runlen;
      continue;
    }
    memcpy(buf, buf - delta, runlen);
    buf += runlen;
    if (buf - image < size)
      *(buf++) = value;
  }

  for (size_t y = 0; y < sprite->h; y += 2) {
    for (size_t x = 0; x < sprite->w; x++) {
      uint8_t hi = image[y * sprite->w + x];
      uint8_t li = 0;
      if (y + 1 < sprite->h)
        li = image[(y + 1) * sprite->w + x];
      uint8_t *h = colormap[hi];
      uint8_t *l = colormap[li];
      if (hi && li)
        printf(BG FG "▄", h[0], h[1], h[2], l[0], l[1], l[2]);
      else if (hi)
        printf(FG "▀", h[0], h[1], h[2]);
      else if (li)
        printf(FG "▄", l[0], l[1], l[2]);
      else
        printf(" ");
      printf("\033[m");
    }
    printf("\n");
  }
  return 0;
}
