#ifndef TYPES_H_
#define TYPES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
  uint8_t form[64];
  uint8_t perm[256];
} HuffmanHeader;

typedef struct {
  const HuffmanHeader dys;
  const HuffmanHeader dxs;
  const HuffmanHeader runlen;
  const HuffmanHeader values;
} Lz77Header;

typedef struct {
  bool is_leaf;
  uint8_t value;
} HuffmanBranch;

typedef struct {
  HuffmanBranch l;
  HuffmanBranch r;
} HuffmanNode;

typedef struct {
  const uint8_t *bits;
  size_t offset;
} BitstreamContext;

typedef struct {
  HuffmanNode nodes[256];
} HuffmanContext;

typedef struct {
  uint8_t w;
  uint8_t h;
  uint16_t bitlen;
} Sprite;

typedef struct {
  const Sprite *images;
  const size_t count;
  const Lz77Header lz77;
  const HuffmanHeader colormaps;
  const uint8_t *bitstream;
} Sprites;

#endif
