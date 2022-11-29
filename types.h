#ifndef TYPES_H_
#define TYPES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "constants.h"

typedef struct {
  uint8_t form[64];
  uint8_t perm[256];
} HuffmanHeader;

typedef struct {
  const HuffmanHeader dzs;
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
  uint8_t d;
  uint16_t bitlen;
} Sprite;

typedef struct {
  const Sprite images[SPRITE_COUNT];
  const uint16_t count;
  const uint16_t ids;
  const Lz77Header lz77;
  const HuffmanHeader palettes;
  const uint8_t bitstream[BITSTREAM_LEN];
  const uint8_t variants[VARIANT_COUNT];
  const uint16_t limits[GROUP_COUNT];
  const uint8_t groups[GROUP_COUNT];
  const uint8_t frames[SHEET_COUNT];
} Sprites;

typedef struct {
  uint16_t id_lo;
  uint16_t id_hi;
  uint16_t sheet_lo;
  uint16_t sheet_hi;
  uint16_t variants_lo;
  uint16_t variants_hi;
  uint16_t frame_lo;
  uint16_t frame_hi;
  bool test;
} Arguments;

#endif
