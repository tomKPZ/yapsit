#ifndef TYPES_H_
#define TYPES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "constants.h"

typedef struct {
  const uint8_t *bits;
  size_t offset;
} BitstreamContext;

typedef struct {
  bool is_leaf;
  uint8_t value;
} HuffmanBranch;

typedef struct {
  HuffmanBranch l;
  HuffmanBranch r;
} HuffmanNode;

typedef struct {
  HuffmanNode nodes[256];
} HuffmanContext;

typedef struct {
  uint8_t w;
  uint8_t h;
  uint8_t d;
  uint8_t bitlen_h;
  uint8_t bitlen_l;
} Sprite;

typedef struct {
  Sprite images[SPRITE_COUNT];
  uint8_t bitstream[BITSTREAM_LEN];
  uint8_t variants[VARIANT_COUNT];
  uint16_t limits[GROUP_COUNT];
  uint8_t groups[GROUP_COUNT];
  uint8_t frames[SHEET_COUNT];
  uint32_t large_lens[LARGE_LENS_COUNT];
} Sprites;

typedef struct {
  uint16_t lo;
  uint16_t hi;
} Range;

typedef struct {
  Range id;
  Range sheet;
  Range variants;
  Range frame;
  Range width;
  Range height;
  uint32_t numerator;
  uint32_t denominator;
  bool test;
} Arguments;

#endif
