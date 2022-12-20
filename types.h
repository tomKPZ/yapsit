#ifndef TYPES_H_
#define TYPES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "constants.h"

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
  // TODO: this can be removed
  uint8_t d;
  // TODO: try to reduce to uint16_t.
  uint32_t bitlen;
} Sprite;

typedef struct {
  Sprite images[SPRITE_COUNT];
  uint8_t bitstream[BITSTREAM_LEN];
  uint8_t variants[VARIANT_COUNT];
  uint16_t limits[GROUP_COUNT];
  uint8_t groups[GROUP_COUNT];
  uint8_t frames[SHEET_COUNT];
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
  bool test;
} Arguments;

#endif
