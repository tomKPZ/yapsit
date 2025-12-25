#ifndef TYPES_H_
#define TYPES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "constants.h"

typedef struct {
  uint8_t w;
  uint8_t h;
  uint8_t d;
} Sprite;

typedef struct {
  uint16_t limits[GROUP_COUNT];
  uint8_t variants[VARIANT_COUNT];
  uint8_t groups[GROUP_COUNT];
  uint8_t frames[SHEET_COUNT];
  uint8_t palette_counts[GROUP_COUNT];
  Sprite images[SPRITE_COUNT];
  uint32_t offsets[SPRITE_COUNT];
  uint32_t sizes[SPRITE_COUNT];
  uint8_t dictionary[DICT_LEN];
  uint8_t compressed[COMPRESSED_LEN];
} Sprites;

typedef struct {
  uint16_t lo;
  uint16_t hi;
} Range;

typedef struct {
  uint32_t numerator;
  uint32_t denominator;
  Range id;
  Range sheet;
  Range variants;
  Range frame;
  Range width;
  Range height;
  bool test;
} Arguments;

typedef struct {
  const Sprite *sprite;
  const uint8_t *variants;
  uint8_t gid;
  uint8_t sheet;
  uint8_t z;
  uint32_t index;
} SpriteContext;

extern const Sprites sprites;

#endif
