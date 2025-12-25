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
  uint32_t image_offset;
  uint32_t palette_offset;
} Sprite;

typedef struct {
  const uint16_t *limits;
  const uint8_t *variants;
  const uint8_t *groups;
  const uint8_t *frames;
  const uint8_t *palette_counts;
  const Sprite *images;
  const uint8_t *image_data;
  const uint8_t *palette_data;
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
} SpriteContext;

extern Sprites sprites;
extern const uint8_t sprites_zstd_dict[];
extern const size_t sprites_zstd_dict_len;
extern const uint8_t sprites_zstd_data[];
extern const size_t sprites_zstd_data_len;

#endif
