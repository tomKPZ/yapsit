#ifndef TYPES_H_
#define TYPES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
  uint8_t form[4];
  uint8_t perm[8];
} HuffmanHeader;

typedef struct {
  bool is_leaf;
  union {
    struct {
      uint8_t l;
      uint8_t r;
    } node;
    uint8_t val;
  } data;
} HuffmanNode;

typedef struct {
  const uint8_t *bits;
  size_t offset;
} BitstreamContext;

typedef struct {
  HuffmanNode nodes[31];
  BitstreamContext bits;
} HuffmanContext;

typedef struct {
  uint8_t count;
  uint8_t value;
  HuffmanContext counts;
  HuffmanContext values;
} RunlengthContext;

typedef struct {
  uint8_t w;
  uint8_t h;
  uint16_t colormap[15];
  uint16_t shiny[15];
  uint16_t count_size;
  uint16_t value_size;
} Sprite;

#endif
