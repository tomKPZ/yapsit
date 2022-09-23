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
  BitstreamContext bits;
} HuffmanContext;

typedef struct {
  uint8_t w;
  uint8_t h;
  uint16_t colormap[15];
  uint16_t shiny[15];
  uint16_t deltas_size;
  uint16_t runlen_size;
  uint16_t values_size;
} Sprite;

#endif
