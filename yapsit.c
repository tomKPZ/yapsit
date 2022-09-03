#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/auxv.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define FG "\033[38;2;%d;%d;%dm"
#define BG "\033[48;2;%d;%d;%dm"

struct Sprite {
  unsigned int w;
  unsigned int h;
  const uint8_t *image;
  const uint16_t colormap[16];
  const uint16_t shiny[16];
} const sprites[] = {
#include "pokemon.h"
};

const size_t n_sprites = sizeof(sprites) / sizeof(sprites[0]);

static uint8_t pixel(const struct Sprite *sprite, size_t x, size_t y) {
  size_t i = y * sprite->w + x;
  if (i >= sprite->w * sprite->h)
    return 0;
  uint8_t val = sprite->image[i / 2];
  return i % 2 == 0 ? val >> 4 : val & 0x0F;
}

static bool A(uint16_t c) { return c >> 15; }
static uint8_t R(uint16_t c) { return ((c >> 10) & 0b11111) * 8 * 255 / 248; }
static uint8_t G(uint16_t c) { return ((c >> 5) & 0b11111) * 8 * 255 / 248; }
static uint8_t B(uint16_t c) { return (c & 0b11111) * 8 * 255 / 248; }

int main() {
  srand(*(unsigned int *)getauxval(AT_RANDOM));

  struct winsize w;
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

  size_t n = 0;
  const struct Sprite *sprite = NULL;
  for (size_t i = 0; i < n_sprites; i++) {
    if (sprites[i].w > w.ws_col || (sprites[i].h + 1) / 2 + 2 > w.ws_row)
      continue;
    if (rand() % (++n) == 0)
      sprite = &sprites[i];
  }
  if (!sprite)
    return 0;

  const uint16_t *colormap =
      rand() % 16 == 0 ? sprite->shiny : sprite->colormap;

  for (size_t y = 0; y < sprite->h; y += 2) {
    for (size_t x = 0; x < sprite->w; x++) {
      uint16_t h = colormap[pixel(sprite, x, y)];
      uint16_t l = colormap[pixel(sprite, x, y + 1)];
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
    printf("\n");
  }
  return 0;
}
