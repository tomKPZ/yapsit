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
  const uint8_t colormap[16][4];
  const uint8_t shiny[16][4];
} const sprites[] = {
#include "pokemon.h"
};

const size_t n_sprites = sizeof(sprites) / sizeof(sprites[0]);

uint8_t pixel(const struct Sprite *sprite, size_t x, size_t y) {
  size_t i = y * sprite->w + x;
  if (i >= sprite->w * sprite->h)
    return 0;
  uint8_t val = sprite->image[i / 2];
  return i % 2 == 0 ? val >> 4 : val & 0x0F;
}

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

  const uint8_t(*colormap)[4] =
      rand() % 16 == 0 ? sprite->shiny : sprite->colormap;

  for (size_t y = 0; y < sprite->h; y += 2) {
    for (size_t x = 0; x < sprite->w; x++) {
      const uint8_t *h = colormap[pixel(sprite, x, y)];
      const uint8_t *l = colormap[pixel(sprite, x, y + 1)];
      if (h[3] && l[3])
        printf(BG FG "▄", h[0], h[1], h[2], l[0], l[1], l[2]);
      else if (h[3])
        printf(FG "▀", h[0], h[1], h[2]);
      else if (l[3])
        printf(FG "▄", l[0], l[1], l[2]);
      else
        printf(" ");
      printf("\033[m");
    }
    printf("\n");
  }
  return 0;
}
