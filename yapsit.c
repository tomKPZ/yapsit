#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/auxv.h>
#include <sys/ioctl.h>
#include <unistd.h>

struct Sprite {
  unsigned int w;
  unsigned int h;
  const uint8_t *image;
  const uint8_t *colormap[3];
} const sprites[] = {
#include "pokemon.h"
};

const size_t n_sprites = sizeof(sprites) / sizeof(sprites[0]);

int main() {
  srand(*(unsigned int *)getauxval(AT_RANDOM));

  struct winsize w;
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

  size_t n = 0;
  const struct Sprite *sprite = NULL;
  for (size_t i = 0; i < n_sprites; i++) {
    if (2 * sprites[i].w > w.ws_col || 2 + sprites[i].h > w.ws_row)
      continue;
    if (rand() % (++n) == 0)
      sprite = &sprites[i];
  }
  if (!sprite)
    return 0;

  for (size_t y = 0; y < sprite->h; y++) {
    for (size_t x = 0; x < sprite->w; x++) {
      uint8_t color = sprite->image[y * sprite->w + x];
      if (color == 0) {
        printf("  ");
        continue;
      }
      uint8_t r = sprite->colormap[0][color - 1];
      uint8_t g = sprite->colormap[1][color - 1];
      uint8_t b = sprite->colormap[2][color - 1];
      printf("\033[48;2;%d;%d;%dm  \033[m", r, g, b);
    }
    printf("\n");
  }
  return 0;
}
