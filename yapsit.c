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
    if (sprites[i].w > w.ws_col || (sprites[i].h + 1) / 2 + 2 > w.ws_row)
      continue;
    if (rand() % (++n) == 0)
      sprite = &sprites[i];
  }
  if (!sprite)
    return 0;

  for (size_t y = 0; y < sprite->h; y += 2) {
    for (size_t x = 0; x < sprite->w; x++) {
      uint8_t h = sprite->image[y * sprite->w + x];
      uint8_t l = 0;
      if (y + 1 < sprite->h)
        l = sprite->image[(y + 1) * sprite->w + x];
      uint8_t rh, gh, bh, rl, gl, bl;
      if (h) {
        rh = sprite->colormap[0][h - 1];
        gh = sprite->colormap[1][h - 1];
        bh = sprite->colormap[2][h - 1];
      }
      if (l) {
        rl = sprite->colormap[0][l - 1];
        gl = sprite->colormap[1][l - 1];
        bl = sprite->colormap[2][l - 1];
      }
      if (h && l)
        printf(BG FG "▄", rh, gh, bh, rl, gl, bl);
      else if (h)
        printf(FG "▀", rh, gh, bh);
      else if (l)
        printf(FG "▄", rl, gl, bl);
      else
        printf(" ");
      printf("\033[m");
    }
    printf("\n");
  }
  return 0;
}
