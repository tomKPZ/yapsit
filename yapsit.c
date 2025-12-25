#include <assert.h>
#include <getopt.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <zstd.h>

#include "constants.h"
#include "types.h"

#define SHINY_NUMERATOR 1
#define SHINY_DENOMINATOR 16
#define DRAW_BUFFER 44294

static uint8_t max_u8(uint8_t a, uint8_t b) { return a > b ? a : b; }
static uint8_t min_u8(uint8_t a, uint8_t b) { return a > b ? b : a; }

static inline bool in_range(size_t x, const Range *range) {
  return x >= range->lo && x <= range->hi;
}

static bool any_variants_in_range(const uint8_t *variants, uint8_t count,
                                  const Range *range) {
  uint8_t max_v = 0;
  for (uint8_t i = 0; i < count; i++)
    max_v = max_u8(max_v, variants[i]);
  return range->lo < max_v;
}

static bool sheet_and_frame_in_range(const Arguments *args,
                                     const SpriteContext *context) {
  for (uint8_t s = 0; s < sprites.groups[context->gid]; s++) {
    uint8_t sheet = context->sheet + s;
    if (in_range(sheet, &args->sheet) &&
        args->frame.lo < sprites.frames[sheet]) {
      return true;
    }
  }
  return false;
}

static size_t sprite_image_size(const Sprite *sprite) {
  return (size_t)sprite->w * sprite->h * sprite->d;
}

static size_t sprite_palette_stride(uint8_t palette_count) {
  return (size_t)palette_count * 16 * 3;
}

static size_t sprite_data_size(const Sprite *sprite, uint8_t palette_count) {
  return sprite_image_size(sprite) +
         sprite->d * sprite_palette_stride(palette_count);
}

static bool choose_sprite(const Arguments *args, SpriteContext *out) {
  memset(out, 0, sizeof(SpriteContext));
  size_t n = 0;
  SpriteContext context;
  memset(&context, 0, sizeof(SpriteContext));
  context.sprite = sprites.images;
  context.variants = sprites.variants;
  context.index = 0;
  for (context.gid = 0; context.gid < GROUP_COUNT; context.gid++) {
    const bool sf_in_range = sheet_and_frame_in_range(args, &context);
    for (uint16_t id = 0; id < sprites.limits[context.gid]; id++) {
      if (sf_in_range && in_range(id, &args->id) &&
          in_range(context.sprite->w - 1, &args->width) &&
          in_range(context.sprite->h - 1, &args->height) &&
          any_variants_in_range(context.variants, sprites.groups[context.gid],
                                &args->variants) &&
          rand() % ++n == 0) {
        memcpy(out, &context, sizeof(SpriteContext));
      }
      context.variants += sprites.groups[context.gid];
      context.sprite++;
      context.index++;
    }
    context.sheet += sprites.groups[context.gid];
  }
  if (!out->sprite)
    return false;

  n = 0;
  bool success = false;
  uint8_t z = 0;
  for (size_t g = 0; g < sprites.groups[out->gid]; g++) {
    for (size_t v = 0; v < out->variants[g]; v++) {
      for (uint8_t f = 0; f < sprites.frames[out->sheet + g]; f++, z++) {
        if (in_range(out->sheet + g, &args->sheet) &&
            in_range(v, &args->variants) && in_range(f, &args->frame) &&
            rand() % ++n == 0) {
          out->z = z;
          success = true;
        }
      }
    }
  }
  assert(z == out->sprite->d);
  return success;
}

static void load_palette(uint8_t palette[16][3], const uint8_t *src) {
  memcpy(palette, src, 16 * 3);
}

static bool decompress_group(uint8_t *buffer, size_t buffer_size,
                             uint32_t index, ZSTD_DCtx *dctx,
                             const ZSTD_DDict *ddict) {
  const uint32_t offset = sprites.offsets[index];
  const uint32_t size = sprites.sizes[index];
  const uint8_t *src = sprites.compressed + offset;
  size_t result =
      ZSTD_decompress_usingDDict(dctx, buffer, buffer_size, src, size, ddict);
  if (ZSTD_isError(result))
    return false;
  return result == buffer_size;
}

static char *out;
static uint8_t pfg = 0;
static uint8_t pbg = 0;

static void output_u8(uint8_t i) {
  out[2] = i % 10 + '0';
  i /= 10;
  out[1] = i % 10 + '0';
  i /= 10;
  out[0] = i + '0';
  out += 3;
}

static void output_color(const uint8_t color[3]) {
  for (size_t i = 0; i < 3; i++) {
    *out++ = ';';
    output_u8(color[i]);
  }
}

static void output_str(const char *str) {
  size_t len = strlen(str);
  memcpy(out, str, len);
  out += len;
}

static void fg(uint8_t i, const uint8_t palette[16][3]) {
  if (pfg == i)
    return;
  pfg = i;

  output_str("\033[38;2");
  output_color(palette[i]);
  *out++ = 'm';
}

static void bg(uint8_t i, const uint8_t palette[16][3]) {
  if (pbg == i)
    return;
  pbg = i;

  output_str("\033[48;2");
  output_color(palette[i]);
  *out++ = 'm';
}

static void reset() {
  if (pfg || pbg)
    output_str("\033[m");
  pfg = 0;
  pbg = 0;
}

static void draw(uint8_t w, uint8_t h, const uint8_t *image,
                 const uint8_t palette[16][3]) {
  // Trim the image.
  uint8_t x_l = 255;
  uint8_t y_l = 255;
  uint8_t x_h = 0;
  uint8_t y_h = 0;
  for (size_t y = 0; y < h; y += 2) {
    for (size_t x = 0; x < w; x++) {
      if (image[y * w + x]) {
        x_l = min_u8(x_l, x);
        x_h = max_u8(x_h, x);
        y_l = min_u8(y_l, y);
        y_h = max_u8(y_h, y);
      }
    }
  }

  char buf[DRAW_BUFFER];
  out = buf;
  for (size_t y = y_l; y <= y_h; y += 2) {
    for (size_t x = x_l; x <= x_h; x++) {
      uint8_t u = image[y * w + x];
      uint8_t l = 0;
      if (y + 1 < h)
        l = image[(y + 1) * w + x];
      if (u && l) {
        if ((pbg == u) + (pfg == l) > (pfg == u) + (pbg == l)) {
          fg(l, palette);
          bg(u, palette);
          output_str("\u2584");
        } else {
          fg(u, palette);
          bg(l, palette);
          output_str("\u2580");
        }
      } else if (u) {
        if (pbg)
          reset();
        fg(u, palette);
        output_str("\u2580");
      } else if (l) {
        if (pbg)
          reset();
        fg(l, palette);
        output_str("\u2584");
      } else {
        reset();
        *out++ = ' ';
      }
    }
    reset();
    *out++ = '\n';
  }
  *out++ = 0;
  assert(out - buf <= DRAW_BUFFER);
  puts(buf);
}

static bool parse_u32(const char *s, uint32_t *i) {
  *i = 0;
  for (; *s; s++) {
    if (*s > '9' || *s < '0')
      return false;
    uint64_t j = *i * 10 + *s - '0';
    if (j > 0xffffffff)
      return false;
    *i = j;
  }
  return true;
}

static bool parse_range_endpoint(const char *s, uint16_t *i) {
  uint32_t j;
  if (!parse_u32(s, &j))
    return false;
  *i = j - 1;
  return j != 0 && j <= 0xffff;
}

static bool parse_range(char *arg, Range *range) {
  char *upper = strchr(arg, '-');
  if (upper)
    *upper++ = '\0';
  else
    upper = arg;
  return parse_range_endpoint(arg, &range->lo) &&
         parse_range_endpoint(upper, &range->hi);
}

static void init_range(Range *range) {
  range->lo = 0;
  range->hi = 0xffff;
}

static const char usage[] =
    "Usage: yapsit [OPTION...]\n"
    "Show a random pokemon sprite.\n"
    "\n"
    "  -i, --id=ID[-ID]           Filter by ID\n"
    "  -s, --sheet=SID[-SID]      Filter by sprite sheet\n"
    "  -v, --variants=VID[-VID]   Filter by variant ID\n"
    "  -f, --frame=FID[-FID]      Filter by frame number\n"
    "  -W, --width=W[-MAX]        Filter by sprite width\n"
    "  -H, --height=H[-MAX]       Filter by sprite height\n"
    "  -n, --numerator=N          Shiny chance numerator\n"
    "  -d, --denominator=D        Shiny chance denominator\n"
    "  -t, --test                 Output all sprites\n"
    "  -h, --help                 Give this help list\n";

static struct option options[] = {
    {"id", required_argument, 0, 'i'},
    {"sheet", required_argument, 0, 's'},
    {"variants", required_argument, 0, 'v'},
    {"frame", required_argument, 0, 'f'},
    {"width", required_argument, 0, 'W'},
    {"height", required_argument, 0, 'H'},
    {"numerator", required_argument, 0, 'n'},
    {"denominator", required_argument, 0, 'd'},
    {"test", no_argument, 0, 't'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0},
};

static void init_args(Arguments *args, int argc, char *argv[]) {
  args->numerator = SHINY_NUMERATOR;
  args->denominator = SHINY_DENOMINATOR;
  init_range(&args->id);
  init_range(&args->sheet);
  init_range(&args->variants);
  init_range(&args->frame);
  init_range(&args->width);
  init_range(&args->height);
  args->test = false;
  while (true) {
    switch (getopt_long(argc, argv, "i:s:v:f:W:H:n:d:th", options, NULL)) {
    case -1:
      return;
    case 'i':
      if (!parse_range(optarg, &args->id))
        exit(EXIT_FAILURE);
      break;
    case 's':
      if (!parse_range(optarg, &args->sheet))
        exit(EXIT_FAILURE);
      break;
    case 'v':
      if (!parse_range(optarg, &args->variants))
        exit(EXIT_FAILURE);
      break;
    case 'f':
      if (!parse_range(optarg, &args->frame))
        exit(EXIT_FAILURE);
      break;
    case 'W':
      if (!parse_range(optarg, &args->width))
        exit(EXIT_FAILURE);
      break;
    case 'H':
      if (!parse_range(optarg, &args->height))
        exit(EXIT_FAILURE);
      break;
    case 'n':
      if (!parse_u32(optarg, &args->numerator))
        exit(EXIT_FAILURE);
      break;
    case 'd':
      if (!parse_u32(optarg, &args->denominator))
        exit(EXIT_FAILURE);
      break;
    case 't':
      args->test = true;
      break;
    case 'h':
      puts(usage);
      exit(EXIT_SUCCESS);
    case '?':
    default:
      fputs(usage, stderr);
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char *argv[]) {
  Arguments args;
  init_args(&args, argc, argv);
  ZSTD_DCtx *dctx = ZSTD_createDCtx();
  ZSTD_DDict *ddict = ZSTD_createDDict(sprites.dictionary, DICT_LEN);
  if (!dctx || !ddict)
    return EXIT_FAILURE;
  uint8_t buffer[DECOMPRESS_BUFFER];
  int result = EXIT_SUCCESS;

  if (args.test) {
    const Sprite *sprite = sprites.images;
    uint32_t index = 0;
    for (size_t gid = 0; gid < GROUP_COUNT; gid++) {
      const uint8_t palette_count = sprites.palette_counts[gid];
      for (size_t id = 0; id < sprites.limits[gid]; id++, sprite++) {
        uint8_t w = sprite->w, h = sprite->h, d = sprite->d;
        size_t image_size = sprite_image_size(sprite);
        size_t palette_stride = sprite_palette_stride(palette_count);
        size_t data_size = sprite_data_size(sprite, palette_count);
        assert(data_size <= DECOMPRESS_BUFFER);
        if (!decompress_group(buffer, data_size, index, dctx, ddict)) {
          result = EXIT_FAILURE;
          goto cleanup;
        }
        uint8_t *image = buffer;
        uint8_t *palette_base = buffer + image_size;
        for (size_t z = 0; z < d; z++) {
          uint8_t *frame = image + w * h * z;
          uint8_t *sprite_palette = palette_base + z * palette_stride;
          for (int j = 0; j < palette_count; j++) {
            uint8_t palette[16][3];
            load_palette(palette, sprite_palette + j * 16 * 3);
            draw(w, h, frame, palette);
          }
        }
        index++;
      }
    }
  } else {
    srand(time(NULL) ^ getpid());

    SpriteContext s;
    if (!choose_sprite(&args, &s)) {
      result = EXIT_FAILURE;
      goto cleanup;
    }
    uint8_t w = s.sprite->w, h = s.sprite->h;
    size_t image_size = sprite_image_size(s.sprite);
    size_t palette_stride = sprite_palette_stride(sprites.palette_counts[s.gid]);
    size_t data_size = sprite_data_size(s.sprite, sprites.palette_counts[s.gid]);
    assert(data_size <= DECOMPRESS_BUFFER);
    if (!decompress_group(buffer, data_size, s.index, dctx, ddict)) {
      result = EXIT_FAILURE;
      goto cleanup;
    }
    uint8_t *image = buffer;
    uint8_t *palette_base = buffer + image_size;

    uint8_t palette[16][3];
    bool shiny = rand() % args.denominator < args.numerator;
    uint8_t palette_index =
        min_u8(shiny, sprites.palette_counts[s.gid] - 1);
    uint8_t *sprite_palette = palette_base + s.z * palette_stride;
    load_palette(palette, sprite_palette + palette_index * 16 * 3);
    draw(w, h, image + w * h * s.z, palette);
  }

cleanup:
  ZSTD_freeDDict(ddict);
  ZSTD_freeDCtx(dctx);
  return result;
}
