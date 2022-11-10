#include <stddef.h>
#include <stdint.h>
#include <string.h>

// TODO : multiple representations of(dz, dy, dx) for the same delta.
static int nbits(const int output[5], const uint8_t data2bits[5][256]) {
  int sum = 0;
  for (size_t i = 0; i < 5; i++) {
    if (output[i] >= 0)
      sum += data2bits[i][output[i]];
  }
  return sum;
}

static int min(int a, int b) { return a < b ? a : b; }

void lz77(const uint8_t width, const uint8_t height, const uint8_t depth,
          const uint8_t data[], const uint8_t data2bits[5][256], int dp[][7]) {
  int n = depth * width * height;
  for (int i = n; i-- > 0;) {
    int ans[7] = {
        dp[i + 1][0],
        i + 1,
        i < width * height ? -1 : 0,
        i < width ? -1 : 128,
        i == 0 ? -1 : 128,
        -1,
        data[i],
    };
    ans[0] += nbits(&ans[2], data2bits);
    for (int j = 0; j < i; j++) {
      int upper = min(n - i + j, j + 256);
      for (int k = j; k < upper; k++) {
        if (data[k] != data[k + i - j])
          break;
        int y1 = j / width;
        int x1 = j % width;
        int z1 = y1 / height;
        y1 %= height;
        int y2 = i / width;
        int x2 = i % width;
        int z2 = y2 / height;
        y2 %= height;
        int runlen = k - j + 1;
        int index = i + runlen + 1;
        int node[7] = {
            dp[index][0],
            index,
            z2 ? z2 - z1 : -1,
            z2 || y2 ? y2 - y1 + 128 : -1,
            z2 || y2 || x2 ? x2 - x1 + 128 : -1,
            i != j ? runlen - 1 : -1,
            i + runlen < n ? data[i + runlen] : -1,
        };
        node[0] += nbits(&node[2], data2bits);
        if (node[0] <= ans[0])
          memcpy(ans, node, 7 * sizeof(int));
      }
    }
    memcpy(dp[i], ans, 7 * sizeof(int));
  }
}
