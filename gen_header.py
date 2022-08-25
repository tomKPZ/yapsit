#!/usr/bin/env python3

import collections
import concurrent.futures
import io

import bs4
import requests
from matplotlib import image

URL = "https://pokemondb.net/sprites"


page = bs4.BeautifulSoup(requests.get(URL).text, "html.parser")
mons = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    for gen in page.select(".infocard-list"):
        for pokemon in gen.select(".infocard"):
            name = pokemon.text.strip()
            src = pokemon.find("span").get("data-src")
            if src.endswith("/s.png"):
                continue
            future = executor.submit(lambda src: requests.get(src).content, src)
            mons.append(future)
for pokemon in mons:
    sprite = image.imread(io.BytesIO(pokemon.result()))

    n, m, _ = sprite.shape
    inf = float("inf")
    xl, yl, xh, yh = inf, inf, -inf, -inf
    for y in range(n):
        for x in range(m):
            if sprite[y][x][3]:
                xl = min(xl, x)
                yl = min(yl, y)
                xh = max(xh, x)
                yh = max(yh, y)

    colormap = collections.OrderedDict()
    w = xh - xl + 1
    h = yh - yl + 1
    print("{%d,%d,(uint8_t[]){" % (w, h), end="")
    for y in range(yl, yh + 1):
        for x in range(xl, xh + 1):
            r, g, b, a = (round(v * 255) for v in sprite[y][x])
            if a == 0:
                color = 0
            elif (r, g, b) in colormap:
                color = colormap[(r, g, b)]
            else:
                colormap[(r, g, b)] = len(colormap) + 1
                color = colormap[(r, g, b)]
            print("%d," % color, end="")
    print("},{")
    for channel in range(3):
        print("(uint8_t[]){")
        for color in colormap:
            print("0x%02X," % color[channel], end="")
        print("},")
    print("}},")
