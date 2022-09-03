#!/usr/bin/env python3

import collections
import heapq
import itertools
import json

sprites = json.loads(open("data.json").read())


def rle(data):
    counts = []
    runs = []
    for x, g in itertools.groupby(data):
        count = len(list(g))
        while count > 16:
            counts.append(15)
            runs.append(x)
            count -= 16
        counts.append(count - 1)
        runs.append(x)
    return (counts, runs)


def he(data):
    counter = collections.Counter(data)
    heap = [(counter[i], i, i) for i in range(16)]
    heapq.heapify(heap)
    while len(heap) > 1:
        c1, _, v1 = heapq.heappop(heap)
        c2, _, v2 = heapq.heappop(heap)
        heapq.heappush(heap, (c1 + c2, -len(heap), (v1, v2)))
    tree = heap[0][2]

    data2bits = {}
    acc = []
    form = []
    perm = []

    def dfs(node):
        if type(node) == int:
            form.append(1)
            data2bits[node] = acc[::]
            perm.append(node)
            return
        form.append(0)
        l, r = node
        acc.append(0)
        dfs(l)
        acc.pop()
        acc.append(1)
        dfs(r)
        acc.pop()

    dfs(tree)
    bits = sum((data2bits[x] for x in data), start=[])
    return (bits, form, perm)


uncompressed = 0
compressed = 0
for sprite in sprites:
    uncompressed += 4 * len(sprite)
    counts, runs = rle(sprite)
    compressed += len(he(counts)[0]) + len(he(runs)[0]) + 160
print(compressed, uncompressed, compressed / uncompressed)
