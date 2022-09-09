#!/usr/bin/env python3

import functools
import random


def lz77_dp(data):
    n = len(data)
    inf = float("inf")

    @functools.cache
    def aux(i):
        if i >= n:
            return (0, None)
        prefix = data[:i]
        suffix = data[i:]
        runs = {}
        for j in range(i):
            for k in range(j, min(i, len(suffix) + j)):
                if prefix[k] != suffix[k - j]:
                    break
                runs[k - j + 1] = i - j
        if not runs:
            size, lst = aux(i + 1)
            return (size + 1, ((0, 0, data[i]), lst))
        ans = (inf,)
        for runlen, delta in runs.items():
            lstlen, lst = aux(i + runlen + 1)
            nxt = data[i + runlen] if i + runlen < n else None
            ans = min(ans, (1 + lstlen, ((delta, runlen, nxt), lst)))
        return ans

    node = aux(0)[1]
    ans = []
    while node is not None:
        first, rest = node
        ans.append(first)
        node = rest
    return ans


def lz77_greedy(data):
    n = len(data)

    def aux(i):
        if i >= n:
            return (0, None)
        prefix = data[:i]
        suffix = data[i:]
        run = (0, 0)
        for j in range(i):
            for k in range(j, min(i, len(suffix) + j)):
                if prefix[k] != suffix[k - j]:
                    break
                run = max(run, (k - j + 1, i - j))
        if not run[0]:
            size, lst = aux(i + 1)
            return (size + 1, ((0, 0, data[i]), lst))

        runlen, delta = run
        lstlen, lst = aux(i + runlen + 1)
        nxt = data[i + runlen] if i + runlen < n else None
        return (1 + lstlen, ((delta, runlen, nxt), lst))

    node = aux(0)[1]
    ans = []
    while node is not None:
        first, rest = node
        ans.append(first)
        node = rest
    return ans


def unlz77(data):
    ans = []
    for dist, size, nxt in data:
        if dist == 0 and size == 0:
            ans.append(nxt)
            continue
        start = len(ans) - dist
        ans.extend(ans[start : start + size])
        if nxt is not None:
            ans.append(nxt)
    return ans


def test(string):
    assert string == "".join(unlz77(lz77_dp(string)))
    assert string == "".join(unlz77(lz77_greedy(string)))
    l1, l2 = (len(lz77_dp(string)), len(lz77_greedy(string)))
    if l1 != l2:
        print(l1, l2)


while True:
    string = "".join(random.choice("ab") for _ in range(500))
    test(string)
