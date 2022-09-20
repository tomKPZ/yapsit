#!/usr/bin/env python3

import random


def lz77(data):
    n = len(data)
    dp = [0] * n
    for i in reversed(range(n)):
        size, lst = dp[i + 1] if i + 1 < n else (0, None)
        ans = (size + 1, ((0, 0, data[i]), lst))
        for j in range(i):
            for k in range(j, min(i, n - i + j)):
                if data[k] != data[i + k - j]:
                    break
                runlen = k - j + 1
                delta = i - j
                index = i + runlen + 1
                lstlen, lst = dp[index] if index < n else (0, None)
                nxt = data[i + runlen] if i + runlen < n else None
                ans = min(ans, (1 + lstlen, ((delta, runlen, nxt), lst)))
        dp[i] = ans

    node = dp[0][1]
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
    assert string == "".join(unlz77(lz77(string)))


for _ in range(10):
    test("".join(random.choice("ab") for _ in range(500)))
