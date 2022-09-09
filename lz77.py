#!/usr/bin/env python3


def lz77(data):
    ans = []
    i = 0
    n = len(data)
    while i < n:
        run = (0, 0)
        for j in range(i):
            for k in range(j, min(i, n - i + j)):
                if data[k] != data[i + k - j]:
                    break
                run = max(run, (k - j + 1, j - i))
        if not run[0]:
            ans.append((0, 0, data[i]))
            i += 1
            continue

        runlen, delta = run
        nxt = data[i + runlen] if i + runlen < n else None
        ans.append((-delta, runlen, nxt))
        i += 1 + runlen
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


test("banana")
test("abracadabra")
