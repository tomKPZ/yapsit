#!/usr/bin/env python3


def lz77(data):
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
    assert string == "".join(unlz77(lz77(string)))


test("banana")
test("abracadabra")
