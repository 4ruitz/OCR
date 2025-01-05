#!/usr/bin/env python3

l = ["1", "2", "3", "a", "b", "c"]
m = (*l, "d")
l.append('l')
n = "_".join(x for x in l)
o = "_".join(x for x in m)
print(type(n), n)
print(type(o), o)
print(True if n[12] == o[-1] else False)