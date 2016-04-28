import math

def sig(x):
    return float(1) / float(1 + math.exp(-x))

def inv_sig(x):
    return sig(x) * (1 - sig(x))

def err(o, t):
    return 0.5 * ((t - o) ** 2)

def inv_err(o, t):
    return (o - t)