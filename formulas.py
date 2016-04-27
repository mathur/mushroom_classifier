import math

def sig(z):
    return float(1) / float(1 + math.exp(-z))

def inv_sig(z):
    return sig(z) * (1 - sig(z))

def err(o, t):
    return 0.5 * ((t - o) ** 2)

def inv_err(o, t):
    return (o - t)