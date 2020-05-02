import math

def LAW_OF_COSINE_I1(b, c, alpha, a):
    return math.cosh(a) - math.cosh(b) * math.cosh(c) + math.sinh(b) * math.sinh(c) * math.cos(alpha)

def LAW_OF_COSINE_I2(a, c, beta, b):
    return math.cosh(b) - math.cosh(a) * math.cosh(c) + math.sinh(a) * math.sinh(c) * math.cos(beta)

def LAW_OF_COSINE_I3(a, b, gamma, c):
    return math.cosh(c) - math.cosh(a) * math.cosh(b) + math.sinh(a) * math.sinh(b) * math.cos(gamma)
