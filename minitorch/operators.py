"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Return the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Check if x is less than y."""
    return float(x < y)


def eq(x: float, y: float) -> float:
    """Check if x is equal to y."""
    return float(x == y)


def max(x: float, y: float) -> float:
    """Return the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check whether two numbers are close within tolerance."""
    return float(abs(x - y) < 1e-2)


def sigmoid(x: float) -> float:
    """Compute sigmoid."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Apply rectified linear unit."""
    return max(0.0, x)


def log(x: float) -> float:
    """Compute natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential function."""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Backprop for log: d/dx log(x) * upstream."""
    return y / x


def inv(x: float) -> float:
    """Compute multiplicative inverse."""
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    """Backprop for inverse: d/dx (1/x) * upstream."""
    return -y / x**2


def relu_back(x: float, y: float) -> float:
    """Backprop for relu: pass upstream only when x > 0."""
    return float(y * (x > 0))


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Apply a unary function to each element."""
    return [f(x) for x in ls]


def zipWith(
    f: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    """Combine two iterables elementwise with a binary function."""
    return [f(x, y) for x, y in zip(ls1, ls2)]


def reduce(
    f: Callable[[float, float], float], ls: Iterable[float], initial: float
) -> float:
    """Fold an iterable from left to right."""
    result = initial
    for x in ls:
        result = f(result, x)
    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Return a list with all elements negated."""
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add two iterables elementwise."""
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Return the sum of all elements."""
    return reduce(add, ls, 0.0)


def prod(ls: Iterable[float]) -> float:
    """Return the product of all elements."""
    return reduce(mul, ls, 1.0)
