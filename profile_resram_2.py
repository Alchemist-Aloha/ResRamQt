import cProfile
import pstats
import io
import sys
from ResRamQt import load_input, cross_sections
import numpy as np

def test_speed():
    obj = load_input()
    import time
    start = time.time()
    for _ in range(10):
        cross_sections(obj)
    print("Time taken:", time.time() - start)

if __name__ == '__main__':
    test_speed()
