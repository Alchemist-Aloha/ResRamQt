import numpy as np
import time
from ResRamQt import load_input, g, A, cross_sections

def run_test():
    obj = load_input()

    start = time.time()
    for _ in range(10):
        cross_sections(obj)
    print("Old time:", time.time() - start)

if __name__ == '__main__':
    run_test()
