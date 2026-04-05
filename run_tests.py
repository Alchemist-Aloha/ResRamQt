from ResRamQt import cross_sections, load_input, A, g
import numpy as np

def run_tests():
    obj = load_input()

    # Just run a single simulation step to make sure it doesn't crash
    abs_cross, fl_cross, raman_cross, boltz_states, boltz_coef = cross_sections(obj)

    print(f"Computed Abs shape: {abs_cross.shape}")
    print(f"Computed Fl shape: {fl_cross.shape}")
    print(f"Computed Raman shape: {raman_cross.shape}")

    print("Tests passed successfully.")

if __name__ == '__main__':
    run_tests()
