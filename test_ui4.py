import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
import sys
import numpy as np
import time

app = QApplication(sys.argv)
plot = pg.PlotWidget()

st = time.time()
for j in range(100):
    plot.clear()

    # Is it faster if we don't clear and just re-assign?
    # the issue with clear() is that creating a lot of lines each time is slow

    # but the current code calls self.clear_canvas() and then self.canvas.plot(...)
print(time.time() - st)
