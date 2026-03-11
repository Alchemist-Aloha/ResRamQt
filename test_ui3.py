import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
import sys
import numpy as np
import time

app = QApplication(sys.argv)
plot = pg.PlotWidget()

lines = []

st = time.time()
for j in range(100):
    plot.clear()

    # We can use MultiLine instead of PlotDataItem for performance, or keep plots around?
    # In ResRamQt.py, they clear the canvas and re-plot:
    # self.clear_canvas()
    # for i in ...
    #    self.canvas2.plot(...)

print(time.time() - st)
