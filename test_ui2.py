import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
import sys
import numpy as np
import time

app = QApplication(sys.argv)
plot = pg.PlotWidget()

lines = []
for i in range(26):
    pen = pg.mkPen('r')
    line = plot.plot(np.arange(1000), np.random.rand(1000), pen=pen)
    lines.append(line)

st = time.time()
for j in range(100):
    plot.clear()
    for i in range(26):
        pen = pg.mkPen('r')
        plot.plot(np.arange(1000), np.random.rand(1000), pen=pen)
print(time.time() - st)
