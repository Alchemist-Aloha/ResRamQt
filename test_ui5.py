import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
import sys
import numpy as np
import time

app = QApplication(sys.argv)
plot = pg.PlotWidget()

lines = []
st = time.time()
for j in range(10):
    for i in range(26):
        if len(lines) <= i:
            pen = pg.mkPen('r')
            line = plot.plot(np.arange(1000), np.random.rand(1000), pen=pen)
            line.setDownsampling(ds=True, auto=True, method="subsample")
            lines.append(line)
        else:
            lines[i].setData(np.arange(1000), np.random.rand(1000))

print("Plotting using state caching: ", time.time() - st)

st2 = time.time()
for j in range(10):
    plot.clear()
    for i in range(26):
        pen = pg.mkPen('r')
        line = plot.plot(np.arange(1000), np.random.rand(1000), pen=pen)
        line.setDownsampling(ds=True, auto=True, method="subsample")

print("Plotting with clear: ", time.time() - st2)
