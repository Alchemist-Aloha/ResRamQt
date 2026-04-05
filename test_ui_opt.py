import sys
from PyQt6.QtWidgets import QApplication
from ResRamQt import SpectrumApp
import time

app = QApplication(sys.argv)
app.setQuitOnLastWindowClosed(False)
window = SpectrumApp()

st = time.time()
for _ in range(10):
    window.plot_data()
print("Plotting using state caching: ", time.time() - st)

app.quit()
