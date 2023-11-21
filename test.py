import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

# Create a PyQtGraph application
app = QtGui.QApplication([])
win = pg.GraphicsLayoutWidget()
win.show()

# Create a plot item
plot = win.addPlot()

# Set up empty scatter plot
scatter = pg.ScatterPlotItem()
plot.addItem(scatter)

# Generate random data for demonstration purposes
x_data = np.random.rand(100) * 10
y_data = np.random.rand(100) * 10

# Function to add points one by one
def add_point():
    if len(x_data) > 0:
        x = x_data[0]
        y = y_data[0]
        scatter.addPoints([x], [y])
        x_data = x_data[1:]
        y_data = y_data[1:]
        QtCore.QTimer.singleShot(100, add_point)  # Delay between points

# Call the function to add points one by one
add_point()

# Start the Qt event loop
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
