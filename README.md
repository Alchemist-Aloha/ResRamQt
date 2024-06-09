**ResRamQt App Introduction**

The ResRamQt is a PyQt6-based graphical user interface (GUI) application that enables users to load and visualize resonance Raman Spectra. The app features simulation and fit of the Raman excitation profiles with respect to displacement of each vibration mode. The output file will include solvent reorganization energy, internal reorganization energy and total reorganization energy based on the displacement values from the best fit. The sample dataset is included as 'abs_exp.dat','deltas.dat', 'freqs.dat', 'inp.txt' and 'profs_exp.dat'. 

**Key Features**

* Load experimental resonance Raman data from a folder
* Visualize the data in a table and plot
* Modify diaplcement values and see the changes in excitation profile instantly
* Fit the data using methods available in the 'lmfit' package
* Save the output data to a folder

**Installation**
  
1. Install Python 3.9-3.12.
3. Clone the repository:
```Python
git clone https://github.com/Alchemist-Aloha/ResRamQt.git
```
3. Install the requirements:
```Python
pip3 install -r requirements.txt
```
4. Run ResRamQt.py
```Python
Python ResRamQt.py
``` 
