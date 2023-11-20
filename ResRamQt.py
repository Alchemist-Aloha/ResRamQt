from PyQt5.QtCore import Qt, QThreadPool, pyqtSlot, QRunnable, pyqtSignal, QTimer, QObject
from PyQt5.QtWidgets import QMessageBox, QLabel, QFileDialog, QCheckBox, QTextBrowser, QHeaderView, QApplication, QMainWindow, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtGui
import sys
from datetime import datetime
import os
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import lmfit

class load_input():
    def __init__(self, dir=None):
        if dir is None:
            # Set default directory as empty if none provided
            self.dir = ''
        else:
            self.dir = dir
        # Ground state normal mode frequencies cm^-1
        self.wg = np.asarray(np.loadtxt(self.dir+'freqs.dat'))
        # Excited state normal mode frequencies cm^-1
        self.we = np.asarray(np.loadtxt(self.dir+'freqs.dat'))
        # Dimensionless displacements
        self.delta = np.asarray(np.loadtxt(self.dir+'deltas.dat'))
        # divide color map to number of freqs
        self.colors = plt.cm.hsv(np.linspace(0, 1, len(self.wg)))
        self.cmap = ListedColormap(self.colors)
        self.S = (self.delta**2)/2  # calculate in cross_sections()
        try:
            self.abs_exp = np.loadtxt(self.dir+'abs_exp.dat')
        except:
            print('No experimental absorption spectrum found in directory/')

        try:
            self.profs_exp = np.loadtxt(self.dir+'profs_exp.dat')
        except:
            print('No experimental Raman cross section found in directory/')
        self.inp_txt()

    # Function to read input file
    def inp_txt(self):
        try:
            with open(self.dir+'inp.txt', 'r') as i:  # loading inp.txt
                print(f'load from {self.dir}inp.txt')
                self.inp = i.readlines()
            i.close()
        except:
            with open(self.dir+'inp_new.txt', 'r') as i:  # loading inp_new.txt
                print(f'load from {self.dir}inp.txt')
        
                self.inp = i.readlines()
            i.close()
                # Process each line in inp.txt
        j = 0
        for l in self.inp:
            l = l.partition('#')[0] # Remove comments
            l = l.rstrip()  # Remove trailing whitespaces
            self.inp[j] = l
            j += 1
        # Constants and parameters from inp.txt
        self.hbar = 5.3088  # plancks constant cm^-1*ps
        self.T = float(self.inp[13])  # Temperature K
        self.kbT = 0.695*self.T  # kbT energy (cm^-1/K)*cm^-1=cm^-1
        self.cutoff = self.kbT*0.1  # cutoff for boltzmann dist in wavenumbers
        if self.T > 10.0:
            self.beta = 1/self.kbT  # beta in cm
            # array of average thermal occupation numbers for each mode
            self.eta = 1/(np.exp(self.wg/self.kbT)-1)
        elif self.T < 10.0:
            self.beta = 1/self.kbT
            # beta = float("inf")
            self.eta = np.zeros(len(self.wg))

        # Homogeneous broadening parameter cm^-1
        self.gamma = float(self.inp[0])
        # Static inhomogenous broadening parameter cm^-1
        self.theta = float(self.inp[1])
        self.E0 = float(self.inp[2])  # E0 cm^-1

        ## Brownian Oscillator parameters ##
        self.k = float(self.inp[3])  # kappa parameter
        self.D = self.gamma*(1+0.85*self.k+0.88*self.k**2) / \
            (2.355+1.76*self.k)  # D parameter
        self.L = self.k*self.D  # LAMBDA parameter

        '''can be moved to save()'''
        self.s_reorg = self.beta * \
            (self.L/self.k)**2/2  # reorganization energy cm^-1
        # internal reorganization energy
        self.w_reorg = 0.5*np.sum((self.delta)**2*self.wg)
        self.reorg = self.w_reorg + self.s_reorg  # Total reorganization energy

        ## Time and energy range stuff ##
        self.ts = float(self.inp[4])  # Time step (ps)
        self.ntime = float(self.inp[5])  # 175 # ntime steps
        self.UB_time = self.ntime*self.ts  # Upper bound in time range
        self.t = np.linspace(0, self.UB_time, int(
            self.ntime))  # time range in ps
        # How far plus and minus E0 you want
        self.EL_reach = float(self.inp[6])
        # range for spectra cm^-1
        self.EL = np.linspace(self.E0-self.EL_reach,
                            self.E0+self.EL_reach, 1000)
        # static inhomogeneous convolution range
        self.E0_range = np.linspace(-self.EL_reach *
                                    0.5, self.EL_reach*0.5, 501)

        self.th = np.array(self.t/self.hbar)  # t/hbar

        self.ntime_rot = self.ntime/np.sqrt(2)
        self.ts_rot = self.ts/np.sqrt(2)
        self.UB_time_rot = self.ntime_rot*self.ts_rot
        self.tp = np.linspace(0, self.UB_time_rot, int(self.ntime_rot))
        self.tm = None
        self.tm = np.append(-np.flip(self.tp[1:], axis=0), self.tp)
        # Excitation axis after convolution with inhomogeneous distribution
        self.convEL = np.linspace(self.E0-self.EL_reach*0.5, self.E0+self.EL_reach*0.5,
                                (max(len(self.E0_range), len(self.EL))-min(len(self.E0_range), len(self.EL))+1))

        self.M = float(self.inp[7])  # Transition dipole length angstroms
        self.n = float(self.inp[8])  # Refractive index

        # Raman pump wavelengths to compute spectra at
        self.rpumps = np.asarray(np.loadtxt(self.dir+'rpumps.dat'))
        self.rp = np.zeros_like(self.rpumps)
        for rps in range(len(self.rpumps)):
            # rp = min(range(len(convEL)),key=lambda j:abs(convEL[j]-rpumps[i]))
            min_diff = float('inf')

            for j in range(len(self.convEL)):
                diff = np.absolute(self.convEL[j] - self.rpumps[rps])
                if diff < min_diff:
                    min_diff = diff
                    self.rp[rps] = j
        self.rp = self.rp.astype(int)
        self.raman_maxcalc = float(self.inp[10])
        self.rshift = np.arange(float(self.inp[9]), float(self.inp[10]), float(
            self.inp[11]))  # range and step size of Raman spectrum
        self.res = float(self.inp[12])  # Peak width in Raman spectra

        # Determine order from Boltzmann distribution of possible initial states #
        # desired boltzmann coefficient for cutoff
        self.convergence = float(self.inp[14])
        self.boltz_toggle = int(self.inp[15])

        if self.boltz_toggle == 1:
            self.boltz_state, self.boltz_coef, self.dos_energy = self.boltz_states()
            if self.T == 0.0:
                self.state = 0
            else:
                self.state = min(range(len(self.boltz_coef)),
                                key=lambda j: abs(self.boltz_coef[j]-self.convergence))

            if self.state == 0:
                self.order = 1
            else:
                self.order = max(max(self.boltz_state[:self.state])) + 1
        if self.boltz_toggle == 0:
            self.boltz_state, self.boltz_coef, self.dos_energy = [0, 0, 0]
            self.order = 1

        self.a = np.arange(self.order)
        self.b = self.a
        self.Q = np.identity(len(self.wg), dtype=int)

        # wq = None
        # wq = np.append(wg,wg)
        
        ## Prefactors for absorption and Raman cross-sections ##
        if self.order == 1:
            # (0.3/pi) puts it in differential cross section
            self.preR = 2.08e-20*(self.ts**2)
        elif self.order > 1:
            self.preR = 2.08e-20*(self.ts_rot**2)

        self.preA = ((5.744e-3)/self.n)*self.ts
        self.preF = self.preA*self.n**2
        
    def boltz_states(self):
        wg = self.wg.astype(int)
        cutoff = range(int(self.cutoff))
        dos = range(len(self.cutoff))
        states = []
        dos_energy = []

        def count_combs(left, i, comb, add):
            if add:
                comb.append(add)
            if left == 0 or (i+1) == len(wg):
                if (i+1) == len(wg) and left > 0:
                    if left % wg[i]:  # can't get the exact score with this kind of wg
                        return 0         # so give up on this recursive branch
                    comb.append((left/wg[i], wg[i]))  # fix the amount here
                    i += 1
                while i < len(wg):
                    comb.append((0, wg[i]))
                    i += 1
                states.append([x[0] for x in comb])
                return 1
            cur = wg[i]
            return sum(count_combs(left-x*cur, i+1, comb[:], (x, cur)) for x in range(0, int(left/cur)+1))

        boltz_dist = []  # np.zeros(len(dos))
        for i in range(len(cutoff)):
            dos[i] = count_combs(self.cutoff[i], 0, [], None)
            if dos[i] > 0.0:
                boltz_dist.append([np.exp(-cutoff[i]*self.beta)])
                dos_energy.append(cutoff[i])

        norm = np.sum(boltz_dist)

        np.reshape(states, -1, len(cutoff))

        return states, boltz_dist/norm, dos_energy
        



def g(t,obj):
    # Calculate parameters D and L based on obj attributes
    obj.D = obj.gamma*(1+0.85*obj.k+0.88*obj.k**2)/(2.355+1.76*obj.k)  # D parameter
    obj.L = obj.k*obj.D  # LAMBDA parameter
     # Calculate the function g using the calculated parameters
    g = ((obj.D/obj.L)**2)*(obj.L*t-1+np.exp(-obj.L*t))+1j*((obj.beta*obj.D**2)/(2*obj.L))*(1-np.exp(-obj.L*t))
    # g = p.gamma*np.abs(t)#
    return g



def A(t,obj):
    # K=np.zeros((len(p.wg),len(t)),dtype=complex)
    # Initialize K matrix based on the type of t provided
    if type(t) == np.ndarray:
        K = np.zeros((len(obj.wg), len(obj.th)), dtype=complex)
    else:
        K = np.zeros((len(obj.wg), 1), dtype=complex)
    # Calculate the K matrix
    for l in np.arange(len(obj.wg)):
        K[l, :] = (1+obj.eta[l])*obj.S[l]*(1-np.exp(-1j*obj.wg[l]*t)) + \
            obj.eta[l]*obj.S[l]*(1-np.exp(1j*obj.wg[l]*t))
    # Calculate the function A based on the K matrix
    A = obj.M**2*np.exp(-np.sum(K, axis=0))
    return A



def R(t1, t2,obj):
    # Initialize Ra and R arrays for calculations
    Ra = np.zeros((len(obj.a), len(obj.wg), len(obj.wg), len(obj.EL)), dtype=complex)
    R = np.zeros((len(obj.wg), len(obj.wg), len(obj.EL)), dtype=complex)
    # for l in np.arange(len(p.wg)):
    # 	for q in p.Q:
    for idxq, q in enumerate(obj.Q, start=0):
        for idxl, l in enumerate(q, start=0):

            wg = obj.wg[idxl]
            S = obj.S[idxl]
            eta = obj.eta[idxl]
            if l == 0:
                for idxa, a in enumerate(obj.a, start=0):
                    Ra[idxa, idxq, idxl, :] = ((1./factorial(a))**2)*((eta*(1+eta))**a)*S**(2*a)*(
                        ((1-np.exp(-1j*wg*t1))*np.conj((1-np.exp(-1j*wg*t1))))*((1-np.exp(-1j*wg*t1))*np.conj((1-np.exp(-1j*wg*t1)))))**a
                R[idxq, idxl, :] = np.sum(Ra[:, idxq, idxl, :], axis=0)
            elif l > 0:
                for idxa, a in enumerate(obj.a[l:], start=0):
                    Ra[idxa, idxq, idxl, :] = ((1./(factorial(a)*factorial(a-l))))*(((1+eta)*S*(1-np.exp(-1j*wg*t1))*(
                        1-np.exp(1j*wg*t2)))**a)*(eta*S*(1-np.exp(1j*wg*t1))*(1-np.exp(-1j*wg*t2)))**(a-l)
                R[idxq, idxl, :] = np.sum(Ra[:, idxq, idxl, :], axis=0)
            elif l < 0:
                for idxa, a in enumerate(obj.b[-l:], start=0):
                    Ra[idxa, idxq, idxl, :] = ((1./(factorial(a)*factorial(a+l))))*(((1+eta)*S*(1-np.exp(-1j*wg*t1))*(
                        1-np.exp(1j*wg*t2)))**(a+l))*(eta*S*(1-np.exp(1j*wg*t1))*(1-np.exp(-1j*wg*t2)))**(a)
                R[idxq, idxl, :] = np.sum(Ra[:, idxq, idxl, :], axis=0)
    return np.prod(R, axis=1)


def cross_sections(obj):
    sqrt2 = np.sqrt(2)
    obj.S = (obj.delta**2)/2  # calculate in cross_sections()
    obj.EL = np.linspace(obj.E0-obj.EL_reach, obj.E0+obj.EL_reach, 1000)  # range for spectra cm^-1
    q_r = np.ones((len(obj.wg), len(obj.wg), len(obj.th)), dtype=complex)
    K_r = np.zeros((len(obj.wg), len(obj.EL), len(obj.th)), dtype=complex)
    # elif p.order > 1:
    # 	K_r = np.zeros((len(p.tm),len(p.tp),len(p.wg),len(p.EL)),dtype=complex)
    integ_r1 = np.zeros((len(obj.tm), len(obj.EL)), dtype=complex)
    integ_r = np.zeros((len(obj.wg), len(obj.EL)), dtype=complex)
    raman_cross = np.zeros((len(obj.wg), len(obj.convEL)), dtype=complex)

    if obj.theta == 0.0:
        H = 1.  # np.ones(len(p.E0_range))
    else:
        H = (1/(obj.theta*np.sqrt(2*np.pi)))*np.exp(-((obj.E0_range)**2)/(2*obj.theta**2))

    thth, ELEL = np.meshgrid(obj.th, obj.EL, sparse=True)

    K_a = np.exp(1j*(ELEL-(obj.E0))*thth-g(thth,obj))*A(thth,obj)
    K_f = np.exp(1j*(ELEL-(obj.E0))*thth-np.conj(g(thth,obj)))*np.conj(A(thth,obj))

    ## If the order desired is 1 use the simple first order approximation ##
    if obj.order == 1:
        for idxq, q in enumerate(obj.Q, start=0):
            for idxl, l in enumerate(q, start=0):
                if q[idxl] > 0:
                    q_r[idxq, idxl, :] = (1./factorial(q[idxl]))**(0.5)*(((1+obj.eta[idxl])**(
                        0.5)*obj.delta[idxl])/sqrt2)**(q[idxl])*(1-np.exp(-1j*obj.wg[idxl]*thth))**(q[idxl])
                elif q[idxl] < 0:
                    q_r[idxq, idxl, :] = (1./factorial(np.abs(q[idxl])))**(0.5)*(((obj.eta[l])**(
                        0.5)*obj.delta[l])/sqrt2)**(-q[idxl])*(1-np.exp(1j*obj.wg[idxl]*thth))**(-q[idxl])
            K_r[idxq, :, :] = K_a*(np.prod(q_r, axis=1)[idxq])

    # If the order is greater than 1, carry out the sums R and compute the full double integral
    ##### Higher order is still broken, need to fix #####
    elif obj.order > 1:

        tpp, tmm, ELEL = np.meshgrid(obj.tp, obj.tm, obj.EL, sparse=True)
        # *A((tpp+tmm)/(np.sqrt(2)))*np.conj(A((tpp-tmm)/(np.sqrt(2))))#*R((tpp+tmm)/(np.sqrt(2)),(tpp-tmm)/(np.sqrt(2)))
        K_r = np.exp(1j*(ELEL-obj.E0)*sqrt2*tmm-g(tpp+tmm) /
                     (sqrt2)-np.conj(g((tpp-tmm)/(sqrt2))))

        for idxtm, tm in enumerate(obj.tm, start=0):
            integ_r1[idxtm, :] = np.trapz(
                K_r[(np.abs(len(obj.tm)/2-idxtm)):, idxtm, :], axis=0)

        integ = np.trapz(integ_r1, axis=0)
    ######################################################

    integ_a = np.trapz(K_a, axis=1)
    abs_cross = obj.preA*obj.convEL * \
        np.convolve(integ_a, np.real(H), 'valid')/(np.sum(H))

    integ_f = np.trapz(K_f, axis=1)
    fl_cross = obj.preF*obj.convEL * \
        np.convolve(integ_f, np.real(H), 'valid')/(np.sum(H))

    # plt.plot(p.convEL,abs_cross)
    # plt.plot(p.convEL,fl_cross)
    # plt.show()

    # plt.plot(integ_a)
    # plt.plot(integ_f)
    # plt.show()
    # print p.s_reorg
    # print p.w_reorg
    # print p.reorg

    for l in range(len(obj.wg)):
        if obj.order == 1:
            integ_r[l, :] = np.trapz(K_r[l, :, :], axis=1)
            raman_cross[l, :] = obj.preR*obj.convEL*(obj.convEL-obj.wg[l])**3*np.convolve(
                integ_r[l, :]*np.conj(integ_r[l, :]), np.real(H), 'valid')/(np.sum(H))
        elif obj.order > 1:
            raman_cross[l, :] = obj.preR*obj.convEL*(obj.convEL-obj.wg[l])**3*np.convolve(
                integ_r[l, :], np.real(H), 'valid')/(np.sum(H))

    # plt.plot(p.convEL,fl_cross)
    # plt.plot(p.convEL,abs_cross)
    # plt.show()

    # plt.plot(p.convEL,raman_cross[0])
    # plt.plot(p.convEL,raman_cross[1])
    # plt.plot(p.convEL,raman_cross[2])
    # plt.plot(p.convEL,raman_cross[3])
    # plt.show()
    # exit()

    return abs_cross, fl_cross, raman_cross, obj.boltz_state, obj.boltz_coef


def run_save(obj):
    #global current_time_str
    abs_cross, fl_cross, raman_cross, boltz_states, boltz_coef = cross_sections(
        obj)
    raman_spec = np.zeros((len(obj.rshift), len(obj.rpumps)))

    # get current time as YYMMDD_HH-MM-SS
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y%m%d_%H-%M-%S")

    for i in range(len(obj.rpumps)):
        for l in np.arange(len(obj.wg)):
            raman_spec[:, i] += np.real((raman_cross[l, obj.rp[i]])) * \
                (1/np.pi)*(0.5*obj.res)/((obj.rshift-obj.wg[l])**2+(0.5*obj.res)**2)
    '''
    raman_full = np.zeros((len(convEL),len(rshift)))
    for i in range(len(convEL)):
        for l in np.arange(len(wg)):
            raman_full[i,:] += np.real((raman_cross[l,i]))*(1/np.pi)*(0.5*res)/((rshift-wg[l])**2+(0.5*res)**2)
    '''

    # plt.contour(raman_full)
    # plt.show()

    # make data folder
    '''
    if any([i == 'data' for i in os.listdir('./')]) == True:
        pass
    else:
        os.mkdir('./data')
    '''
    os.mkdir('./'+current_time_str + '_data')

    obj.s_reorg = obj.beta*(obj.L/obj.k)**2/2  # reorganization energy cm^-1
    obj.w_reorg = 0.5*np.sum((obj.delta)**2*obj.wg)  # internal reorganization energy
    obj.reorg = obj.w_reorg + obj.s_reorg  # Total reorganization energy
    np.set_printoptions(threshold=sys.maxsize)
    np.savetxt(current_time_str + "_data/profs.dat",
               np.real(np.transpose(raman_cross)), delimiter="\t")
    np.savetxt(current_time_str + "_data/raman_spec.dat",
               raman_spec, delimiter="\t")
    np.savetxt(current_time_str + "_data/EL.dat", obj.convEL)
    np.savetxt(current_time_str + "_data/deltas.dat", obj.delta)
    np.savetxt(current_time_str + "_data/Abs.dat", np.real(abs_cross))
    np.savetxt(current_time_str + "_data/Fl.dat", np.real(fl_cross))
    # np.savetxt("data/Disp.dat",np.real(disp_cross))
    np.savetxt(current_time_str + "_data/rshift.dat", obj.rshift)
    np.savetxt(current_time_str + "_data/rpumps.dat", obj.rpumps)

    inp_list = [float(x) for x in obj.inp]  # need rewrite
    inp_list[7] = obj.M
    inp_list[0] = obj.gamma
    inp_list[1] = obj.theta
    inp_list[2] = obj.E0
    inp_list[3] = obj.k
    inp_list[8] = obj.n

    np.savetxt(current_time_str + "_data/inp.dat", inp_list)
    np.savetxt(current_time_str + "_data/freqs.dat", obj.wg)

    try:
        obj.abs_exp = np.loadtxt('abs_exp.dat')
        np.savetxt(current_time_str+'_data/abs_exp.dat',
                   obj.abs_exp, delimiter="\t")
    except:
        print('No experimental absorption spectrum found in directory/')

    try:
        obj.profs_exp = np.loadtxt('profs_exp.dat')
        np.savetxt(current_time_str+'_data/profs_exp.dat',
                   obj.profs_exp, delimiter="\t")
    except:
        print('No experimental Raman cross section found in directory/')

    with open(current_time_str + "_data/output.txt", 'w') as o:
        o.write("E00 = "), o.write(str(obj.E0)), o.write(" cm-1 \n")
        o.write("gamma = "), o.write(str(obj.gamma)), o.write(" cm-1 \n")
        o.write("theta = "), o.write(str(obj.theta)), o.write(" cm-1 \n")
        o.write("M = "), o.write(str(obj.M)), o.write(" Angstroms \n")
        o.write("n = "), o.write(str(obj.n)), o.write("\n")
        o.write("T = "), o.write(str(obj.T)), o.write(" Kelvin \n")
        o.write("solvent reorganization energy = "), o.write(
            str(obj.s_reorg)), o.write(" cm-1 \n")
        o.write("internal reorganization energy = "), o.write(
            str(obj.w_reorg)), o.write(" cm-1 \n")
        o.write("reorganization energy = "), o.write(
            str(obj.reorg)), o.write(" cm-1 \n\n")
        o.write("Boltzmann averaged states and their corresponding weights \n")
        o.write(str(obj.boltz_coef)), o.write("\n")
        o.write(str(obj.boltz_state)), o.write("\n")
    o.close()

    with open(current_time_str + "_data/inp_new.txt", 'w') as file:
        # Write the data to the file
        file.write(f"{obj.gamma} # gamma linewidth parameter (cm^-1)\n")
        file.write(
            f"{obj.theta} # theta static inhomogeneous linewidth parameter (cm^-1)\n")
        file.write(f"{obj.E0} # E0 (cm^-1)\n")
        file.write(f"{obj.k} # kappa solvent parameter\n")
        file.write(f"{obj.ts} # time step (ps)\n")
        file.write(f"{obj.ntime} # number of time steps\n")
        file.write(
            f"{obj.EL_reach} # range plus and minus E0 to calculate lineshapes\n")
        file.write(f"{obj.M} # transition length M (Angstroms)\n")
        file.write(f"{obj.n} # refractive index n\n")
        file.write(f"{obj.inp[9]} # start raman shift axis (cm^-1)\n")
        file.write(f"{obj.inp[10]} # end raman shift axis (cm^-1)\n")
        file.write(f"{obj.inp[11]} # rshift axis step size (cm^-1)\n")
        file.write(f"{obj.inp[12]} # raman spectrum resolution (cm^-1)\n")
        file.write(f"{obj.T} # Temperature (K)\n")
        file.write(
            f"{obj.inp[14]} # convergence for sums # no effect since order > 1 broken\n")
        file.write(f"{obj.inp[15]} # Boltz Toggle\n")
    return resram_data(current_time_str + "_data")

def raman_residual(param,fit_obj):
    if fit_obj is None:
        fit_obj = load_input()
    for i in range(len(fit_obj.delta)):
        fit_obj.delta[i] = param.valuesdict()['delta'+str(i)]
    fit_obj.gamma = param.valuesdict()['gamma']
    fit_obj.M = param.valuesdict()['transition_length']
    fit_obj.k = param.valuesdict()['kappa'] # kappa parameter
    fit_obj.theta = param.valuesdict()['theta'] # kappa parameter
    fit_obj.E0 = param.valuesdict()['E0'] # kappa parameter
    #print(delta,gamma,M,k,theta,E0)
    abs_cross,fl_cross,raman_cross,boltz_state,boltz_coef  = cross_sections(fit_obj)
    correlation = (np.corrcoef(np.real(abs_cross), fit_obj.abs_exp[:,1])[0, 1])
    #print("Correlation of absorption is "+ str(correlation))
    #Minimize the negative correlation to get better fit

    if fit_obj.profs_exp.ndim == 1:    #Convert 1D array to 2D
        fit_obj.profs_exp = np.reshape(fit_obj.profs_exp,(-1,1))
        #print("Raman cross section expt is converted to a 2D array")
    sigma = np.zeros_like(fit_obj.delta)
    for i in range(len(fit_obj.rpumps)):
    
        for j in range(len(fit_obj.wg)):
            #print(j,i)
            sigma[j] = sigma[j] + (1e7*(np.real(raman_cross[j,fit_obj.rp[i]])-fit_obj.profs_exp[j,i]))**2


    total_sigma = np.sum(sigma)
    #print("Total Raman sigma is "+ str(total_sigma))
    loss= total_sigma - 300*(correlation - 1)
    #print(loss)
    return loss, total_sigma, 300*(1-correlation)



def param_init(fit_switch, obj = None):
    if obj is None:
        obj = load_input()
    params_lmfit = lmfit.Parameters()
    for i in range(len(obj.delta)):
        if fit_switch[i] == 1:
            params_lmfit.add('delta'+str(i),value=obj.delta[i],min=0.0,max=1.0)
        else:
            params_lmfit.add('delta'+str(i),value=obj.delta[i],vary = False)
    
    if fit_switch[len(obj.delta)]==1:
        params_lmfit.add('gamma',value = obj.gamma,min = 0.6*obj.gamma, max = 1.4*obj.gamma)
    else:
        params_lmfit.add('gamma',value = obj.gamma,vary = False)

    if fit_switch[len(obj.delta)+1]==1:
        params_lmfit.add('transition_length',value = obj.M,min = 0.8*obj.M, max = 1.2*obj.M)
    else:
        params_lmfit.add('transition_length',value = obj.M,vary =False)

    if fit_switch[len(obj.delta)+2]==1:
        params_lmfit.add('theta',value = obj.theta,min = 0.5*obj.theta, max = 1.5*obj.theta)
    else:
        params_lmfit.add('theta',value = obj.theta,vary = False)

    if fit_switch[len(obj.delta)+3]==1:
        params_lmfit.add('kappa',value = obj.k,min = 0.9*obj.k, max = 1.1*obj.k)
    else:
        params_lmfit.add('kappa',value = obj.k,vary = False)

    if fit_switch[len(obj.delta)+5]==1:
        params_lmfit.add('E0',value = obj.E0,min = 0.95*obj.E0, max = 1.05*obj.E0)
    else:
        params_lmfit.add('E0',value = obj.E0,vary = False)

    #print("Initial parameters: "+ str(params_lmfit))
    return params_lmfit


class resram_data:
    def __init__(self, input):
        self.freqs = np.loadtxt(input+'/freqs.dat')
        self.rpumps = np.loadtxt(input+'/rpumps.dat')
        self.abs = np.loadtxt(input+'/Abs.dat')
        self.EL = np.loadtxt(input+'/EL.dat')
        self.fl = np.loadtxt(input+'/Fl.dat')
        self.raman_spec = np.loadtxt(input+'/raman_spec.dat')
        self.rshift = np.loadtxt(input+'/rshift.dat')
        self.profs = np.loadtxt(input+'/profs.dat')
        self.inp = np.loadtxt(input+'/inp.dat')
        self.M = self.inp[7]
        self.gamma = self.inp[0]
        self.theta = self.inp[1]
        self.E0 = self.inp[2]
        self.kappa = self.inp[3]
        self.n = self.inp[8]
        try:
            self.abs_exp = np.loadtxt('abs_exp.dat')
        except:
            print('No experimental absorption spectrum found in directory/')
        try:
            self.abs_exp = np.loadtxt(input+'/abs_exp.dat')
        except:
            print('No experimental absorption spectrum found in directory ' + input)
        try:
            self.profs_exp = np.loadtxt('profs_exp.dat')
        except:
            print('No experimental Raman cross section found in directory/')
        try:
            self.profs_exp = np.loadtxt(input+'/profs_exp.dat')
        except:
            print('No experimental Raman cross section found in directory ' + input)
        self.fig_profs, self.ax_profs = plt.subplots()
        self.fig_abs, self.ax_abs = plt.subplots()
        self.fig_raman, self.ax_raman = plt.subplots()

    def plot(self):
        # divide color map to number of freqs
        colors = plt.cm.hsv(np.linspace(0, 1, len(self.freqs)))
        cmap = ListedColormap(colors)
        # plot raman spectra at all excitation
        for i in range(len(self.rpumps)-1):
            self.ax_raman.plot(self.rshift, self.raman_spec[:, i])
        # plt.xlim(1100,1800)
        self.fig_raman.show()

        # plot excitation profile with expt value
        for i in range(len(self.rpumps)):  # iterate over pump wn
            # rp = min(range(len(convEL)),key=lambda j:abs(convEL[j]-rpumps[i]))
            min_diff = float('inf')
            rp = None

            # iterate over all exitation wn to find the one closest to pump
            for j in range(len(self.EL)):
                diff = np.absolute(self.EL[j] - self.rpumps[i])
                if diff < min_diff:
                    min_diff = diff
                    rp = j
            # print(rp)
            for j in range(len(self.freqs)):  # iterate over all raman freqs
                # print(j,i)
                # sigma[j] = sigma[j] + (1e8*(np.real(raman_cross[j,rp])-rcross_exp[j,i]))**2
                color = cmap(j)
                self.ax_profs.plot(self.EL, self.profs[:, j], color=color)
                try:
                    self.ax_profs.plot(
                        self.EL[rp], self.profs_exp[j, i], "o", color=color)
                except:
                    continue
                    print("no experimental Raman cross section data")

        # ax.set_xlim(16000,22500)
        # ax.set_ylim(0,0.5e-7)
        self.fig_profs.show()
        self.ax_abs.plot(self.EL, self.abs)
        self.ax_abs.plot(self.EL, self.fl)
        try:
            self.ax_abs.plot(self.EL, self.abs_exp[:, -1])
        except:
            print("no experimental absorption data")
        self.fig_abs.show()



class WorkerSignals(QObject):
    result_ready = pyqtSignal(str)
    finished = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self,obj_load, tolerance,maxnfev,fit_alg,fit_switch):
        super().__init__()
        self.signals = WorkerSignals()
        self.obj_load = obj_load
        self.tolerance = tolerance
        self.maxnfev = maxnfev
        self.fit_alg = fit_alg
        self.fit_switch = fit_switch

    @pyqtSlot()
    def run(self):
        # global delta, M, gamma, maxnfev, tolerance, fit_alg
        params_lmfit = param_init(self.fit_switch,self.obj_load)
        
        print("Fit is running, please wait...\n")
        # raman_residual(params_lmfit)
        # notice = QMessageBox()
        # notice.setText("Fit is running, please wait...")
        # notice.setStandardButtons(QMessageBox.NoButton) #No button needed
        # notice.exec_()
        fit_kws = dict(tol=self.tolerance)
        try:
            result = lmfit.minimize(raman_residual, params_lmfit, args=(self.obj_load,), method=self.fit_alg,
                                    **fit_kws, max_nfev=self.maxnfev)  # max_nfev = 10000000, **fit_kws
        except:
            print(
                "Something went wrong before fitting start. Use powell algorithm instead")
            result = lmfit.minimize(
                raman_residual, params_lmfit, args=(self.obj_load,),method="powell", **fit_kws, max_nfev=self.maxnfev)
        print(lmfit.fit_report(result))
        for i in range(len(self.obj_load.delta)):
            self.obj_load.delta[i] = result.params.valuesdict()['delta'+str(i)]
        self.obj_load.gamma = result.params.valuesdict()['gamma']
        self.obj_load.M = result.params.valuesdict()['transition_length']
        self.obj_load.k = result.params.valuesdict()['kappa'] # kappa parameter
        self.obj_load.theta = result.params.valuesdict()['theta'] # kappa parameter
        self.obj_load.E0 = result.params.valuesdict()['E0'] # kappa parameter
        run_save(self.obj_load)
        print("Fit done\n")
        self.signals.result_ready.emit("Fit done")
        self.signals.finished.emit(self.obj_load)


class SpectrumApp(QMainWindow):
    def __init__(self):
        super().__init__()
        '''
        #output to outputlogger
        self.output_logger = OutputWidget()
        
        try:
            sys.stdout = self.output_logger
        except:
            print("Cannot redirect output")
        '''
        # multithread
        self.dir = ''
        self.threadpool = QThreadPool()
        #print("Multithreading with maximum %d threads" %self.threadpool.maxThreadCount())
        self.load_files()
        self.plot_switch = np.ones(len(self.obj_load.delta)+18)
        self.fit_switch = np.ones(len(self.obj_load.delta)+18)
        self.setWindowTitle("Raman Spectrum Analyzer")
        self.setGeometry(100, 100, 960, 540)
        # main layout horizontal
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        # left layout vertical
        self.left_layout = QVBoxLayout()

        # Calculate the figure size in inches based on pixels and screen DPI
        dpi = self.physicalDpiX()  # Get the screen's DPI
        fig_width_pixels = 1280  # Desired figure width in pixels
        fig_height_pixels = 720  # Desired figure height in pixels
        fig_width = fig_width_pixels / dpi
        fig_height = fig_height_pixels / dpi
        self.canvas = FigureCanvas(plt.figure(
            figsize=(fig_width, fig_height)))  # fig profs
        self.canvas2 = FigureCanvas(plt.figure(
            figsize=(fig_width/2, fig_height)))  # fig raman spec
        self.canvas3 = FigureCanvas(plt.figure(
            figsize=(fig_width/2, fig_height)))  # fig abs
        self.left_layout.addWidget(self.canvas, 5)
        self.layout.addLayout(self.left_layout, 3)
        self.left_bottom_layout = QHBoxLayout()
        self.left_bottom_layout.addWidget(self.canvas2, 7)
        self.left_bottom_layout.addWidget(self.canvas3, 3)
        self.left_layout.addLayout(self.left_bottom_layout, 3)

        # Add a Python output section
        # self.left_layout.addWidget(self.output_logger, 1)  # Use a stretch factor of 1.5
        self.right_layout = QVBoxLayout()
        self.create_variable_table()
        # self.right_layout.addWidget(self.table_widget) #included in create_variable_table

        self.create_buttons()
        self.layout.addLayout(self.right_layout, 1)
        # timer for updating plot
        self.update_timer = QTimer(self)
        
        self.init_ui()
        
    def init_ui(self):
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax2 = self.canvas2.figure.add_subplot(111)
        self.ax3 = self.canvas3.figure.add_subplot(111)

        self.plot_data()

        print("Initialized")
        self.showMaximized()

    def sendto_table(self):
        self.table_widget.itemChanged.disconnect(self.update_data)

        for row in range(len(self.obj_load.delta)):
            label = QTableWidgetItem(f"delta@{self.obj_load.wg[row]:.3f} cm-1")
            self.table_widget.setItem(
                row, 1, QTableWidgetItem(f"{self.obj_load.delta[row]:.2f}"))
            self.table_widget.setItem(row, 0, label)
        self.table_widget.setItem(len(self.obj_load.delta), 0, QTableWidgetItem("gamma"))
        self.table_widget.setItem(len(self.obj_load.delta), 1, QTableWidgetItem(str(self.obj_load.gamma)))
        self.table_widget.setItem(
            len(self.obj_load.delta)+1, 0, QTableWidgetItem("Transition Length"))
        self.table_widget.setItem(len(self.obj_load.delta)+1, 1, QTableWidgetItem(str(self.obj_load.M)))
        self.table_widget.setItem(len(self.obj_load.delta)+2, 0, QTableWidgetItem("theta"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+2, 1, QTableWidgetItem(str(self.obj_load.theta)))
        self.table_widget.setItem(len(self.obj_load.delta)+3, 0, QTableWidgetItem("kappa"))
        self.table_widget.setItem(len(self.obj_load.delta)+3, 1, QTableWidgetItem(str(self.obj_load.k)))
        self.table_widget.setItem(
            len(self.obj_load.delta)+4, 0, QTableWidgetItem("Refractive Index"))
        self.table_widget.setItem(len(self.obj_load.delta)+4, 1, QTableWidgetItem(str(self.obj_load.n)))
        self.table_widget.setItem(len(self.obj_load.delta)+5, 0, QTableWidgetItem("E00"))
        self.table_widget.setItem(len(self.obj_load.delta)+5, 1, QTableWidgetItem(str(self.obj_load.E0)))
        self.table_widget.setItem(
            len(self.obj_load.delta)+6, 0, QTableWidgetItem("Time step (ps)"))
        self.table_widget.setItem(len(self.obj_load.delta)+6, 1, QTableWidgetItem(str(self.obj_load.ts)))
        self.table_widget.setItem(
            len(self.obj_load.delta)+7, 0, QTableWidgetItem("Time step number"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+7, 1, QTableWidgetItem(str(self.obj_load.ntime)))
        self.table_widget.setItem(
            len(self.obj_load.delta)+17, 0, QTableWidgetItem("Temp (K)"))
        self.table_widget.setItem(len(self.obj_load.delta)+17, 1, QTableWidgetItem(str(self.obj_load.T)))
        self.table_widget.setItem(
            len(self.obj_load.delta)+18, 0, QTableWidgetItem("Raman maxcalc"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+18, 1, QTableWidgetItem(str(self.obj_load.raman_maxcalc)))
        self.table_widget.itemChanged.connect(self.update_data)
        self.update_spectrum()

    def load_table(self):        
        
        for i in range(len(self.obj_load.delta)):
            self.obj_load.delta[i] = float(self.table_widget.item(i, 1).text())
            self.plot_switch[i] = int(float(self.table_widget.item(i, 2).text()))
            self.fit_switch[i] = int(float(self.table_widget.item(i, 3).text()))

        self.obj_load.gamma = float(self.table_widget.item(len(self.obj_load.delta), 1).text())
        self.fit_switch[len(self.obj_load.delta)] = int(
            float(self.table_widget.item(len(self.obj_load.delta), 3).text()))
        self.obj_load.M = float(self.table_widget.item(len(self.obj_load.delta)+1, 1).text())
        self.fit_switch[len(
            self.obj_load.delta)+1] = int(float(self.table_widget.item(len(self.obj_load.delta)+1, 3).text()))
        self.obj_load.theta = float(self.table_widget.item(
            len(self.obj_load.delta)+2, 1).text())  # theta parameter
        self.fit_switch[len(
            self.obj_load.delta)+2] = int(float(self.table_widget.item(len(self.obj_load.delta)+2, 3).text()))
        self.obj_load.k = float(self.table_widget.item(
            len(self.obj_load.delta)+3, 1).text())  # kappa parameter
        self.fit_switch[len(
            self.obj_load.delta)+3] = int(float(self.table_widget.item(len(self.obj_load.delta)+3, 3).text()))
        self.obj_load.n = float(self.table_widget.item(
            len(self.obj_load.delta)+4, 1).text())  # refractive index
        self.obj_load.E0 = float(self.table_widget.item(
            len(self.obj_load.delta)+5, 1).text())  # E00 parameter
        self.fit_switch[len(
            self.obj_load.delta)+5] = int(float(self.table_widget.item(len(self.obj_load.delta)+5, 3).text()))
        self.obj_load.ts = float(self.table_widget.item(len(self.obj_load.delta)+6, 1).text())
        self.obj_load.ntime = float(self.table_widget.item(len(self.obj_load.delta)+7, 1).text())

        self.fit_alg = self.table_widget.item(
            len(self.obj_load.delta)+8, 1).text()  # fitting algorithm
        self.maxnfev = int(self.table_widget.item(
            len(self.obj_load.delta)+9, 1).text())  # max fitting steps
        self.tolerance = float(self.table_widget.item(
            len(self.obj_load.delta)+10, 1).text())  # fitting tolerance
        self.raman_xmin = float(self.table_widget.item(len(self.obj_load.delta)+11, 1).text())
        self.raman_xmax = float(self.table_widget.item(len(self.obj_load.delta)+12, 1).text())
        self.profs_xmin = float(self.table_widget.item(len(self.obj_load.delta)+13, 1).text())
        self.profs_xmax = float(self.table_widget.item(len(self.obj_load.delta)+14, 1).text())
        self.abs_xmin = float(self.table_widget.item(len(self.obj_load.delta)+15, 1).text())
        self.abs_xmax = float(self.table_widget.item(len(self.obj_load.delta)+16, 1).text())
        self.obj_load.T = float(self.table_widget.item(len(self.obj_load.delta)+17, 1).text())
        self.obj_load.raman_maxcalc = float(self.table_widget.item(len(self.obj_load.delta)+18, 1).text())

    def clear_canvas(self):
        if self.ax is not None:
            self.ax.clear()  # Clear the current plot
            self.canvas.draw()  # Redraw the canvas to clear it

        if self.ax2 is not None:
            self.ax2.clear()  # Clear the current plot
            self.canvas2.draw()  # Redraw the canvas to clear it

        if self.ax3 is not None:
            self.ax3.clear()  # Clear the current plot
            self.canvas3.draw()  # Redraw the canvas to clear it

    def start_update_timer(self):
        self.update_timer.start(1000)  # 1 seconds (in milliseconds)

    def stop_update_timer(self):
        self.update_timer.stop()

    def fit(self):
        self.load_table()
        print("Initial deltas: "+str(self.obj_load.delta))
        self.worker = Worker(self.obj_load, self.tolerance,self.maxnfev,self.fit_alg,self.fit_switch)  # thread for fitting
        self.fit_button.setEnabled(False)
        self.fit_button.setText("Fit Running")
        self.threadpool.start(self.worker)
        # after fitting, global variables should be optimized values
        # self.plot_data()
        self.worker.signals.finished.connect(self.handle_worker_result)
        self.worker.signals.result_ready.connect(self.update_fit)

    @pyqtSlot(object)
    def handle_worker_result(self, result_object):
        # Handle the result object received from the worker
        self.obj_load = result_object
        print("Received fitting result")
        # You can use the result_object in the main window as needed
        
    def update_fit(self, result):
        self.fit_button.setText("Fit")
        self.fit_button.setEnabled(True)  # Re-enable the button
        self.sendto_table()

    def on_toggle(self, state):
        if state:
            self.start_update_timer()
            self.update_timer.timeout.connect(self.sendto_table)
        else:
            self.update_timer.timeout.disconnect(self.sendto_table)
            self.stop_update_timer()

    def plot_data(self):
        self.clear_canvas()
        self.load_table()
        abs_cross, fl_cross, raman_cross, boltz_states, boltz_coef = cross_sections(self.obj_load)
        raman_spec = np.zeros((len(self.obj_load.rshift), len(self.obj_load.rpumps)))
        for i in range(len(self.obj_load.rpumps)):
            # rp = min(range(len(convEL)),key=lambda j:abs(convEL[j]-rpumps[i]))
            min_diff = float('inf')
            min_index = None

            for j in range(len(self.obj_load.convEL)):
                diff = np.absolute(self.obj_load.convEL[j] - self.obj_load.rpumps[i])
                if diff < min_diff:
                    min_diff = diff
                    min_index = j

            rp = min_index
            # print(rp)
            # print(rpumps)
            for l in np.arange(len(self.obj_load.wg)):
                raman_spec[:, i] += np.real((raman_cross[l, rp])) * \
                    (1/np.pi)*(0.5*self.obj_load.res)/((self.obj_load.rshift-self.obj_load.wg[l])**2+(0.5*self.obj_load.res)**2)
            nm = 1e7/self.obj_load.rpumps[i]
            self.ax2.plot(self.obj_load.rshift, np.real((raman_spec)[:, i]), label=f'{nm:.3f} nm laser')  # plot raman spectrum
        self.ax2.set_title('Raman Spectra')
        self.ax2.set_xlim(self.raman_xmin, self.raman_xmax)
        self.ax2.set_xlabel('Raman Shift (cm-1)')
        self.ax2.set_ylabel(
            'Raman Cross Section \n(1e-14 Angstrom**2/Molecule)')
        self.ax2.legend(loc='best')
        self.canvas2.draw()

        # Plot Raman excitation profiles
        for i in range(len(self.obj_load.rpumps)):  # iterate over pump wn
            min_diff = float('inf')
            rp = None

            # iterate over all exitation wn to find the one closest to pump
            for j in range(len(self.obj_load.convEL)):
                diff = np.absolute(self.obj_load.convEL[j] - self.obj_load.rpumps[i])
                if diff < min_diff:
                    min_diff = diff
                    rp = j
                # print(rp)

            for j in range(len(self.obj_load.wg)):  # iterate over all raman freqs
                # print(j,i)
                # sigma[j] = sigma[j] + (1e8*(np.real(raman_cross[j,rp])-rcross_exp[j,i]))**2
                if self.plot_switch[j] == 1:
                    color = self.obj_load.cmap(j)
                    self.ax.plot(self.obj_load.convEL[rp], self.obj_load.profs_exp[j, i], "o", color=color)
        for j in range(len(self.obj_load.wg)):  # iterate over all raman freqs
            if self.plot_switch[j] == 1:
                color = self.obj_load.cmap(j)
                self.ax.plot(self.obj_load.convEL, np.real(np.transpose(raman_cross))[
                             :, j], color=color, label=f'{self.obj_load.wg[j]:.2f} cm-1')
        self.ax.set_title('Raman Excitation Profiles')
        self.ax.set_xlim(self.profs_xmin, self.profs_xmax)
        self.ax.set_xlabel('Wavenumber (cm-1)')
        self.ax.set_ylabel(
            'Raman Cross Section \n(1e-14 Angstrom**2/Molecule)')
        self.ax.legend(loc='best', ncol=2, prop={'size': 8})
        self.canvas.draw()

        # plot absorption
        self.ax3.plot(self.obj_load.convEL, np.real(abs_cross), label='Abs')
        self.ax3.plot(self.obj_load.convEL, np.real(fl_cross), label='FL')
        try:
            self.ax3.plot(self.obj_load.convEL, self.obj_load.abs_exp[:, 1], label='Abs expt.')
        except:
            print("No experimental absorption spectrum")
        self.ax3.set_title('Absorption and Emission Spectra')
        self.ax3.set_xlim(self.abs_xmin, self.abs_xmax)
        self.ax3.set_xlabel('Wavenumber (cm-1)')
        self.ax3.set_ylabel('Cross Section \n(1e-14 Angstrom**2/Molecule)')
        self.ax3.legend(loc='best')
        self.canvas3.draw()

    def create_variable_table(self):
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(4)
        self.table_widget.setRowCount(len(self.obj_load.delta)+19)
        self.table_widget.setHorizontalHeaderLabels(
            ["Variables", "Values", "Plot Raman \nEx. Profile", "Fit?"])
        self.table_widget.itemChanged.connect(self.update_data)
        self.obj2table()
        print("Initialized. Files loaded from the working folder.")
        #initialize parameters for ResRam Gui only
        self.fit_alg = self.table_widget.item(
            len(self.obj_load.delta)+8, 1).text()  # fitting algorithm
        self.maxnfev = int(self.table_widget.item(
            len(self.obj_load.delta)+9, 1).text())  # max fitting steps
        self.tolerance = float(self.table_widget.item(
            len(self.obj_load.delta)+10, 1).text())  # fitting tolerance
        self.raman_xmin = float(self.table_widget.item(len(self.obj_load.delta)+11, 1).text())
        self.raman_xmax = float(self.table_widget.item(len(self.obj_load.delta)+12, 1).text())
        self.profs_xmin = float(self.table_widget.item(len(self.obj_load.delta)+13, 1).text())
        self.profs_xmax = float(self.table_widget.item(len(self.obj_load.delta)+14, 1).text())
        self.abs_xmin = float(self.table_widget.item(len(self.obj_load.delta)+15, 1).text())
        self.abs_xmax = float(self.table_widget.item(len(self.obj_load.delta)+16, 1).text())
        # Set headers to resize to contents
        self.table_widget.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self.table_widget.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents)

        self.right_layout.addWidget(self.table_widget)

    def select_subfolder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly | QFileDialog.ReadOnly

        self.folder_path = QFileDialog.getExistingDirectory(
            self, "Select Subfolder", os.getcwd(), options=options)

        if self.folder_path:
            print("Selected folder:", self.folder_path)
            self.dir = self.folder_path+"/"
            self.obj_load = load_input(self.dir)
            self.sendto_table()
            self.dirlabel.setText("Current data folder: "+self.dir)
            self.plot_data()

    def create_buttons(self):
        # self.add_button = QPushButton("Add Data")
        self.update_button = QPushButton("Update table")
        self.save_button = QPushButton("Save parameters")
        self.initialize_button = QPushButton("Intialize")
        self.fit_button = QPushButton("Fit")
        self.load_button = QPushButton("Load folder")

        # self.add_button.clicked.connect(self.add_data)
        self.update_button.clicked.connect(self.sendto_table)
        self.save_button.clicked.connect(self.save_data)
        self.initialize_button.clicked.connect(self.initialize)
        self.fit_button.clicked.connect(self.fit)
        self.load_button.clicked.connect(self.select_subfolder)

        # toggle switch
        self.updater_switch = QCheckBox("Auto Refresh")
        self.updater_switch.setCheckable(True)
        self.updater_switch.setStyleSheet(
            "QCheckBox::indicator { width: 40px; height: 20px; }")
        self.updater_switch.toggled.connect(self.on_toggle)
        button_layout = QHBoxLayout()

        # Add a stretch to push widgets to the right
        spacer = QSpacerItem(
            40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.dirlabel = QLabel("Current data folder:/ "+self.dir)
        button_layout.addWidget(self.dirlabel)
        button_layout.addItem(spacer)
        # button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.update_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.initialize_button)
        button_layout.addWidget(self.fit_button)
        button_layout.addWidget(self.updater_switch)
        # button_layout.addWidget(self.delete_button)
        self.left_layout.addLayout(button_layout)

    def update_spectrum(self):
        # Clear the previous series data
        self.plot_data()

    def add_data(self):#no use
        # Add a new row to the table
        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)
        self.table_widget.setItem(
            row_position, 0, QTableWidgetItem("New Data"))
        self.table_widget.setItem(row_position, 1, QTableWidgetItem("0.0"))
        self.update_spectrum()

    def save_data(self):
        run_save(self.obj_load)
        
    def load_files(self):
        self.obj_load = load_input(self.dir)
        return self.obj_load
        
    def initialize(self):
        self.dir = ''
        self.load_files()
        self.obj2table()
        print("Initialized. Files loaded from the working folder.")        
        self.update_spectrum()
        self.dirlabel.setText("Current data folder: /"+self.dir)
        
    def obj2table(self):
        self.table_widget.itemChanged.disconnect(self.update_data)        
        for row in range(len(self.obj_load.delta)):
            item = QTableWidgetItem(f"{self.obj_load.delta[row]:.3f}")
            label = QTableWidgetItem(f"delta@{self.obj_load.wg[row]:2f} cm-1")
            self.table_widget.setItem(row, 0, label)
            self.table_widget.setItem(row, 1, item)
            self.table_widget.setItem(row, 2, QTableWidgetItem("1"))
            self.table_widget.setItem(row, 3, QTableWidgetItem("1"))
        self.table_widget.setItem(len(self.obj_load.delta), 0, QTableWidgetItem("gamma"))
        self.table_widget.setItem(len(self.obj_load.delta), 1, QTableWidgetItem(str(self.obj_load.inp[0])))
        self.table_widget.setItem(len(self.obj_load.delta), 3, QTableWidgetItem("1"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+1, 0, QTableWidgetItem("Transition Length (A)"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+1, 1, QTableWidgetItem(str(self.obj_load.inp[7])))
        self.table_widget.setItem(len(self.obj_load.delta)+1, 3, QTableWidgetItem("1"))
        self.table_widget.setItem(len(self.obj_load.delta)+2, 0, QTableWidgetItem("theta"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+2, 1, QTableWidgetItem(str(self.obj_load.inp[1])))
        self.table_widget.setItem(len(self.obj_load.delta)+2, 3, QTableWidgetItem("1"))
        self.table_widget.setItem(len(self.obj_load.delta)+3, 0, QTableWidgetItem("kappa"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+3, 1, QTableWidgetItem(str(self.obj_load.inp[3])))
        self.table_widget.setItem(len(self.obj_load.delta)+3, 3, QTableWidgetItem("0"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+4, 0, QTableWidgetItem("Refractive Index"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+4, 1, QTableWidgetItem(str(self.obj_load.inp[8])))
        self.table_widget.setItem(len(self.obj_load.delta)+4, 3, QTableWidgetItem("0"))
        self.table_widget.setItem(len(self.obj_load.delta)+5, 0, QTableWidgetItem("E00"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+5, 1, QTableWidgetItem(str(self.obj_load.inp[2])))
        self.table_widget.setItem(len(self.obj_load.delta)+5, 3, QTableWidgetItem("0"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+6, 0, QTableWidgetItem("Time step (ps)"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+6, 1, QTableWidgetItem(str(self.obj_load.inp[4])))
        self.table_widget.setItem(
            len(self.obj_load.delta)+7, 0, QTableWidgetItem("Time step number"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+7, 1, QTableWidgetItem(str(self.obj_load.inp[5])))
        self.table_widget.setItem(
            len(self.obj_load.delta)+8, 0, QTableWidgetItem("Fitting lgorithm"))
        self.table_widget.setItem(len(self.obj_load.delta)+8, 1, QTableWidgetItem("powell"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+9, 0, QTableWidgetItem("Fitting maxnfev"))
        self.table_widget.setItem(len(self.obj_load.delta)+9, 1, QTableWidgetItem("100"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+10, 0, QTableWidgetItem("Fitting tolerance"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+10, 1, QTableWidgetItem("0.00000001"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+11, 0, QTableWidgetItem("Raman xmin"))
        self.table_widget.setItem(len(self.obj_load.delta)+11, 1, QTableWidgetItem("200"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+12, 0, QTableWidgetItem("Raman xmax"))
        self.table_widget.setItem(len(self.obj_load.delta)+12, 1, QTableWidgetItem("1700"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+13, 0, QTableWidgetItem("Exci. Prof. xmin"))
        self.table_widget.setItem(len(self.obj_load.delta)+13, 1, QTableWidgetItem("15000"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+14, 0, QTableWidgetItem("Exci. Prof. xmax"))
        self.table_widget.setItem(len(self.obj_load.delta)+14, 1, QTableWidgetItem("22500"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+15, 0, QTableWidgetItem("Abs/FL xmin"))
        self.table_widget.setItem(len(self.obj_load.delta)+15, 1, QTableWidgetItem("12000"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+16, 0, QTableWidgetItem("Abs/Fl xmax"))
        self.table_widget.setItem(len(self.obj_load.delta)+16, 1, QTableWidgetItem("23000"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+17, 0, QTableWidgetItem("Temp (K)"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+17, 1, QTableWidgetItem(str(self.obj_load.inp[13])))
        self.table_widget.setItem(
            len(self.obj_load.delta)+18, 0, QTableWidgetItem("Raman maxcalc"))
        self.table_widget.setItem(
            len(self.obj_load.delta)+18, 1, QTableWidgetItem(str(self.obj_load.inp[10])))
        self.table_widget.itemChanged.connect(self.update_data)


    def update_data(self):#No use
        # Update the selected row in the table
        selected_rows = self.table_widget.selectionModel().selectedRows()
        for index in selected_rows:
            name = self.table_widget.item(index.row(), 0).text()
            value = float(self.table_widget.item(index.row(), 1).text())
            # You can edit the name and value here as needed
        self.update_spectrum()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F5:
            self.update_spectrum()
            

class OutputWidget(QTextBrowser):#Not compatible with qrunner. 
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)  # Make the text browser read-only

    def write(self, text):
        # Append the text to the output panel
        self.insertPlainText(text)
        self.ensureCursorVisible()  # Scroll to the latest text
        
def exception_hook(exctype, value, traceback):
    """
    Custom exception hook to handle uncaught exceptions.
    Display an error message box with the exception details.
    """
    msg = f"Unhandled exception: {exctype.__name__}\n{value}"
    QMessageBox.critical(None, "Unhandled Exception", msg)
    sys.__excepthook__(exctype, value, traceback)  # Call default exception hook

def main():
    app = QApplication(sys.argv)
    # Set the custom exception hook
    sys.excepthook = exception_hook
    app.setWindowIcon(QtGui.QIcon('ico.ico'))
    window = SpectrumApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
