from PyQt5.QtCore import Qt, QThreadPool, pyqtSlot, QRunnable, pyqtSignal, QTimer, QObject
from PyQt5.QtWidgets import QLabel, QFileDialog, QCheckBox, QTextBrowser, QHeaderView, QApplication, QMainWindow, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtGui
import sys
from datetime import datetime
import os
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import lmfit

dir = ''

def load(freqs_inp, delta_inp, rpumps_inp, inp_inp, abs_exp_inp, profs_exp_inp):
    global ntime, EL_reach, cmap, ts, abs_exp, profs_exp, wg, we, delta, convEL
    global theta, D, L, M, E0, th, EL, tm, tp, E0_range, gamma, k, beta, eta, S
    global order, Q, preA, preF, preR, boltz_coef, rshift, b, rpumps, res, n, T
    global s_reorg, w_reorg, reorg, convergence, inp
    # Ground state normal mode frequencies cm^-1
    wg = np.asarray(np.loadtxt(dir+freqs_inp))
    # Excited state normal mode frequencies cm^-1
    we = np.asarray(np.loadtxt(dir+freqs_inp))
    # Dimensionless displacements
    delta = np.asarray(np.loadtxt(dir+delta_inp))
    # divide color map to number of freqs
    colors = plt.cm.hsv(np.linspace(0, 1, len(wg)))
    cmap = ListedColormap(colors)
    # print(delta)
    S = (delta**2)/2  # calculate in cross_sections()
    try:
        abs_exp = np.loadtxt(dir+abs_exp_inp)
    except:
        print('No experimental absorption spectrum found in directory/')

    try:
        profs_exp = np.loadtxt(dir+profs_exp_inp)
    except:
        print('No experimental Raman cross section found in directory/')

    with open(dir+inp_inp, 'r') as i:  # loading inp.txt

        inp = i.readlines()

        j = 0
        for l in inp:
            l = l.partition('#')[0]
            l = l.rstrip()
            inp[j] = l
            j += 1

        hbar = 5.3088  # plancks constant cm^-1*ps
        T = float(inp[13])  # Temperature K
        kbT = 0.695*T  # kbT energy (cm^-1/K)*cm^-1=cm^-1
        cutoff = kbT*0.1  # cutoff for boltzmann dist in wavenumbers
        if T > 10.0:
            beta = 1/kbT  # beta cm
            # array of average thermal occupation numbers for each mode
            eta = 1/(np.exp(wg/kbT)-1)
        elif T < 10.0:
            beta = 1/kbT
            # beta = float("inf")
            eta = np.zeros(len(wg))

        gamma = float(inp[0])  # Homogeneous broadening parameter cm^-1
        theta = float(inp[1])  # Static inhomogenous broadening parameter cm^-1
        E0 = float(inp[2])  # E0 cm^-1

        ## Brownian Oscillator parameters ##
        k = float(inp[3])  # kappa parameter
        D = gamma*(1+0.85*k+0.88*k**2)/(2.355+1.76*k)  # D parameter
        L = k*D  # LAMBDA parameter

        s_reorg = beta*(L/k)**2/2  # reorganization energy cm^-1
        w_reorg = 0.5*np.sum((delta)**2*wg)  # internal reorganization energy
        reorg = w_reorg + s_reorg  # Total reorganization energy

        ## Time and energy range stuff ##
        ts = float(inp[4])  # Time step (ps)
        ntime = float(inp[5])  # 175 # ntime steps
        UB_time = ntime*ts  # Upper bound in time range
        t = np.linspace(0, UB_time, int(ntime))  # time range in ps
        EL_reach = float(inp[6])  # How far plus and minus E0 you want
        # range for spectra cm^-1
        EL = np.linspace(E0-EL_reach, E0+EL_reach, 1000)
        # static inhomogeneous convolution range
        E0_range = np.linspace(-EL_reach*0.5, EL_reach*0.5, 501)

        th = np.array(t/hbar)  # t/hbar

        ntime_rot = ntime/np.sqrt(2)
        ts_rot = ts/np.sqrt(2)
        UB_time_rot = ntime_rot*ts_rot
        tp = np.linspace(0, UB_time_rot, int(ntime_rot))
        tm = None
        tm = np.append(-np.flip(tp[1:], axis=0), tp)
        # Excitation axis after convolution with inhomogeneous distribution
        convEL = np.linspace(E0-EL_reach*0.5, E0+EL_reach*0.5,
                             (max(len(E0_range), len(EL))-min(len(E0_range), len(EL))+1))

        M = float(inp[7])  # Transition dipole length angstroms
        n = float(inp[8])  # Refractive index

        # Raman pump wavelengths to compute spectra at
        rpumps = np.asarray(np.loadtxt(dir+rpumps_inp))
        rshift = np.arange(float(inp[9]), float(inp[10]), float(
            inp[11]))  # range and step size of Raman spectrum
        res = float(inp[12])  # Peak width in Raman spectra

        # Determine order from Boltzmann distribution of possible initial states #
        # desired boltzmann coefficient for cutoff
        convergence = float(inp[14])
        boltz_toggle = float(inp[15])

        if boltz_toggle == 1.:
            boltz_states, boltz_coef, dos_energy = boltz_states()
            if T == 0.0:
                state = 0
            else:
                state = min(range(len(boltz_coef)),
                            key=lambda j: abs(boltz_coef[j]-convergence))

            if state == 0:
                order = 1
            else:
                order = max(max(boltz_states[:state])) + 1
        if boltz_toggle == 0:
            boltz_states, boltz_coef, dos_energy = [0, 0, 0]
            order = 1

        a = np.arange(order)
        b = a
        Q = np.identity(len(wg), dtype=int)

        # wq = None
        # wq = np.append(wg,wg)
    i.close()
    ## Prefactors for absorption and Raman cross-sections ##
    if order == 1:
        # (0.3/pi) puts it in differential cross section
        preR = 2.08e-20*(ts**2)
    elif order > 1:
        preR = 2.08e-20*(ts_rot**2)

    preA = ((5.744e-3)/n)*ts
    preF = preA*n**2


def recalc():
    global ntime, raman_maxcalc, ts, EL_reach, cmap, abs_exp, profs_exp, wg, we, delta, convEL, theta, D, L, M, E0, th, EL, tm, tp, E0_range, gamma, k, beta, eta, S, order, Q, preA, preF, preR, boltz_coef, rshift, b, rpumps, res, n, T, s_reorg, w_reorg, reorg, convergence, inp
    hbar = 5.3088  # plancks constant cm^-1*ps
    kbT = 0.695*T  # kbT energy (cm^-1/K)*cm^-1=cm^-1
    cutoff = kbT*0.1  # cutoff for boltzmann dist in wavenumbers
    if T > 10.0:
        beta = 1/kbT  # beta cm
        # array of average thermal occupation numbers for each mode
        eta = 1/(np.exp(wg/kbT)-1)
    elif T < 10.0:
        beta = 1/kbT
        # beta = float("inf")
        eta = np.zeros(len(wg))
    S = (delta**2)/2
    L = k*D  # LAMBDA parameter
    s_reorg = beta*(L/k)**2/2  # reorganization energy cm^-1
    w_reorg = 0.5*np.sum((delta)**2*wg)  # internal reorganization energy
    reorg = w_reorg + s_reorg  # Total reorganization energy
    UB_time = ntime*ts  # Upper bound in time range
    t = np.linspace(0, UB_time, int(ntime))  # time range in ps
    EL = np.linspace(E0-EL_reach, E0+EL_reach, 1000)  # range for spectra cm^-1
    # static inhomogeneous convolution range
    E0_range = np.linspace(-EL_reach*0.5, EL_reach*0.5, 501)

    th = np.array(t/hbar)  # t/hbar

    ntime_rot = ntime/np.sqrt(2)
    ts_rot = ts/np.sqrt(2)
    UB_time_rot = ntime_rot*ts_rot
    tp = np.linspace(0, UB_time_rot, int(ntime_rot))
    tm = None
    tm = np.append(-np.flip(tp[1:], axis=0), tp)
    # Excitation axis after convolution with inhomogeneous distribution
    convEL = np.linspace(E0-EL_reach*0.5, E0+EL_reach*0.5,
                         (max(len(E0_range), len(EL))-min(len(E0_range), len(EL))+1))
    Q = np.identity(len(wg), dtype=int)
    preA = ((5.744e-3)/n)*ts
    preF = preA*n**2
    rshift = np.arange(float(inp[9]), raman_maxcalc, float(inp[11]))


def boltz_states():
    wg = wg.astype(int)
    cutoff = range(int(cutoff))
    dos = range(len(cutoff))
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
        dos[i] = count_combs(cutoff[i], 0, [], None)
        if dos[i] > 0.0:
            boltz_dist.append([np.exp(-cutoff[i]*beta)])
            dos_energy.append(cutoff[i])

    norm = np.sum(boltz_dist)

    np.reshape(states, -1, len(cutoff))

    return states, boltz_dist/norm, dos_energy


# np.set_printoptions(threshold=sys.maxsize)

# warnings.filterwarnings('ignore') # Supresses 'casting to real discards complex part' warning


def g(t):
    D = gamma*(1+0.85*k+0.88*k**2)/(2.355+1.76*k)  # D parameter
    L = k*D  # LAMBDA parameter
    g = ((D/L)**2)*(L*t-1+np.exp(-L*t))+1j*((beta*D**2)/(2*L))*(1-np.exp(-L*t))
    # g = p.gamma*np.abs(t)#
    return g


def A(t):
    # K=np.zeros((len(p.wg),len(t)),dtype=complex)

    if type(t) == np.ndarray:
        K = np.zeros((len(wg), len(th)), dtype=complex)
    else:
        K = np.zeros((len(wg), 1), dtype=complex)
    for l in np.arange(len(wg)):
        K[l, :] = (1+eta[l])*S[l]*(1-np.exp(-1j*wg[l]*t)) + \
            eta[l]*S[l]*(1-np.exp(1j*wg[l]*t))
    A = M**2*np.exp(-np.sum(K, axis=0))
    return A


def R(t1, t2):
    Ra = np.zeros((len(a), len(wg), len(wg), len(EL)), dtype=complex)
    R = np.zeros((len(wg), len(wg), len(EL)), dtype=complex)
    # for l in np.arange(len(p.wg)):
    # 	for q in p.Q:
    for idxq, q in enumerate(Q, start=0):
        for idxl, l in enumerate(q, start=0):

            wg = wg[idxl]
            S = S[idxl]
            eta = eta[idxl]
            if l == 0:
                for idxa, a in enumerate(a, start=0):
                    Ra[idxa, idxq, idxl, :] = ((1./factorial(a))**2)*((eta*(1+eta))**a)*S**(2*a)*(
                        ((1-np.exp(-1j*wg*t1))*np.conj((1-np.exp(-1j*wg*t1))))*((1-np.exp(-1j*wg*t1))*np.conj((1-np.exp(-1j*wg*t1)))))**a
                R[idxq, idxl, :] = np.sum(Ra[:, idxq, idxl, :], axis=0)
            elif l > 0:
                for idxa, a in enumerate(a[l:], start=0):
                    Ra[idxa, idxq, idxl, :] = ((1./(factorial(a)*factorial(a-l))))*(((1+eta)*S*(1-np.exp(-1j*wg*t1))*(
                        1-np.exp(1j*wg*t2)))**a)*(eta*S*(1-np.exp(1j*wg*t1))*(1-np.exp(-1j*wg*t2)))**(a-l)
                R[idxq, idxl, :] = np.sum(Ra[:, idxq, idxl, :], axis=0)
            elif l < 0:
                for idxa, a in enumerate(b[-l:], start=0):
                    Ra[idxa, idxq, idxl, :] = ((1./(factorial(a)*factorial(a+l))))*(((1+eta)*S*(1-np.exp(-1j*wg*t1))*(
                        1-np.exp(1j*wg*t2)))**(a+l))*(eta*S*(1-np.exp(1j*wg*t1))*(1-np.exp(-1j*wg*t2)))**(a)
                R[idxq, idxl, :] = np.sum(Ra[:, idxq, idxl, :], axis=0)
    return np.prod(R, axis=1)


def cross_sections(convEL, delta, theta, D, L, M, E0):
    global tm, EL, wg, abs_cross, fl_cross, raman_cross, boltz_states, boltz_coef, S

    S = (delta**2)/2
    EL = np.linspace(E0-EL_reach, E0+EL_reach, 1000)  # range for spectra cm^-1
    q_r = np.ones((len(wg), len(wg), len(th)), dtype=complex)
    K_r = np.zeros((len(wg), len(EL), len(th)), dtype=complex)
    # elif p.order > 1:
    # 	K_r = np.zeros((len(p.tm),len(p.tp),len(p.wg),len(p.EL)),dtype=complex)
    integ_r1 = np.zeros((len(tm), len(EL)), dtype=complex)
    integ_r = np.zeros((len(wg), len(EL)), dtype=complex)
    raman_cross = np.zeros((len(wg), len(convEL)), dtype=complex)

    if theta == 0.0:
        H = 1.  # np.ones(len(p.E0_range))
    else:
        H = (1/(theta*np.sqrt(2*np.pi)))*np.exp(-((E0_range)**2)/(2*theta**2))

    thth, ELEL = np.meshgrid(th, EL, sparse=True)

    K_a = np.exp(1j*(ELEL-(E0))*thth-g(thth))*A(thth)
    K_f = np.exp(1j*(ELEL-(E0))*thth-np.conj(g(thth)))*np.conj(A(thth))

    ## If the order desired is 1 use the simple first order approximation ##
    if order == 1:
        for idxq, q in enumerate(Q, start=0):
            for idxl, l in enumerate(q, start=0):
                if q[idxl] > 0:
                    q_r[idxq, idxl, :] = (1./factorial(q[idxl]))**(0.5)*(((1+eta[idxl])**(
                        0.5)*delta[idxl])/np.sqrt(2))**(q[idxl])*(1-np.exp(-1j*wg[idxl]*thth))**(q[idxl])
                elif q[idxl] < 0:
                    q_r[idxq, idxl, :] = (1./factorial(np.abs(q[idxl])))**(0.5)*(((eta[l])**(
                        0.5)*delta[l])/np.sqrt(2))**(-q[idxl])*(1-np.exp(1j*wg[idxl]*thth))**(-q[idxl])
            K_r[idxq, :, :] = K_a*(np.prod(q_r, axis=1)[idxq])

    # If the order is greater than 1, carry out the sums R and compute the full double integral
    ##### Higher order is still broken, need to fix #####
    elif order > 1:

        tpp, tmm, ELEL = np.meshgrid(tp, tm, EL, sparse=True)
        # *A((tpp+tmm)/(np.sqrt(2)))*np.conj(A((tpp-tmm)/(np.sqrt(2))))#*R((tpp+tmm)/(np.sqrt(2)),(tpp-tmm)/(np.sqrt(2)))
        K_r = np.exp(1j*(ELEL-E0)*np.sqrt(2)*tmm-g(tpp+tmm) /
                     (np.sqrt(2))-np.conj(g((tpp-tmm)/(np.sqrt(2)))))

        for idxtm, tm in enumerate(tm, start=0):
            integ_r1[idxtm, :] = np.trapz(
                K_r[(np.abs(len(tm)/2-idxtm)):, idxtm, :], axis=0)

        integ = np.trapz(integ_r1, axis=0)
    ######################################################

    integ_a = np.trapz(K_a, axis=1)
    abs_cross = preA*convEL * \
        np.convolve(integ_a, np.real(H), 'valid')/(np.sum(H))

    integ_f = np.trapz(K_f, axis=1)
    fl_cross = preF*convEL * \
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

    for l in range(len(wg)):
        if order == 1:
            integ_r[l, :] = np.trapz(K_r[l, :, :], axis=1)
            raman_cross[l, :] = preR*convEL*(convEL-wg[l])**3*np.convolve(
                integ_r[l, :]*np.conj(integ_r[l, :]), np.real(H), 'valid')/(np.sum(H))
        elif order > 1:
            raman_cross[l, :] = preR*convEL*(convEL-wg[l])**3*np.convolve(
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

    return (abs_cross, fl_cross, raman_cross, boltz_states, boltz_coef)


def run_save():
    global raman_cross, abs_cross, fl_cross, boltz_coef, boltz_states
    abs_cross, fl_cross, raman_cross, boltz_states, boltz_coef = cross_sections(
        convEL, delta, theta, D, L, M, E0)
    raman_spec = np.zeros((len(rshift), len(rpumps)))

    # get current time as YYMMDD_HH-MM-SS
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y%m%d_%H-%M-%S")

    for i in range(len(rpumps)):
        # rp = min(range(len(convEL)),key=lambda j:abs(convEL[j]-rpumps[i]))
        min_diff = float('inf')
        min_index = None

        for j in range(len(convEL)):
            diff = np.absolute(convEL[j] - rpumps[i])
            if diff < min_diff:
                min_diff = diff
                min_index = j

        rp = min_index
        # print(rp)
        # print(rpumps)
        for l in np.arange(len(wg)):
            raman_spec[:, i] += np.real((raman_cross[l, rp])) * \
                (1/np.pi)*(0.5*res)/((rshift-wg[l])**2+(0.5*res)**2)

    raman_full = np.zeros((len(convEL), len(rshift)))
    for i in range(len(convEL)):
        for l in np.arange(len(wg)):
            raman_full[i, :] += np.real((raman_cross[l, i])) * \
                (1/np.pi)*(0.5*res)/((rshift-wg[l])**2+(0.5*res)**2)

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

    s_reorg = beta*(L/k)**2/2  # reorganization energy cm^-1
    w_reorg = 0.5*np.sum((delta)**2*wg)  # internal reorganization energy
    reorg = w_reorg + s_reorg  # Total reorganization energy
    np.set_printoptions(threshold=sys.maxsize)
    np.savetxt(current_time_str + "_data/profs.dat",
               np.real(np.transpose(raman_cross)), delimiter="\t")
    np.savetxt(current_time_str + "_data/raman_spec.dat",
               raman_spec, delimiter="\t")
    np.savetxt(current_time_str + "_data/EL.dat", convEL)
    np.savetxt(current_time_str + "_data/deltas.dat", delta)
    np.savetxt(current_time_str + "_data/Abs.dat", np.real(abs_cross))
    np.savetxt(current_time_str + "_data/Fl.dat", np.real(fl_cross))
    # np.savetxt("data/Disp.dat",np.real(disp_cross))
    np.savetxt(current_time_str + "_data/rshift.dat", rshift)
    np.savetxt(current_time_str + "_data/rpumps.dat", rpumps)

    inp_list = [float(x) for x in inp]  # need rewrite
    inp_list[7] = M
    inp_list[0] = gamma
    inp_list[1] = theta
    inp_list[2] = E0
    inp_list[3] = k
    inp_list[8] = n

    np.savetxt(current_time_str + "_data/inp.txt", inp_list)
    np.savetxt(current_time_str + "_data/freqs.dat", wg)

    try:
        abs_exp = np.loadtxt(dir+'abs_exp.dat')
        np.savetxt(current_time_str+'_data/abs_exp.dat',
                   abs_exp, delimiter="\t")
    except:
        print('No experimental absorption spectrum found in directory/')

    try:
        profs_exp = np.loadtxt(dir+'profs_exp.dat')
        np.savetxt(current_time_str+'_data/profs_exp.dat',
                   profs_exp, delimiter="\t")
    except:
        print('No experimental Raman cross section found in directory/')

    with open(current_time_str + "_data/output.txt", 'w') as o:
        o.write("E00 = "), o.write(str(E0)), o.write(" cm-1 \n")
        o.write("gamma = "), o.write(str(gamma)), o.write(" cm-1 \n")
        o.write("theta = "), o.write(str(theta)), o.write(" cm-1 \n")
        o.write("M = "), o.write(str(M)), o.write(" Angstroms \n")
        o.write("n = "), o.write(str(n)), o.write("\n")
        o.write("T = "), o.write(str(T)), o.write(" Kelvin \n")
        o.write("solvent reorganization energy = "), o.write(
            str(s_reorg)), o.write(" cm-1 \n")
        o.write("internal reorganization energy = "), o.write(
            str(w_reorg)), o.write(" cm-1 \n")
        o.write("reorganization energy = "), o.write(
            str(reorg)), o.write(" cm-1 \n\n")
        o.write("Boltzmann averaged states and their corresponding weights \n")
        o.write(str(boltz_coef)), o.write("\n")
        o.write(str(boltz_states)), o.write("\n")

    o.close()


def raman_residual(param):
    global profs_exp, gamma, M, abs_exp, k, fl_cross, D, L
    for i in range(len(delta)):
        delta[i] = param.valuesdict()['delta'+str(i)]
    gamma = param.valuesdict()['gamma']
    M = param.valuesdict()['transition_length']
    D = gamma*(1+0.85*k+0.88*k**2)/(2.355+1.76*k)  # D parameter
    L = k*D  # LAMBDA parameter
    k = param.valuesdict()['kappa']  # kappa parameter
    theta = param.valuesdict()['theta']  # kappa parameter
    E0 = param.valuesdict()['E0']  # kappa parameter
    print(delta, gamma, M, k, theta, E0)

    abs_cross, fl_cross, raman_cross, boltz_states, boltz_coef = cross_sections(
        convEL, delta, theta, D, L, M, E0)
    correlation = (np.corrcoef(np.real(abs_cross), abs_exp[:, 1])[0, 1])
    print("Correlation of absorption is " + str(correlation))
    # Minimize the negative correlation to get better fit

    if profs_exp.ndim == 1:  # Convert 1D array to 2D
        profs_exp = np.reshape(profs_exp, (-1, 1))
        # print("Raman cross section expt is converted to a 2D array")
    sigma = np.zeros_like(delta)
    for i in range(len(rpumps)):
        # rp = min(range(len(convEL)),key=lambda j:abs(convEL[j]-rpumps[i]))
        min_diff = float('inf')
        rp = None

        for j in range(len(convEL)):
            diff = np.absolute(convEL[j] - rpumps[i])
            if diff < min_diff:
                min_diff = diff
                rp = j
        # print(rp)
        for j in range(len(wg)):
            # print(j,i)
            sigma[j] = sigma[j] + \
                (1e7*(np.real(raman_cross[j, rp])-profs_exp[j, i]))**2

    total_sigma = np.sum(sigma)
    print("Total sigma is " + str(total_sigma))
    loss = total_sigma - 100*(correlation - 1)
    # print(loss)

    return loss


def param_init():
    global fit_switch
    params_lmfit = lmfit.Parameters()
    for i in range(len(delta)):
        if fit_switch[i] == 1:
            params_lmfit.add('delta'+str(i), value=delta[i], min=0.0, max=1.0)
        else:
            params_lmfit.add('delta'+str(i), value=delta[i], vary=False)

    if fit_switch[len(delta)] == 1:
        params_lmfit.add('gamma', value=gamma, min=0.5*gamma, max=1.5*gamma)
    else:
        params_lmfit.add('gamma', value=gamma, vary=False)

    if fit_switch[len(delta)+1] == 1:
        params_lmfit.add('transition_length', value=M, min=0.6*M, max=1.4*M)
    else:
        params_lmfit.add('transition_length', value=M, vary=False)

    if fit_switch[len(delta)+2] == 1:
        params_lmfit.add('theta', value=theta, min=0.7*theta, max=1.3*theta)
    else:
        params_lmfit.add('theta', value=theta, vary=False)

    if fit_switch[len(delta)+3] == 1:
        params_lmfit.add('kappa', value=k, min=0.8*k, max=1.2*k)
    else:
        params_lmfit.add('kappa', value=k, vary=False)

    if fit_switch[len(delta)+5] == 1:
        params_lmfit.add('E0', value=E0, min=0.95*E0, max=1.05*E0)
    else:
        params_lmfit.add('E0', value=E0, vary=False)

    print("Initial parameters: " + str(params_lmfit))
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


# intializing software, load all input files into global variables
load("freqs.dat", "deltas.dat", "rpumps.dat",
     "inp.txt", "abs_exp.dat", "profs_exp.dat")


class WorkerSignals(QObject):
    result_ready = pyqtSignal(str)


class Worker(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        # global delta, M, gamma, maxnfev, tolerance, fit_alg
        params_lmfit = param_init()
        print("Initial deltas: "+str(delta))
        print("Fit is running, please wait...\n")
        # raman_residual(params_lmfit)
        # notice = QMessageBox()
        # notice.setText("Fit is running, please wait...")
        # notice.setStandardButtons(QMessageBox.NoButton) #No button needed
        # notice.exec_()
        fit_kws = dict(tol=tolerance)
        try:
            result = lmfit.minimize(raman_residual, params_lmfit, method=fit_alg,
                                    **fit_kws, max_nfev=maxnfev)  # max_nfev = 10000000, **fit_kws
        except:
            print(
                "Something went wrong before fitting start. Use powell algorithm instead")
            result = lmfit.minimize(
                raman_residual, params_lmfit, method="powell", **fit_kws, max_nfev=maxnfev)
        print(lmfit.fit_report(result))
        run_save()
        print("Fit done\n")
        self.signals.result_ready.emit("Fit done")


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
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" %
              self.threadpool.maxThreadCount())

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

        for row in range(len(delta)):
            label = QTableWidgetItem("delta@"+str(wg[row])+" cm-1")
            self.table_widget.setItem(
                row, 1, QTableWidgetItem(str(delta[row])))
            self.table_widget.setItem(row, 0, label)
        self.table_widget.setItem(len(delta), 0, QTableWidgetItem("gamma"))
        self.table_widget.setItem(len(delta), 1, QTableWidgetItem(str(gamma)))
        self.table_widget.setItem(
            len(delta)+1, 0, QTableWidgetItem("Transition Length"))
        self.table_widget.setItem(len(delta)+1, 1, QTableWidgetItem(str(M)))
        self.table_widget.setItem(len(delta)+2, 0, QTableWidgetItem("theta"))
        self.table_widget.setItem(
            len(delta)+2, 1, QTableWidgetItem(str(theta)))
        self.table_widget.setItem(len(delta)+3, 0, QTableWidgetItem("kappa"))
        self.table_widget.setItem(len(delta)+3, 1, QTableWidgetItem(str(k)))
        self.table_widget.setItem(
            len(delta)+4, 0, QTableWidgetItem("Refractive Index"))
        self.table_widget.setItem(len(delta)+4, 1, QTableWidgetItem(str(n)))
        self.table_widget.setItem(len(delta)+5, 0, QTableWidgetItem("E00"))
        self.table_widget.setItem(len(delta)+5, 1, QTableWidgetItem(str(E0)))
        self.table_widget.setItem(
            len(delta)+6, 0, QTableWidgetItem("Time step (ps)"))
        self.table_widget.setItem(len(delta)+6, 1, QTableWidgetItem(str(ts)))
        self.table_widget.setItem(
            len(delta)+7, 0, QTableWidgetItem("Time step number"))
        self.table_widget.setItem(
            len(delta)+7, 1, QTableWidgetItem(str(ntime)))
        self.table_widget.setItem(
            len(delta)+17, 0, QTableWidgetItem("Temp (K)"))
        self.table_widget.setItem(len(delta)+17, 1, QTableWidgetItem(str(T)))
        self.table_widget.setItem(
            len(delta)+18, 0, QTableWidgetItem("Raman maxcalc"))
        self.table_widget.setItem(
            len(delta)+18, 1, QTableWidgetItem(str(raman_maxcalc)))
        self.table_widget.itemChanged.connect(self.update_data)
        self.update_spectrum()

    def load_table(self):
        global fit_switch, k, raman_maxcalc, T, fit_alg, ts, ntime, E0, n, theta, gamma, D, L, M, delta, raman_xmin, raman_xmax, profs_xmin, profs_xmax, abs_xmin, abs_xmax, maxnfev, tolerance, plot_switch
        plot_switch = np.ones(len(delta)+18)
        fit_switch = np.ones(len(delta)+18)
        for i in range(len(delta)):
            delta[i] = float(self.table_widget.item(i, 1).text())
            plot_switch[i] = int(float(self.table_widget.item(i, 2).text()))
            fit_switch[i] = int(float(self.table_widget.item(i, 3).text()))

        gamma = float(self.table_widget.item(len(delta), 1).text())
        fit_switch[len(delta)] = int(
            float(self.table_widget.item(len(delta), 3).text()))
        M = float(self.table_widget.item(len(delta)+1, 1).text())
        fit_switch[len(
            delta)+1] = int(float(self.table_widget.item(len(delta)+1, 3).text()))
        theta = float(self.table_widget.item(
            len(delta)+2, 1).text())  # theta parameter
        fit_switch[len(
            delta)+2] = int(float(self.table_widget.item(len(delta)+2, 3).text()))
        k = float(self.table_widget.item(
            len(delta)+3, 1).text())  # kappa parameter
        fit_switch[len(
            delta)+3] = int(float(self.table_widget.item(len(delta)+3, 3).text()))
        n = float(self.table_widget.item(
            len(delta)+4, 1).text())  # refractive index
        E0 = float(self.table_widget.item(
            len(delta)+5, 1).text())  # E00 parameter
        fit_switch[len(
            delta)+5] = int(float(self.table_widget.item(len(delta)+5, 3).text()))
        ts = float(self.table_widget.item(len(delta)+6, 1).text())
        ntime = float(self.table_widget.item(len(delta)+7, 1).text())

        fit_alg = self.table_widget.item(
            len(delta)+8, 1).text()  # fitting algorithm
        maxnfev = int(self.table_widget.item(
            len(delta)+9, 1).text())  # max fitting steps
        tolerance = float(self.table_widget.item(
            len(delta)+10, 1).text())  # fitting tolerance
        raman_xmin = float(self.table_widget.item(len(delta)+11, 1).text())
        raman_xmax = float(self.table_widget.item(len(delta)+12, 1).text())
        profs_xmin = float(self.table_widget.item(len(delta)+13, 1).text())
        profs_xmax = float(self.table_widget.item(len(delta)+14, 1).text())
        abs_xmin = float(self.table_widget.item(len(delta)+15, 1).text())
        abs_xmax = float(self.table_widget.item(len(delta)+16, 1).text())
        T = float(self.table_widget.item(len(delta)+17, 1).text())
        raman_maxcalc = float(self.table_widget.item(len(delta)+18, 1).text())

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
        worker = Worker()  # thread for fitting
        worker.signals.result_ready.connect(self.update_fit)
        self.fit_button.setEnabled(False)
        recalc()
        self.threadpool.start(worker)
        # after fitting, global variables should be optimized values
        # self.plot_data()

    def update_fit(self, result):
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
        recalc()
        abs_cross, fl_cross, raman_cross, boltz_states, boltz_coef = cross_sections(
            convEL, delta, theta, D, L, M, E0)
        raman_spec = np.zeros((len(rshift), len(rpumps)))
        for i in range(len(rpumps)):
            # rp = min(range(len(convEL)),key=lambda j:abs(convEL[j]-rpumps[i]))
            min_diff = float('inf')
            min_index = None

            for j in range(len(convEL)):
                diff = np.absolute(convEL[j] - rpumps[i])
                if diff < min_diff:
                    min_diff = diff
                    min_index = j

            rp = min_index
            # print(rp)
            # print(rpumps)
            for l in np.arange(len(wg)):
                raman_spec[:, i] += np.real((raman_cross[l, rp])) * \
                    (1/np.pi)*(0.5*res)/((rshift-wg[l])**2+(0.5*res)**2)
            nm = 1e7/rpumps[i]
            self.ax2.plot(rshift, np.real((raman_spec)[:, i]), label=f'{nm:3f} nm laser')  # plot raman spectrum
        self.ax2.set_title('Raman Spectra')
        self.ax2.set_xlim(raman_xmin, raman_xmax)
        self.ax2.set_xlabel('Raman Shift (cm-1)')
        self.ax2.set_ylabel(
            'Raman Cross Section \n(1e-14 Angstrom**2/Molecule)')
        self.ax2.legend(loc='best')
        self.canvas2.draw()

        # Plot Raman excitation profiles
        for i in range(len(rpumps)):  # iterate over pump wn
            min_diff = float('inf')
            rp = None

            # iterate over all exitation wn to find the one closest to pump
            for j in range(len(convEL)):
                diff = np.absolute(convEL[j] - rpumps[i])
                if diff < min_diff:
                    min_diff = diff
                    rp = j
                # print(rp)

            for j in range(len(wg)):  # iterate over all raman freqs
                # print(j,i)
                # sigma[j] = sigma[j] + (1e8*(np.real(raman_cross[j,rp])-rcross_exp[j,i]))**2
                if plot_switch[j] == 1:
                    color = cmap(j)
                    self.ax.plot(convEL[rp], profs_exp[j, i], "o", color=color)
        for j in range(len(wg)):  # iterate over all raman freqs
            if plot_switch[j] == 1:
                color = cmap(j)
                self.ax.plot(convEL, np.real(np.transpose(raman_cross))[
                             :, j], color=color, label=f'{wg[j]:2f} cm-1')
        self.ax.set_title('Raman Excitation Profiles')
        self.ax.set_xlim(profs_xmin, profs_xmax)
        self.ax.set_xlabel('Wavenumber (cm-1)')
        self.ax.set_ylabel(
            'Raman Cross Section \n(1e-14 Angstrom**2/Molecule)')
        self.ax.legend(loc='best', ncol=2, prop={'size': 8})
        self.canvas.draw()

        # plot absorption
        self.ax3.plot(convEL, np.real(abs_cross), label='Abs')
        self.ax3.plot(convEL, np.real(fl_cross), label='FL')
        try:
            self.ax3.plot(convEL, abs_exp[:, 1], label='Abs expt.')
        except:
            print("No experimental absorption spectrum")
        self.ax3.set_title('Absorption and Emission Spectra')
        self.ax3.set_xlim(abs_xmin, abs_xmax)
        self.ax3.set_xlabel('Wavenumber (cm-1)')
        self.ax3.set_ylabel('Cross Section \n(1e-14 Angstrom**2/Molecule)')
        self.ax3.legend(loc='best')
        self.canvas3.draw()

    def create_variable_table(self):
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(4)
        self.table_widget.setRowCount(len(delta)+19)
        self.table_widget.setHorizontalHeaderLabels(
            ["Variables", "Values", "Plot Raman \nEx. Profile", "Fit?"])
        for row in range(len(delta)):
            item = QTableWidgetItem(str(delta[row]))
            label = QTableWidgetItem("delta@"+str(wg[row])+" cm-1")
            self.table_widget.setItem(row, 0, label)
            self.table_widget.setItem(row, 1, item)
            self.table_widget.setItem(row, 2, QTableWidgetItem("1"))
            self.table_widget.setItem(row, 3, QTableWidgetItem("1"))
        self.table_widget.setItem(len(delta), 0, QTableWidgetItem("gamma"))
        self.table_widget.setItem(len(delta), 1, QTableWidgetItem(str(inp[0])))
        self.table_widget.setItem(len(delta), 3, QTableWidgetItem("1"))
        self.table_widget.setItem(
            len(delta)+1, 0, QTableWidgetItem("Transition Length (A)"))
        self.table_widget.setItem(
            len(delta)+1, 1, QTableWidgetItem(str(inp[7])))
        self.table_widget.setItem(len(delta)+1, 3, QTableWidgetItem("1"))
        self.table_widget.setItem(len(delta)+2, 0, QTableWidgetItem("theta"))
        self.table_widget.setItem(
            len(delta)+2, 1, QTableWidgetItem(str(inp[1])))
        self.table_widget.setItem(len(delta)+2, 3, QTableWidgetItem("0"))
        self.table_widget.setItem(len(delta)+3, 0, QTableWidgetItem("kappa"))
        self.table_widget.setItem(
            len(delta)+3, 1, QTableWidgetItem(str(inp[3])))
        self.table_widget.setItem(len(delta)+3, 3, QTableWidgetItem("1"))
        self.table_widget.setItem(
            len(delta)+4, 0, QTableWidgetItem("Refractive Index"))
        self.table_widget.setItem(
            len(delta)+4, 1, QTableWidgetItem(str(inp[8])))
        self.table_widget.setItem(len(delta)+4, 3, QTableWidgetItem("0"))
        self.table_widget.setItem(len(delta)+5, 0, QTableWidgetItem("E00"))
        self.table_widget.setItem(
            len(delta)+5, 1, QTableWidgetItem(str(inp[2])))
        self.table_widget.setItem(len(delta)+5, 3, QTableWidgetItem("1"))
        self.table_widget.setItem(
            len(delta)+6, 0, QTableWidgetItem("Time step (ps)"))
        self.table_widget.setItem(
            len(delta)+6, 1, QTableWidgetItem(str(inp[4])))
        self.table_widget.setItem(
            len(delta)+7, 0, QTableWidgetItem("Time step number"))
        self.table_widget.setItem(
            len(delta)+7, 1, QTableWidgetItem(str(inp[5])))
        self.table_widget.setItem(
            len(delta)+8, 0, QTableWidgetItem("Fitting lgorithm"))
        self.table_widget.setItem(len(delta)+8, 1, QTableWidgetItem("powell"))
        self.table_widget.setItem(
            len(delta)+9, 0, QTableWidgetItem("Fitting maxnfev"))
        self.table_widget.setItem(len(delta)+9, 1, QTableWidgetItem("100"))
        self.table_widget.setItem(
            len(delta)+10, 0, QTableWidgetItem("Fitting tolerance"))
        self.table_widget.setItem(
            len(delta)+10, 1, QTableWidgetItem("0.00000001"))
        self.table_widget.setItem(
            len(delta)+11, 0, QTableWidgetItem("Raman xmin"))
        self.table_widget.setItem(len(delta)+11, 1, QTableWidgetItem("200"))
        self.table_widget.setItem(
            len(delta)+12, 0, QTableWidgetItem("Raman xmax"))
        self.table_widget.setItem(len(delta)+12, 1, QTableWidgetItem("1700"))
        self.table_widget.setItem(
            len(delta)+13, 0, QTableWidgetItem("Exci. Prof. xmin"))
        self.table_widget.setItem(len(delta)+13, 1, QTableWidgetItem("15000"))
        self.table_widget.setItem(
            len(delta)+14, 0, QTableWidgetItem("Exci. Prof. xmax"))
        self.table_widget.setItem(len(delta)+14, 1, QTableWidgetItem("22500"))
        self.table_widget.setItem(
            len(delta)+15, 0, QTableWidgetItem("Abs/FL xmin"))
        self.table_widget.setItem(len(delta)+15, 1, QTableWidgetItem("12000"))
        self.table_widget.setItem(
            len(delta)+16, 0, QTableWidgetItem("Abs/Fl xmax"))
        self.table_widget.setItem(len(delta)+16, 1, QTableWidgetItem("23000"))
        self.table_widget.setItem(
            len(delta)+17, 0, QTableWidgetItem("Temp (K)"))
        self.table_widget.setItem(
            len(delta)+17, 1, QTableWidgetItem(str(inp[13])))
        self.table_widget.setItem(
            len(delta)+18, 0, QTableWidgetItem("Raman maxcalc"))
        self.table_widget.setItem(
            len(delta)+18, 1, QTableWidgetItem(str(inp[10])))
        self.table_widget.itemChanged.connect(self.update_data)
        print("Initialized. Files loaded from the working folder.")
        # Set headers to resize to contents
        self.table_widget.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self.table_widget.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents)

        self.right_layout.addWidget(self.table_widget)

    def select_subfolder(self):
        global dir
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly | QFileDialog.ReadOnly

        self.folder_path = QFileDialog.getExistingDirectory(
            self, "Select Subfolder", os.getcwd(), options=options)

        if self.folder_path:
            print("Selected folder:", self.folder_path)
            dir = self.folder_path+"/"
            load("freqs.dat", "deltas.dat", "rpumps.dat",
                 "inp.txt", "abs_exp.dat", "profs_exp.dat")
            self.sendto_table()
            self.dirlabel.setText("Current data folder: "+dir)
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
        self.dirlabel = QLabel("Current data folder:/ "+dir)
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

    def add_data(self):
        # Add a new row to the table
        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)
        self.table_widget.setItem(
            row_position, 0, QTableWidgetItem("New Data"))
        self.table_widget.setItem(row_position, 1, QTableWidgetItem("0.0"))
        self.update_spectrum()

    def save_data(self):
        run_save()

    def initialize(self):
        global delta, wg, inp, dir
        self.table_widget.itemChanged.disconnect(self.update_data)
        dir = ''
        load("freqs.dat", "deltas.dat", "rpumps.dat",
             "inp.txt", "abs_exp.dat", "profs_exp.dat")
        for row in range(len(delta)):
            item = QTableWidgetItem(str(delta[row]))
            label = QTableWidgetItem("delta@"+str(wg[row])+" cm-1")
            self.table_widget.setItem(row, 0, label)
            self.table_widget.setItem(row, 1, item)
            self.table_widget.setItem(row, 2, QTableWidgetItem("1"))
            self.table_widget.setItem(row, 3, QTableWidgetItem("1"))
        self.table_widget.setItem(len(delta), 0, QTableWidgetItem("gamma"))
        self.table_widget.setItem(len(delta), 1, QTableWidgetItem(str(inp[0])))
        self.table_widget.setItem(len(delta), 3, QTableWidgetItem("1"))
        self.table_widget.setItem(
            len(delta)+1, 0, QTableWidgetItem("Transition Length (A)"))
        self.table_widget.setItem(
            len(delta)+1, 1, QTableWidgetItem(str(inp[7])))
        self.table_widget.setItem(len(delta)+1, 3, QTableWidgetItem("1"))
        self.table_widget.setItem(len(delta)+2, 0, QTableWidgetItem("theta"))
        self.table_widget.setItem(
            len(delta)+2, 1, QTableWidgetItem(str(inp[1])))
        self.table_widget.setItem(len(delta)+2, 3, QTableWidgetItem("1"))
        self.table_widget.setItem(len(delta)+3, 0, QTableWidgetItem("kappa"))
        self.table_widget.setItem(
            len(delta)+3, 1, QTableWidgetItem(str(inp[3])))
        self.table_widget.setItem(len(delta)+3, 3, QTableWidgetItem("0"))
        self.table_widget.setItem(
            len(delta)+4, 0, QTableWidgetItem("Refractive Index"))
        self.table_widget.setItem(
            len(delta)+4, 1, QTableWidgetItem(str(inp[8])))
        self.table_widget.setItem(len(delta)+4, 3, QTableWidgetItem("0"))
        self.table_widget.setItem(len(delta)+5, 0, QTableWidgetItem("E00"))
        self.table_widget.setItem(
            len(delta)+5, 1, QTableWidgetItem(str(inp[2])))
        self.table_widget.setItem(len(delta)+5, 3, QTableWidgetItem("0"))
        self.table_widget.setItem(
            len(delta)+6, 0, QTableWidgetItem("Time step (ps)"))
        self.table_widget.setItem(
            len(delta)+6, 1, QTableWidgetItem(str(inp[4])))
        self.table_widget.setItem(
            len(delta)+7, 0, QTableWidgetItem("Time step number"))
        self.table_widget.setItem(
            len(delta)+7, 1, QTableWidgetItem(str(inp[5])))
        self.table_widget.setItem(
            len(delta)+8, 0, QTableWidgetItem("Fitting lgorithm"))
        self.table_widget.setItem(len(delta)+8, 1, QTableWidgetItem("powell"))
        self.table_widget.setItem(
            len(delta)+9, 0, QTableWidgetItem("Fitting maxnfev"))
        self.table_widget.setItem(len(delta)+9, 1, QTableWidgetItem("100"))
        self.table_widget.setItem(
            len(delta)+10, 0, QTableWidgetItem("Fitting tolerance"))
        self.table_widget.setItem(
            len(delta)+10, 1, QTableWidgetItem("0.00000001"))
        self.table_widget.setItem(
            len(delta)+11, 0, QTableWidgetItem("Raman xmin"))
        self.table_widget.setItem(len(delta)+11, 1, QTableWidgetItem("200"))
        self.table_widget.setItem(
            len(delta)+12, 0, QTableWidgetItem("Raman xmax"))
        self.table_widget.setItem(len(delta)+12, 1, QTableWidgetItem("1700"))
        self.table_widget.setItem(
            len(delta)+13, 0, QTableWidgetItem("Exci. Prof. xmin"))
        self.table_widget.setItem(len(delta)+13, 1, QTableWidgetItem("15000"))
        self.table_widget.setItem(
            len(delta)+14, 0, QTableWidgetItem("Exci. Prof. xmax"))
        self.table_widget.setItem(len(delta)+14, 1, QTableWidgetItem("22500"))
        self.table_widget.setItem(
            len(delta)+15, 0, QTableWidgetItem("Abs/FL xmin"))
        self.table_widget.setItem(len(delta)+15, 1, QTableWidgetItem("12000"))
        self.table_widget.setItem(
            len(delta)+16, 0, QTableWidgetItem("Abs/Fl xmax"))
        self.table_widget.setItem(len(delta)+16, 1, QTableWidgetItem("23000"))
        self.table_widget.setItem(
            len(delta)+17, 0, QTableWidgetItem("Temp (K)"))
        self.table_widget.setItem(
            len(delta)+17, 1, QTableWidgetItem(str(inp[13])))
        self.table_widget.setItem(
            len(delta)+18, 0, QTableWidgetItem("Raman maxcalc"))
        self.table_widget.setItem(
            len(delta)+18, 1, QTableWidgetItem(str(inp[10])))
        print("Initialized. Files loaded from the working folder.")
        self.table_widget.itemChanged.connect(self.update_data)
        self.update_spectrum()
        self.dirlabel.setText("Current data folder: /"+dir)

    def update_data(self):
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


'''
    def delete_data(self):
        # Delete the selected rows from the table
        selected_rows = self.table_widget.selectionModel().selectedRows()
        for index in selected_rows:
            self.table_widget.removeRow(index.row())
        self.update_spectrum()
'''


class OutputWidget(QTextBrowser):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)  # Make the text browser read-only

    def write(self, text):
        # Append the text to the output panel
        self.insertPlainText(text)
        self.ensureCursorVisible()  # Scroll to the latest text


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('ico.ico'))
    window = SpectrumApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
