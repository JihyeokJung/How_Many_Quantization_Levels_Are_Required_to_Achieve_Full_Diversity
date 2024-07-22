import numpy as np
import matplotlib.pyplot as plt
import os

from SystemModel import *
from reproduce import *
from simulations import *

'''********************* hyperparameter  *********************'''
Omega_S = 1  # Variance of h_SI (channel for Source - RIS)
Omega_I = 0.5  # Variance of h_ID (channel for RIS-Destination)
P = 1  # Transmit power
eta = 0.8  # Amplitude reflection coefficient
delta = 0.1  # Variance of noise

R0 = 1  # data rate in bits per channel use(BPCU)
epsilon_0 = (2 ** R0 - 1) / (eta ** 2 * Omega_S * Omega_I)


'''********************* Run  *********************'''
# outage_prob_simulation(Omega_S, Omega_I, P, eta, delta, epsilon_0)

# opt_sc_outage_prob_simulation(Omega_S, Omega_I, P, eta, delta, epsilon_0)

fig_num = 1
outage_prob_plot(fig_num)

