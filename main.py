import numpy as np
import matplotlib.pyplot as plt
import os

from SystemModel import *
from reproduce import *
from simulations import *

Omega_S = 1  # Variance of h_SI (channel for Source - RIS)
Omega_I = 0.5  # Variance of h_ID (channel for RIS-Destination)
P = 1  # Transmit power
eta = 0.8  # Amplitude reflection coefficient
delta = 0.1  # Variance of noise

R0 = 1  # data rate in bits per channel use(BPCU)
epsilon_0 = (2 ** R0 - 1) / (eta ** 2 * Omega_S * Omega_I)


""""""""""""""""""""""""""""""""""""""" RUN """""""""""""""""""""""""""""""""""""""""
"""  outage_prob_simulation(Omega_S, Omega_I, P, eta, delta, epsilon_0)           """
"""  opt_sc_outage_prob_simulation(Omega_S, Omega_I, P, eta, delta, epsilon_0)    """
""""""""""""""""""""""""""""""""""""""" RUN """""""""""""""""""""""""""""""""""""""""


""""""""""""""""""""""""""""""""""""""" FIG """""""""""""""""""""""""""""""""""""""""
""" fig_num = 1 : The outage probability versus the SNR with quantized RIS        """
""" fig_num = 2 : The outage probability versus the SNR with optimal RIS          """
""" fig_num = 3 : 1 + 2                                                           """
"""                                                                               """
""" outage_prob_plot(fig_num)                                                     """
""""""""""""""""""""""""""""""""""""""" FIG """""""""""""""""""""""""""""""""""""""""
outage_prob_plot(2)

rho_values = np.logspace(0, 4.5, 10)
narrow_rho_values = np.logspace(0, 2.25, 10)
