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

# fig_num = 2
# outage_prob_plot(fig_num)



rho_values = np.logspace(0, 4.5, 10)

opt_sc_P_out_values_2 = np.array(
    [opt_sc_outage_probability(4, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])

plt.figure(figsize=(8, 8))
plt.title('Outage Probability vs. Transmit SNR')

P_out_values_3_2 = np.array(
    [outage_probability(3, 4, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])

plt.plot(10 * np.log10(rho_values), opt_sc_P_out_values_2, 'rs-', label=f'Optimal Outage Probability')
plt.plot(10 * np.log10(rho_values), P_out_values_3_2, 'bo-', label=f'Outage Probability with L=3')

plt.yscale('log', base=10)
plt.yticks([10 ** -8, 10 ** -6, 10 ** -4, 10 ** -2, 10 ** 0])
plt.xlim(0, 45)
plt.ylim(10 ** -8, 10 ** 0)
plt.xlabel('Transmit SNR(dB)')
plt.ylabel('Outage Probability')
plt.legend()
plt.grid(True)
plt.show()

