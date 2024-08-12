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

opt_fc_P_out_values_4 = np.array(
    [opt_fc_outage_probability(4, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
opt_P_out_values_4 = np.load(f"./Curves/SC_opt_outage_prob_4.npy")

plt.figure(figsize=(8, 8))
plt.title('Outage Probability vs. Transmit SNR')

plt.plot(10 * np.log10(rho_values), opt_fc_P_out_values_4, 'rs-', label=f'Fully-connected Outage Probability with N=4')
plt.plot(10 * np.log10(rho_values), opt_P_out_values_4[1], 'b--', label=f'Single-connected Outage Probability')

plt.yscale('log', base=10)
plt.yticks([10 ** -8, 10 ** -6, 10 ** -4, 10 ** -2, 10 ** 0])
plt.xlim(0, 45)
plt.ylim(10 ** -8, 10 ** 0)
plt.xlabel('Transmit SNR(dB)')
plt.ylabel('Outage Probability')
plt.legend()
plt.grid(True)
plt.show()

