import numpy as np
import matplotlib.pyplot as plt
import os

from SystemModel import *
from reproduce import *
from simulations import *

'''********************* Single-Connected  *********************'''
'''********************** hyperparameter  **********************'''
Omega_S = 1  # Variance of h_SI (channel for Source - RIS)
Omega_I = 0.5  # Variance of h_ID (channel for RIS-Destination)
P = 1  # Transmit power
eta = 0.8  # Amplitude reflection coefficient
delta = 0.1  # Variance of noise

R0 = 1  # data rate in bits per channel use(BPCU)
epsilon_0 = (2 ** R0 - 1) / (eta ** 2 * Omega_S * Omega_I)


'''**************************** Run  ***************************'''
# outage_prob_simulation(Omega_S, Omega_I, P, eta, delta, epsilon_0)

# opt_sc_outage_prob_simulation(Omega_S, Omega_I, P, eta, delta, epsilon_0)
# fig_num = 2
# outage_prob_plot(fig_num)


rho_values = np.logspace(0, 4.5, 10)
narrow_rho_values = np.logspace(0, 2.25, 10)

P_out_values_3_2 = np.array(
    [outage_probability(4, 2, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
P_out_values_3_4 = np.array(
    [outage_probability(4, 4, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
# N = 8 이상 일 때는 outage  prob이 15dB 근처에서 0에 수렴함. -> rho values 세분화
P_out_values_3_8 = np.array(
    [outage_probability(4, 8, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in np.logspace(0, 2.25, 10)])

P_out_values_2_2 = np.array(
    [outage_probability(8, 2, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
P_out_values_2_4 = np.array(
    [outage_probability(8, 4, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
P_out_values_2_8 = np.array(
    [outage_probability(8, 8, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in np.logspace(0, 2.25, 10)])


opt_P_out_values_2 = np.array(
    [opt_outage_probability(2, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
opt_P_out_values_4 = np.array(
    [opt_outage_probability(4, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
# N = 8 이상 일 때는 outage  prob이 15dB 근처에서 0에 수렴함. -> rho values 세분화
opt_P_out_values_8 = np.array(
    [opt_outage_probability(8, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in narrow_rho_values])


plt.figure(figsize=(8, 8))
plt.title('Outage Probability vs. Transmit SNR')

plt.plot(10 * np.log10(rho_values), P_out_values_3_2, 'r^--',
         label=f'L=3')
plt.plot(10 * np.log10(rho_values), P_out_values_3_4, 'r>--')
plt.plot(10 * np.log10(narrow_rho_values), P_out_values_3_8, 'r<--')

plt.plot(10 * np.log10(rho_values), P_out_values_2_2, 'b^--',
         label=f'L=2')
plt.plot(10 * np.log10(rho_values), P_out_values_2_4, 'b>--')
plt.plot(10 * np.log10(narrow_rho_values), P_out_values_2_8, 'b<--')

plt.plot(10 * np.log10(rho_values), opt_P_out_values_2, 'y^--',
         label=f'opt')
plt.plot(10 * np.log10(rho_values), opt_P_out_values_4, 'y>--')
plt.plot(10 * np.log10(narrow_rho_values), opt_P_out_values_8, 'y<--')

plt.yscale('log', base=10)
plt.yticks([10 ** -8, 10 ** -6, 10 ** -4, 10 ** -2, 10 ** 0])
plt.xlim(0, 45)
plt.ylim(10 ** -8, 10 ** 0)
plt.xlabel('Transmit SNR(dB)')
plt.ylabel('Outage Probability')
plt.legend()
plt.grid(True)

# save the figure
fig_dir = f"./Figures"

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

save_name = f"outage_Prob_custom.jpg"
fig_dir = f"{fig_dir}/{save_name}"

plt.savefig(f"{fig_dir}", bbox_inchesbbox_inches='tight')

plt.show()
