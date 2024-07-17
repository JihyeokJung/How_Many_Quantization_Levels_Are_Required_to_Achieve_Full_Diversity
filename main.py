import numpy as np
import matplotlib.pyplot as plt
import os

from SystemModel import outage_probability
from reproduce import outage_prob_plot

'''********************* hyperparameter  *********************'''
Omega_S = 1  # Variance of h_SI (channel for Source - RIS)
Omega_I = 0.5  # Variance of h_ID (channel for RIS-Destination)
P = 1  # Transmit power
eta = 0.8  # Amplitude reflection coefficient
delta = 0.1  # Variance of noise
R0 = 1  # data rate in bits per channel use(BPCU)
epsilon_0 = (2 ** R0 - 1) / (eta ** 2 * Omega_S * Omega_I)

# rho values to infinity
rho_values = np.logspace(0, 4.5, 10)
narrow_rho_values = np.logspace(0, 2.25, 10)


def outage_prob_simulation():
    """ When L = 3"""
    L = 3

    P_out_values_3_2 = np.array(
        [outage_probability(L, 2, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
    P_out_values_3_4 = np.array(
        [outage_probability(L, 4, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
    # N = 8 이상 일 때는 outage  prob이 15dB 근처에서 0에 수렴함. -> rho values 세분화
    P_out_values_3_8 = np.array(
        [outage_probability(L, 8, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in np.logspace(0, 2.25, 10)])

    """When L = 2"""
    L = 2

    P_out_values_2_2 = np.array(
        [outage_probability(L, 2, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
    P_out_values_2_4 = np.array(
        [outage_probability(L, 4, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
    P_out_values_2_8 = np.array(
        [outage_probability(L, 8, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in np.logspace(0, 2.25, 10)])


    # curve 저장
    np.save(f"./Curves/outage_prob_{3}_{2}", np.array([rho_values,P_out_values_3_2]))
    np.save(f"./Curves/outage_prob_{3}_{4}", np.array([rho_values, P_out_values_3_4]))
    np.save(f"./Curves/outage_prob_{3}_{8}", np.array([narrow_rho_values, P_out_values_3_8]))

    np.save(f"./Curves/outage_prob_{2}_{2}", np.array([rho_values, P_out_values_2_2]))
    np.save(f"./Curves/outage_prob_{2}_{4}", np.array([rho_values, P_out_values_2_4]))
    np.save(f"./Curves/outage_prob_{2}_{8}", np.array([narrow_rho_values, P_out_values_2_8]))


outage_prob_simulation()
outage_prob_plot()
'''
# print(narrow_rho_values, 10*np.log10(narrow_rho_values))

# print(outage_probability(2, 8, P, delta, eta, Omega_S, Omega_I, epsilon_0,  31.6227766))
  L = 2, N  = 8, Transmit SNR(rho) = 15dB (linear scale : 31.6227766)
  --> when iter 2e+7, result is 0.0
  --> when iter 5e+7, result is 4e-8
'''