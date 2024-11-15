import numpy as np
import os

from SystemModel import *


def outage_prob_simulation(Omega_S, Omega_I, P, delta, epsilon_0):
    # rho values to infinity
    rho_values = np.logspace(0, 4.5, 10)
    narrow_rho_values = np.logspace(0, 2.25, 10)

    """ When L = 3"""
    L = 3

    P_out_values_3_2 = np.array(
        [outage_probability(L, 2, P, delta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])

    P_out_values_3_4 = np.array(
        [outage_probability(L, 4, P, delta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
    # N = 8 이상 일 때는 outage  prob이 15dB 근처에서 0에 수렴함. -> rho values 세분화
    P_out_values_3_8 = np.array(
        [outage_probability(L, 8, P, delta, Omega_S, Omega_I, epsilon_0, rho) for rho in np.logspace(0, 2.25, 10)])

    """When L = 2"""
    L = 2

    P_out_values_2_2 = np.array(
        [outage_probability(L, 2, P, delta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
    P_out_values_2_4 = np.array(
        [outage_probability(L, 4, P, delta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
    P_out_values_2_8 = np.array(
        [outage_probability(L, 8, P, delta, Omega_S, Omega_I, epsilon_0, rho) for rho in np.logspace(0, 2.25, 10)])

    # curve 저장
    np.save(f"./Curves/outage_prob_{3}_{2}", np.array([rho_values, P_out_values_3_2]))
    np.save(f"./Curves/outage_prob_{3}_{4}", np.array([rho_values, P_out_values_3_4]))
    np.save(f"./Curves/outage_prob_{3}_{8}", np.array([narrow_rho_values, P_out_values_3_8]))

    np.save(f"./Curves/outage_prob_{2}_{2}", np.array([rho_values, P_out_values_2_2]))
    np.save(f"./Curves/outage_prob_{2}_{4}", np.array([rho_values, P_out_values_2_4]))
    np.save(f"./Curves/outage_prob_{2}_{8}", np.array([narrow_rho_values, P_out_values_2_8]))


def opt_outage_prob_simulation(Omega_S, Omega_I, P, eta, delta, epsilon_0):
    # rho values to infinity
    rho_values = np.logspace(0, 4.5, 10)
    narrow_rho_values = np.logspace(0, 2.25, 10)

    opt_P_out_values_2 = np.array(
        [opt_outage_probability(2, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
    opt_P_out_values_4 = np.array(
        [opt_outage_probability(4, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in rho_values])
    # N = 8 이상 일 때는 outage  prob이 15dB 근처에서 0에 수렴함. -> rho values 세분화
    opt_P_out_values_8 = np.array(
        [opt_outage_probability(8, P, delta, eta, Omega_S, Omega_I, epsilon_0, rho) for rho in narrow_rho_values])

    # curve 저장
    np.save(f"./Curves/SC_opt_outage_prob_{2}", np.array([rho_values, opt_P_out_values_2]))
    np.save(f"./Curves/SC_opt_outage_prob_{4}", np.array([rho_values, opt_P_out_values_4]))
    np.save(f"./Curves/SC_opt_outage_prob_{8}", np.array([narrow_rho_values, opt_P_out_values_8]))





