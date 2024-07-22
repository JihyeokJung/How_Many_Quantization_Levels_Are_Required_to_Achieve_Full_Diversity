import numpy as np
import matplotlib.pyplot as plt
import os


def outage_prob_plot(fig_num):
    if fig_num == 1:
        fig_name = f"The outage probability versus the SNR_5e7"

        P_out_values_3_2 = np.load(f"./Curves/outage_prob_3_2.npy")
        P_out_values_3_4 = np.load(f"./Curves/outage_prob_3_4.npy")
        P_out_values_3_8 = np.load(f"./Curves/outage_prob_3_8.npy")

        P_out_values_2_2 = np.load(f"./Curves/outage_prob_2_2.npy")
        P_out_values_2_4 = np.load(f"./Curves/outage_prob_2_4.npy")
        P_out_values_2_8 = np.load(f"./Curves/outage_prob_2_8.npy")

        plt.figure(figsize=(8, 8))
        plt.title('Outage Probability vs. Transmit SNR')

        plt.plot(10 * np.log10(P_out_values_3_2[0]), P_out_values_3_2[1], 'rs-', label=f'Outage Probability with L=3, N=2')
        plt.plot(10 * np.log10(P_out_values_3_4[0]), P_out_values_3_4[1], 'rs-')
        plt.plot(10 * np.log10(P_out_values_3_8[0]), P_out_values_3_8[1], 'rs-')

        plt.plot(10 * np.log10(P_out_values_2_2[0]), P_out_values_2_2[1], 'bo-', label=f'Outage Probability with L=2, N=2')
        plt.plot(10 * np.log10(P_out_values_2_4[0]), P_out_values_2_4[1], 'bo-')
        plt.plot(10 * np.log10(P_out_values_2_8[0]), P_out_values_2_8[1], 'bo-')

        plt.yscale('log', base=10)
        plt.yticks([10**-8, 10**-6, 10**-4, 10**-2, 10**0])
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

        save_name = f"{fig_name}.jpg"
        fig_dir = f"{fig_dir}/{save_name}"

        plt.savefig(f"{fig_dir}", bbox_inches='tight')
        plt.show()
    elif fig_num == 2:
        fig_name = "The outage probability versus the SNR_5e7"

        P_out_values_2_2 = np.load(f"./Curves/SC_opt_outage_prob_2.npy")
        P_out_values_2_4 = np.load(f"./Curves/SC_opt_outage_prob_4.npy")
        P_out_values_2_8 = np.load(f"./Curves/SC_opt_outage_prob_8.npy")

        plt.figure(figsize=(8, 8))
        plt.title('Outage Probability vs. Transmit SNR')

        plt.plot(10 * np.log10(P_out_values_3_2[0]), P_out_values_3_2[1], 'm>-',
                 label=f'Optimal continuous phase-shift')
        plt.plot(10 * np.log10(P_out_values_3_4[0]), P_out_values_3_4[1], 'm>-')
        plt.plot(10 * np.log10(P_out_values_3_8[0]), P_out_values_3_8[1], 'm>-')


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

        save_name = f"{fig_name}.jpg"
        fig_dir = f"{fig_dir}/{save_name}"

        plt.savefig(f"{fig_dir}", bbox_inches='tight')
        plt.show()
