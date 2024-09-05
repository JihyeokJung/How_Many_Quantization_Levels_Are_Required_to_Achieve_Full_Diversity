import numpy as np
from scipy.special import kn
from tqdm import tqdm, trange

from scipy.linalg import svd


def init_system(N, Omega_S, Omega_I, delta):
    # Channel vectors : follwing complex gaussian distribution with 0 mean, Omega variance
    h_SI = (np.random.normal(loc=0, scale=np.sqrt(Omega_S / 2), size=N) + 1j * np.random.normal(loc=0, scale=np.sqrt(
        Omega_S / 2), size=N))
    h_ID = (np.random.normal(loc=0, scale=np.sqrt(Omega_I / 2), size=N) + 1j * np.random.normal(loc=0, scale=np.sqrt(
        Omega_I / 2), size=N))

    # Transmitted signal
    x_s = np.random.randn() + 1j * np.random.randn()
    x_s /= np.abs(x_s)  # Normalize to unit power

    # Noise
    w_D = np.sqrt(delta / 2) * (np.random.randn() + 1j * np.random.randn())

    return h_SI, h_ID, x_s, w_D


# Received signal
def ReceivedSignal(P, eta, h_SI, Phi, h_ID, x_s, w_D):
    ReceivedSignal = np.sqrt(P) * eta * (h_SI.T @ Phi @ h_ID) * x_s + w_D

    return ReceivedSignal


def SNR_calc(P, delta, eta, h_SI, Phi, h_ID, Omega_S, Omega_I):
    rho = P / delta ** 2
    g_n = np.abs(h_SI * h_ID) @ Phi / np.sqrt(Omega_S * Omega_I)  # normalized channel coefficient

    ReceivedSNR = rho * eta ** 2 * Omega_S * Omega_I * np.abs(np.sum(g_n)) ** 2

    return g_n, ReceivedSNR


def get_phase_error(N, L):
    # Phase errors uniformly distributed in [-pi/L, pi/L]
    phase_errors = np.random.uniform(-np.pi / L, np.pi / L, N)

    Phi = np.diag(np.exp(1j * phase_errors))

    return phase_errors, Phi


def cdf_gn_squared(x):
    if x <= 0:
        return 0.0

    if x < 1e-6:
        return - x * np.log(x)  # When the x value is almost 0
    else:
        return 1 - 2 * np.sqrt(x) * kn(1, 2 * np.sqrt(x))


def sc_outage_probability(L, N, P, delta, eta, Omega_S, Omega_I, threshold, rho):
    prob_count = 0

    num_simulations = int(1e+5)
    pbar = tqdm(range(num_simulations))
    for _ in pbar:
        pbar.set_description(f"Outage simulation when L={L} N={N}")
        # 변수 초기화
        h_SI, h_ID, _, _ = init_system(N, Omega_S, Omega_I, delta)
        _, Phi = get_phase_error(N, L)
        g_n, _ = SNR_calc(P, delta, eta, h_SI, Phi, h_ID, Omega_S, Omega_I)

        G_N = np.sum(g_n)
        G_N_sq = np.abs(G_N) ** 2

        if G_N_sq < threshold / rho:
            prob_count += 1

    # Outage probability
    P_G_N = prob_count / num_simulations
    return P_G_N


def opt_sc_outage_probability(N, P, delta, eta, Omega_S, Omega_I, threshold, rho):
    prob_count = 0

    num_simulations = int(1e+5)
    pbar = tqdm(range(num_simulations))
    for _ in pbar:
        pbar.set_description(f"Outage simulation when N={N}")
        # 변수 초기화
        h_SI, h_ID, _, _ = init_system(N, Omega_S, Omega_I, delta)
        Phi = np.diag(np.exp(1j * np.zeros(N)))
        # Phi = np.diag(np.exp(-np.angle(h_SI * h_ID)))
        g_n, _ = SNR_calc(P, delta, eta, h_SI, Phi, h_ID, Omega_S, Omega_I)

        G_N = np.sum(g_n)
        G_N_sq = np.abs(G_N) ** 2

        if G_N_sq < threshold / rho:
            prob_count += 1

    # Outage probability
    P_G_N = prob_count / num_simulations
    return P_G_N


'''
The above part is implemented according to the reference.

Assuming we know the channel, we can find the optimal phase, and map this optimal phase to the quantized phase level.
'''


def get_phase(L, N, h_SI, h_ID):
    quant_levels = np.linspace(0, 2 * np.pi, L, endpoint=False)

    opt_phi = -np.angle(h_SI * h_ID)
    opt_phi = np.mod(opt_phi, 2 * np.pi)

    phi_quantized = np.zeros_like(opt_phi)  # 같은 크기의 zero matrix 생성
    for n in range(N):
        # Find the closest quantization level
        idx = np.argmin(np.abs(quant_levels - opt_phi[n]))
        phi_quantized[n] = quant_levels[idx]

    Phi = np.diag(np.exp(1j * phi_quantized))

    return opt_phi, Phi


def outage_probability(L, N, P, delta, eta, Omega_S, Omega_I, threshold, rho):
    prob_count = 0

    num_simulations = int(1e+5)
    pbar = tqdm(range(num_simulations))
    for _ in pbar:
        pbar.set_description(f"Outage simulation when L={L} N={N}")
        # 변수 초기화
        h_SI, h_ID, _, _ = init_system(N, Omega_S, Omega_I, delta)
        _, Phi = get_phase(L, N, h_SI, h_ID)
        g_n, _ = SNR_calc(P, delta, eta, h_SI, Phi, h_ID, Omega_S, Omega_I)

        G_N = np.sum(g_n)
        G_N_sq = np.abs(G_N) ** 2

        if G_N_sq < threshold / rho:
            prob_count += 1

    # Outage probability
    P_G_N = prob_count / num_simulations
    return P_G_N


def opt_outage_probability(N, P, delta, eta, Omega_S, Omega_I, threshold, rho):
    prob_count = 0

    num_simulations = int(1e+5)
    pbar = tqdm(range(num_simulations))
    for _ in pbar:
        pbar.set_description(f"Outage simulation when N={N}")
        # 변수 초기화
        h_SI, h_ID, _, _ = init_system(N, Omega_S, Omega_I, delta)
        opt_phi = -np.angle(h_SI * h_ID)
        opt_phi = np.mod(opt_phi, 2 * np.pi)
        g_n, _ = SNR_calc(P, delta, eta, h_SI, opt_phi, h_ID, Omega_S, Omega_I)

        G_N = np.sum(g_n)
        G_N_sq = np.abs(G_N) ** 2

        if G_N_sq < threshold / rho:
            prob_count += 1

    # Outage probability
    P_G_N = prob_count / num_simulations
    return P_G_N
