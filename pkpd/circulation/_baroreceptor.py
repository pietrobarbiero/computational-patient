import numpy as np


def _firing_frequency(d_ABPfol_dt, Nbr_t, Nbr, ABPshift, a, a1, a2, K):
    """
    Equation A.93

    :param d_Nbr_dt:
    :param d_ABPfol_dt:
    :param Nbr_t:
    :param Nbr:
    :param ABPshift:
    :param a:
    :param a1:
    :param a2:
    :param K:
    :return:
    """
    return 1/(a2 * a) * (-((a2 + a) * Nbr_t) - Nbr + (K*ABPshift) + (a1 * K * d_ABPfol_dt))


def _n_change(t, tmin, l, N, K, Nbr_list, time_list, T):
    """
    Equations A.94 and A.95

    :param t:
    :param tmin:
    :param l:
    :param N:
    :param K:
    :param Nbr:
    :param T:
    :return:
    """
    if t - tmin > l:
        # TODO: double check this arrangement
        # Nbr = Nbr_list[np.argmin(np.abs(time_list - (t - l)))]
        Nbr = np.interp(t - l, time_list, Nbr_list)
        return (-N + (K * Nbr)) / T

    return 0


def _f(a, b, tau, N, No):
    """
    Equations A.96 and A.97

    :param a:
    :param b:
    :param tau:
    :param N:
    :param No:
    :return:
    """
    return a + (b / (np.exp(tau*(N-No)) + 1.0))


def _b_vaso(a_vaso):
    """
    Equation A.98

    :param a_vaso:
    :return:
    """
    return 1 - a_vaso


def _af_con(amin, Ka, f_con):
    """
    Equation A.99

    :param amin:
    :param Ka:
    :param f_con:
    :return:
    """
    return amin + (Ka * f_con)
