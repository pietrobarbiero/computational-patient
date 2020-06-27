import numpy as np


def _psi(V, Kxp, Kxv):
    """
    Psi

    :param V:
    :param Kxp:
    :param Kxv:
    :return:
    """
    return Kxp * (1 / (np.exp(V / Kxv) - 1))
