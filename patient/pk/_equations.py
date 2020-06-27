import numpy as np


def analytical_PK(drugdose, ka, VF, ke, t, tau, tfinal_dosing, tstart_dosing, glu=None):
    """
    Equation 1

    :param drugdose:
    :param ka:
    :param VF:
    :param ke:
    :param t:
    :param tau:
    :param tfinal_dosing:
    :param tstart_dosing:
    :param glu:
    :return:
    """
    if t > tstart_dosing:
        if t < tfinal_dosing:
            n = np.floor(t / tau) + 1

        else:
            n = np.floor(tfinal_dosing / tau)

        tprime = t - tau * (n - 1)
        drug_conc_theo = drugdose * ka / (VF * (ka - ke)) * (
                (1 - np.exp(-n * ke * tau)) * (np.exp(-ke * tprime)) / (1 - np.exp(-ke * tau))
                - (1 - np.exp(-n * ka * tau)) * (np.exp(-ka * tprime)) / (1 - np.exp(-ka * tau)))

    else:
        drug_conc_theo = 0

    return drug_conc_theo
