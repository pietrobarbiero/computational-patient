import numpy as np
from scipy.integrate import solve_ivp


def _tc_ode(t, ES, K_tr, ES_list):
    ES_list.append(ES[0])
    if t == 0:
        return -K_tr * ES_list[0]
    else:
        return K_tr * ES_list[-2] - K_tr * ES_list[-1]


def transit_compartment_model(ES, K_tr, counter):
    # return ES / counter
    y0 = np.array([ES])
    ES_list = []
    args = (K_tr, ES_list)
    sol = solve_ivp(fun=_tc_ode, t_span=[0, counter], y0=y0,
                    args=args, method="RK23")
    return sol["y"][:, -1][0]


def change_S_ANG17(K_in_ANG17, ES_ANG2_n, ES_p, K_out_ANG17, S_ANG17):
    """
    Equation 1

    :param K_in_ANG17:
    :param ES_ANG2_n:
    :param ES_p:
    :param K_out_ANG17:
    :param S_ANG17:
    :return:
    """
    return K_in_ANG17 * (1 + ES_ANG2_n) * (1 + ES_p) - K_out_ANG17 * S_ANG17


def change_S_ANG2(K_in_ANG2, EI_p, K_out_ANG2, ES_ANG17_m, S_ANG2):
    """
    Equation 2

    :param K_in_ANG2:
    :param EI_p:
    :param K_out_ANG2:
    :param ES_ANG17_m:
    :param S_ANG2:
    :return:
    """
    return K_in_ANG2 * (1 - EI_p) - K_out_ANG2 * (1 + ES_ANG17_m) * S_ANG2


def change_S_SBP(K_in_SBP, SSS_ANG2, S_ANG2, III_ANG17, S_ANG17, K_out_SBP, S_SBP):
    """
    Equation 3

    :param K_in_SBP:
    :param SSS_ANG2:
    :param S_ANG2:
    :param III_ANG17:
    :param S_ANG17:
    :param K_out_SBP:
    :param S_SBP:
    :return:
    """
    return K_in_SBP * (1 + SSS_ANG2 * S_ANG2 - III_ANG17 * S_ANG17) - K_out_SBP * S_SBP
