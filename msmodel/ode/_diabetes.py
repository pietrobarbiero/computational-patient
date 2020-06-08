import os
import pickle

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import scipy
import pandas as pd
import seaborn as sns


def ODE(t, y,
        stepg, stepe,
        k, alpha, theta, R0, EG0, SI,
        d0, r1, r2, m0, c4, c5,
        i0, m, q, lt):
    I, G, beta, IR, MTOR, Tt, C = y

    dI = (beta * theta * G * G) / (alpha + G * G) - k * I
    dG = R0 - G * (EG0 + SI * I / (IR + 1)) + 1.0 * stepg.eval(C) - 0.1 * stepe.eval(C)
    dbeta = beta * (-d0 + r1 * G - r2 * G * G)
    dIR = -i0 * IR + m * MTOR + q * I
    dMTOR = -m0 * MTOR + c4 * I / (IR + 1) + c5 * G
    dTt = 0.001 * (20 / (1 + np.exp(-0.05 * (G - 100)))) * Tt * np.log(lt / Tt)
    dC = 1  # means that C=time, allowing the step function at that time to be used in the glucose equation

    d_conc_dt = np.array([dI, dG, dbeta, dIR, dMTOR, dTt, dC])

    return d_conc_dt


def diabetes_model(days, stepg, stepe,
                   I0, G0, beta0, IR0, MTOR0, Tt, C0,
                   k, alpha, theta, R0, EG0, SI,
                   d0, r1, r2, m0, c4, c5,
                   i0, m, q, lt):
    # initial condition for the ODE solver
    conc_t0 = np.array([I0, G0, beta0, IR0, MTOR0, Tt, C0])
    tau = 0.001

    ODE_args = (
        stepg, stepe,
        k, alpha, theta, R0, EG0, SI,
        d0, r1, r2, m0, c4, c5,
        i0, m, q, lt
    )
    sol = solve_ivp(fun=ODE, t_span=[0, days],
                    y0=conc_t0,
                    max_step=tau,
                    args=ODE_args, method="LSODA")

    return sol


def step_function(x_init, y_init, days):
    x = np.array(x_init) / 24
    count = np.arange(0, days)
    x_final = []
    for i in range(0, days):
        x_final.extend(x + count[i])
    x_final = np.array(x_final)
    k_rep = len(x_final) / len(y_init)
    y_final = int(k_rep) * y_init
    y_final = np.array(y_final)
    y_final = np.append(y_final, np.array(y_init[0]))
    step = StepFun(x_final, y_final)
    return step


class StepFun:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def eval(self, xi):
        diff = self.x - xi
        t = sum(diff < 0)
        return self.y[t]


def call_diabetes(args):
    out_dir = f"./data/{args.age}"
    file_name = f"DIABETES_glu-{int(args.glu)}.csv"
    file = os.path.join(out_dir, file_name)
    if os.path.isfile(file):
        return

    days = int(args.sim_time_end / 24)

    # glucose step function
    x_init = [8, 8.5, 12, 12.5, 20, 20.5]
    y_init = [0, 200, 0, 4200, 0, 4200]
    stepg = step_function(x_init, y_init, days)

    # exercise step function
    xe_init = [18, 18.5]
    ye_init = [0, 2000]
    stepe = step_function(xe_init, ye_init, days)

    # the diabetes equations parameters
    k = 432
    alpha = 20000
    theta = 43.2
    R0 = 864
    EG0 = 0.44
    if args.glu < 8:
        SI = 1.62
    else:
        SI = 0.52
    d0 = 0.06
    r1 = 0.00084
    r2 = 0.0000024
    m0 = 47.7
    c4 = 9
    c5 = 6
    i0 = 87
    m = 2
    q = 0.017
    lt = 5

    # initial conditions
    I0 = 13.59
    G0 = 100
    beta0 = 407.73
    IR0 = 0.359
    MTOR0 = 14.465
    Tt = 1
    C0 = 0

    sol = diabetes_model(days, stepg, stepe,
                         I0, G0, beta0, IR0, MTOR0, Tt, C0,
                         k, alpha, theta, R0, EG0, SI,
                         d0, r1, r2, m0, c4, c5,
                         i0, m, q, lt)

    x = np.concatenate([sol["t"].reshape((1, -1)), sol["y"]])
    sol_df = pd.DataFrame(x.T, columns=["t", "I", "G", "beta", "IR", "MTOR", "Tt", "C"])

    out_dir = f"./data/{args.age}"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    file_name = f"DIABETES_glu-{int(args.glu)}.csv"
    file = os.path.join(out_dir, file_name)
    sol_df.to_csv(file)

    args.glu = sol_df[sol_df["t"] > (days-1)]["G"].max() / 18
    return
