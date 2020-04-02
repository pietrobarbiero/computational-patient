import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_plots(t, y_a_list, y_an_list, y_d_list, args_list, plot_dir):
    labels = []
    for i in range(0, len(args_list)):
        labels.append(f"{args_list[i].dose:.2f} mg/die - {args_list[i].renal_function}")

    f = plt.figure()
    for i in range(0, len(y_an_list)):
        plt.plot(t, y_an_list[i], label=labels[i])
    f.axes[0].ticklabel_format(style='plain')
    plt.xlabel("t (days)")
    plt.ylabel(r"$\frac{[ANG II]}{[ANG II]_0}$")
    plt.xlim([0, np.ceil(t[-1])])
    plt.ylim([0, 100])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "./ANGII_norm.png"))
    plt.show()

    f = plt.figure(figsize=[10, 5])
    for i in range(0, len(y_d_list)):
        plt.plot(t, y_d_list[i], label=labels[i])
    f.axes[0].ticklabel_format(style='plain')
    plt.xlim([0, np.ceil(t[-1])])
    plt.ylim([0, 200])
    plt.xlabel("t (days)")
    plt.ylabel("Drug Diacid Concentration (ng/mL)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "./diacid.png"))
    plt.show()

    f = plt.figure(figsize=[10, 5])
    for i in range(0, len(y_a_list)):
        plt.plot(t, y_a_list[i], label=labels[i])
    plt.xlim([0, np.ceil(t[-1])])
    plt.ylim([1e-3, 1e3])
    plt.xlabel("t (days)")
    plt.ylabel("Ang II Conc. (nmol/L) (ng/mL)")
    f.axes[0].ticklabel_format(style='plain')
    f.axes[0].set_yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "./ANGII.png"))
    plt.show()

    return


def main():

    plot_dir = "./plots/"
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    out_dir = "./data/"
    for entry in os.listdir(out_dir):
        par_dir = os.path.join(out_dir, entry)
        if os.path.isdir(par_dir):
            params_list = []
            angII = []
            angII_norm = []
            diacid = []
            for file in os.listdir(par_dir):
                f1 = os.path.join(par_dir, file)
                vars = pickle.load(open(f1, 'rb'))
                params_list.append(vars["params"])
                angII.append(vars["angII"])
                angII_norm.append(vars["angII_norm"])
                diacid.append(vars["diacid"])
            t = vars["t"]

            plot_sub_dir = os.path.join(plot_dir, entry.split("/")[-1])
            if not os.path.isdir(plot_sub_dir):
                os.makedirs(plot_sub_dir)

            make_plots(t, angII, angII_norm, diacid, params_list, plot_sub_dir)

    return


if __name__ == "__main__":
    main()
