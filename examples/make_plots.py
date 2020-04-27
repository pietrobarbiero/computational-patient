import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_plots(t, y1_list, y2_list, y_d_list, args_list, plot_dir, pressure_list):
    labels = []
    for i in range(0, len(args_list)):
        labels.append(f"{args_list[i].renal_function}")

    f = plt.figure(figsize=[5, 5])

    plt.subplot(211)
    plt.title(f"Drug concentration [{args_list[0].dose:.2f} mg/die]")
    for i in range(0, len(y_d_list)):
        plt.plot(t, y_d_list[i], label=labels[i])
    f.axes[0].ticklabel_format(style='plain')
    plt.xlim([0, np.ceil(t[-1])])
    plt.ylim([0, 100])
    # plt.xlabel("t (days)")
    plt.ylabel("ng/mL")
    plt.legend()

    plt.subplot(212)
    plt.title("Sistolic blood pressure")
    for i in range(0, len(pressure_list)):
        plt.plot(t, pressure_list[i], label=labels[i])
    plt.xlim([0, np.ceil(t[-1])])
    plt.ylim([100, 200])
    plt.xlabel("t (days)")
    plt.ylabel("mmHg")
    # plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "./diacid.png"))
    plt.show()

    # f = plt.figure(figsize=[10, 10])
    #
    # plt.subplot(311)
    # for i in range(0, len(y1_list)):
    #     plt.plot(t, y1_list[i], label="ANG-II - " + labels[i])
    # plt.xlim([0, np.ceil(t[-1])])
    # plt.ylim([1e-3, 1e3])
    # plt.xlabel("t (days)")
    # plt.ylabel("Conc. (nmol/L) (ng/mL)")
    # f.axes[0].ticklabel_format(style='plain')
    # f.axes[0].set_yscale('log')
    # plt.legend()
    #
    # plt.subplot(312)
    # for i in range(0, len(y2_list)):
    #     plt.plot(t, y2_list[i], label="ANG-(1-7) - " + labels[i])
    # plt.xlim([0, np.ceil(t[-1])])
    # plt.ylim([1e0, 1e1])
    # plt.xlabel("t (days)")
    # plt.ylabel("Conc. (nmol/L) (ng/mL)")
    # f.axes[1].ticklabel_format(style='plain')
    # f.axes[1].set_yscale('log')
    # plt.legend()
    #
    # plt.subplot(313)
    # for i in range(0, len(pressure_list)):
    #     plt.plot(t, pressure_list[i], label="SBP - " + labels[i])
    # plt.xlim([0, np.ceil(t[-1])])
    # plt.ylim([100, 200])
    # plt.xlabel("t (days)")
    # plt.ylabel("pressure (mmHg)")
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.savefig(os.path.join(plot_dir, "./ANG.png"))
    # plt.show()

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
            ang17 = []
            diacid = []
            sbp = []
            for file in os.listdir(par_dir):
                f1 = os.path.join(par_dir, file)
                vars = pickle.load(open(f1, 'rb'))
                params_list.append(vars["params"])
                angII.append(vars["angII"])
                ang17.append(vars["ang17"])
                sbp.append(vars["sbp"])
                diacid.append(vars["diacid"])
            t = vars["t"]

            plot_sub_dir = os.path.join(plot_dir, entry.split("/")[-1])
            if not os.path.isdir(plot_sub_dir):
                os.makedirs(plot_sub_dir)

            make_plots(t, angII, ang17, diacid, params_list, plot_sub_dir, sbp)

    return


if __name__ == "__main__":
    main()
