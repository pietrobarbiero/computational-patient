import os
import pickle
import sys
import pandas as pd

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

    return


def main():

    plot_dir = "plots/"
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    out_dir = "data/"
    data = []
    labels = []
    for entry in os.listdir(out_dir):
        par_dir = os.path.join(out_dir, entry)
        if os.path.isdir(par_dir):
            for file in os.listdir(par_dir):
                f1 = os.path.join(par_dir, file)
                y_df = pd.read_csv(f1)
                data.append(y_df)
                labels.append(par_dir)

            plot_sub_dir = os.path.join(plot_dir, entry.split("/")[-1])
            if not os.path.isdir(plot_sub_dir):
                os.makedirs(plot_sub_dir)

            # make_plots(t, angII, ang17, diacid, params_list, plot_sub_dir, sbp)

    plt.figure()
    plt.subplot(221)
    plt.title("Right Atrium")
    for y_df, l in zip(data, labels):
        plt.scatter(y_df["Vra"], y_df["Pra"], label=l)
    plt.ylabel("pressure [mmHg]")
    # plt.legend()
    plt.subplot(222)
    plt.title("Left Atrium")
    for y_df in data:
        plt.scatter(y_df["Vla"], y_df["Pla"])
    plt.subplot(223)
    plt.title("Right Ventricle")
    for y_df in data:
        plt.scatter(y_df["Vrv"], y_df["Prv"])
    plt.xlabel("volume [ml]")
    plt.ylabel("pressure [mmHg]")
    plt.subplot(224)
    plt.title("Left Ventricle")
    for y_df in data:
        plt.scatter(y_df["Vlv"], y_df["Plv"])
    plt.xlabel("volume [ml]")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "phases.png"))
    plt.show()

    tau = 100
    t = np.linspace(0, 100, 100)
    x = 1 - t/tau*np.exp(-t/tau/2) + t/tau*np.exp(-t/tau)
    x = 1 / (1 + np.exp(1/10*(t - 80)))
    plt.figure()
    plt.plot(t, x)
    plt.show()

            #
            # tmax = 300
            # plt.figure(figsize=[10,10])
            # plt.subplot(411)
            # plt.title("Right Atrium Blood Volume [ml]")
            # plt.plot(y_df.iloc[:tmax, 1], y_df.iloc[:tmax, 2])
            # plt.subplot(412)
            # plt.title("Right Ventricle Blood Volume [ml]")
            # plt.plot(y_df.iloc[:tmax, 1], y_df.iloc[:tmax, 3])
            # plt.subplot(413)
            # plt.title("Left Atrium Blood Volume [ml]")
            # plt.plot(y_df.iloc[:tmax, 1], y_df.iloc[:tmax, 4])
            # plt.subplot(414)
            # plt.title("Left Ventricle Blood Volume [ml]")
            # plt.plot(y_df.iloc[:tmax, 1], y_df.iloc[:tmax, 5])
            # plt.xlabel("t [sec]")
            # plt.tight_layout()
            # plt.savefig("volumes.png")
            # plt.show()

    return


if __name__ == "__main__":
    main()
