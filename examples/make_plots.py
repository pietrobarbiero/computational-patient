import os
import pickle
import sys
import gc

import matplotlib
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as tck
from sklearn.gaussian_process import GaussianProcessRegressor


def make_plot(dicts, y_str, plot_dir):

    t = dicts["t"]
    y = dicts[y_str][y_str]
    y = pd.DataFrame(y, columns=[y_str])
    data = pd.concat([t, y], axis=1)
    data.sort_values(by=["subjects", "t"])
    std = data.iloc[:, -1].std() / 10
    for i, subject in enumerate(set(data["subjects"])):
        data.loc[data["subjects"] == subject, data.columns[-1]] -= (i + std)


    plt.figure(figsize=[5, 3])
    g = sns.lineplot(x="t", y=y_str, data=data, hue="subjects",
                 hue_order=["C0", "C1", "C2", "C3", "C4", "C5"])
    sns.despine(left=True, bottom=True)
    # g.axes.set_yscale('log')
    # g.axes.grid(True)
    plt.title(y_str.upper())
    plt.ylabel("ng/mL")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{y_str}.png")
    plt.savefig(f"{plot_dir}/{y_str}.pdf")
    plt.show()
    plt.clf()
    plt.close()
    gc.collect()
    return


def make_box_plot(dicts, measure, plot_dir, title, ylabel):
    sns.set_style("whitegrid")
    plt.figure(figsize=[5, 3])
    data = dicts[measure].sort_values(by="subjects")
    plt.title(title)
    sns.boxplot(x="subjects", y=measure, data=data)
    sns.despine(left=True, bottom=True)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{measure}.png")
    plt.savefig(f"{plot_dir}/{measure}.pdf")
    plt.show()
    plt.clf()
    plt.close()
    gc.collect()


def make_lineplot(dicts, measure, plot_dir, title, ylabel, t0=0):
    dicts.reset_index(inplace=True)
    L = []
    for v in range(0, dicts.shape[0]):
        L.append(f'{int(dicts["infection"][v]):2d}_{int(dicts["glucose"][v]):2d}_{int(dicts["dose"][v]):2d}')
    L = pd.Series(L)
    L.name = "L"
    data = pd.concat([dicts, L], axis=1)
    age = "20"

    data[[measure, "L"]].groupby(["L"]).min()

    sns.set_style("whitegrid")
    plt.figure(figsize=[5, 3])
    lab = list(set(L))
    lab.sort()
    for k, i in enumerate(lab):
        mask = (data["L"] == i) & (data["age"] == age)
        d = data.loc[mask, ["t", measure]].copy()
        g = sns.lineplot(x="t", y=measure, data=d[t0:], alpha=0.6)
    plt.legend(["H", "C+T", "V", "C+V", "C+V+T"], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"{title}")
    sns.despine(left=True, bottom=True)
    plt.xlabel("time [day]")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{measure}_time.png")
    plt.savefig(f"{plot_dir}/{measure}_time.pdf")
    plt.show()
    plt.clf()
    plt.close()
    gc.collect()


def make_curve_plot(dicts, measure, plot_dir, title, xlabel, ylabel):
    dicts.reset_index(inplace=True)
    L = []
    for v in range(0, dicts.shape[0]):
        L.append(f'{float(dicts["glucose"][v]):5.2f}')
    L = pd.Series(L)
    L.name = "L"
    data = pd.concat([dicts, L], axis=1)

    data[[measure[0], measure[1], "L"]].groupby(["L"]).max()

    std = data[measure].std() / 8
    min_y = data[measure].min() - 3 * std
    max_y = data[measure].max()
    sns.set_style("whitegrid")
    plt.figure(figsize=[5, 3])
    lab = list(set(L))
    lab.sort()
    for k, i in enumerate(lab):
        mask = (data["L"] == i)
        d = data.loc[mask, ["t", *measure]].copy()
        d = d[d["t"] > 4]
        g = sns.scatterplot(x=measure[0], y=measure[1], data=d, alpha=0.6)
    plt.legend(["H", "D"], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"{title}")
    # plt.xlim([min_y[0], max_y[0]])
    # plt.ylim([min_y[1], max_y[1]])
    sns.despine(left=True, bottom=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_curve.png")
    plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_curve.pdf")
    plt.show()
    plt.clf()
    plt.close()
    gc.collect()


def make_pvplot(dicts, measure, plot_dir, title, xlabel, ylabel):
    dicts.reset_index(inplace=True)
    L = []
    for v in range(0, dicts.shape[0]):
        L.append(f'{int(dicts["infection"][v]):2d}_{int(dicts["glucose"][v]):2d}_{int(dicts["dose"][v]):2d}')
    L = pd.Series(L)
    L.name = "L"
    data = pd.concat([dicts, L], axis=1)

    data[[measure[0], measure[1], "L"]].groupby(["L"]).max()

    ages = set(data["age"])
    # std = data[measure].std() / 8
    # min_y = data[measure].min() - 3*std
    # max_y = data[measure].max()
    # for age in ages:
    sns.set_style("whitegrid")
    plt.figure(figsize=[5, 3])
    lab = list(set(L))
    lab.sort()
    for k, i in enumerate(lab):
        mask = (data["L"] == i)# & (data["age"] == age)
        d = data.loc[mask, ["t", *measure]].copy()
        g = sns.scatterplot(x=measure[0], y=measure[1], data=d[d["t"] > 4], alpha=0.6)
    plt.legend(["H", "C+T", "V", "C+V", "C+V+T"], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"{title}")
    # plt.xlim([min_y[0], max_y[0]])
    # plt.ylim([min_y[1], max_y[1]])
    sns.despine(left=True, bottom=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_pv.png")
    plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_pv.pdf")
    plt.show()
    plt.clf()
    plt.close()
    gc.collect()


def make_lineplot_diabetes(dicts, measure, plot_dir, title, ylabel, xlabel="time [sec]"):
    dicts.reset_index(inplace=True)
    L = []
    for v in range(0, dicts.shape[0]):
        L.append(f'{float(dicts["glucose"][v]):5.2f}')
    L = pd.Series(L)
    L.name = "L"
    data = pd.concat([dicts, L], axis=1)

    data[[measure, "L"]].groupby(["L"]).max()

    std = data[measure].std() / 8
    min_y = data[measure].min() - 3*std
    max_y = data[measure].max()
    sns.set_style("whitegrid")
    plt.figure(figsize=[5, 3])
    lab = list(set(L))
    lab.sort()
    for k, i in enumerate(lab):
        mask = (data["L"] == i)
        d = data.loc[mask, ["t", measure]].copy()
        g = sns.lineplot(x="t", y=measure, data=d, alpha=0.6)
    plt.legend(["H", "D"], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"{title}")
    # plt.ylim([min_y, max_y])
    sns.despine(left=True, bottom=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{measure}_time.png")
    plt.savefig(f"{plot_dir}/{measure}_time.pdf")
    plt.show()
    plt.clf()
    plt.close()
    gc.collect()


def make_lineplot_age(dicts, measure, plot_dir, title, ylabel, xlabel="time [sec]"):
    try:
        dicts.reset_index(inplace=True)
    except:
        pass
    L = []
    for v in range(0, dicts.shape[0]):
        L.append(f'{int(dicts["infection"][v]):2d}_{int(dicts["glucose"][v]):2d}_{int(dicts["dose"][v]):2d}_{dicts["renal"][v]}')
    L = pd.Series(L)
    L.name = "L"
    data = pd.concat([dicts, L], axis=1)

    data[[measure, "L"]].groupby(["L"]).max()

    ages = set(data["age"])
    std = data[measure].std() / 8
    min_y = data[measure].min() - 3*std
    max_y = data[measure].max()
    # for age in ages:
    sns.set_style("whitegrid")
    plt.figure(figsize=[5, 3])
    lab = list(set(L))
    lab.sort()
    labels = ["H", "C+T", "V", "C+V", "C+V+T"]
    ci = 0.05 * 3
    if measure == "diacid":
        lab = [lab[2], lab[1]]
        labels = ["H", "R"]
        ci = 0.05
    for k, i in enumerate(lab):
        mask = (data["L"] == i)# & (data["age"] == age)
        # d = data.loc[mask, ["t", measure]].copy()
        # g = sns.lineplot(x="t", y=measure, data=d, alpha=0.6)
        d = data.loc[mask, ["t", measure]].copy()
        d1 = d.copy()
        d1.iloc[:, 1] += ci * d1.iloc[:, 1]
        d2 = d.copy()
        d2.iloc[:, 1] -= ci * d2.iloc[:, 1]
        d = pd.concat([d, d1, d2], axis=0)
        g = sns.lineplot(x="t", y=measure, data=d, alpha=0.6)
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"{title}")
    # plt.ylim([min_y, max_y])
    sns.despine(left=True, bottom=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{measure}_time.png")
    plt.savefig(f"{plot_dir}/{measure}_time.pdf")
    plt.show()
    plt.clf()
    plt.close()
    gc.collect()


def search_data_aging(out_dir, file_type, measures):
    data = []
    labels = []
    for entry in os.listdir(out_dir):
        par_dir = os.path.join(out_dir, entry)
        if os.path.isdir(par_dir):
            for file in os.listdir(par_dir):
                if file.startswith(file_type):
                    f1 = os.path.join(par_dir, file)
                    y_df = pd.read_csv(f1)
                    data.append(y_df)
                    labels.append(f1)

    label_list = []
    if file_type == "DIABETES":
        for l in labels:
            age = l.split("/")[1]
            glucose = l.split("/")[2].split("_")[1].split("-")[1][:-4]
            label_list.append([glucose])
        df_label = pd.DataFrame(label_list)
        min_t0 = 0
        dicts = pd.DataFrame()
        for d, l in zip(data, df_label.values):
            label_list = pd.DataFrame([l for i in d.iterrows()],
                                      columns=["glucose"])
            df = pd.concat([d.loc[min_t0:, "t"], d.loc[min_t0:, measures], label_list], axis=1)
            dicts = pd.concat([dicts, df])

    else:
        for l in labels:
            age = l.split("/")[1]
            drug_dose = l.split("/")[2].split("_")[1].split("-")[1]
            glucose = l.split("/")[2].split("_")[2].split("-")[1]
            infection = l.split("/")[2].split("_")[3].split("-")[1]
            renal = l.split("/")[2].split("_")[4].split("-")[1][:-4]
            label_list.append([age, drug_dose, glucose, infection, renal])
        df_label = pd.DataFrame(label_list)

        min_t0 = 0
        dicts = pd.DataFrame()
        for d, l in zip(data, df_label.values):
            label_list = pd.DataFrame([l for i in d.iterrows()], columns=["age", "dose", "glucose", "infection", "renal"])
            df = pd.concat([d.loc[min_t0:, "t"], d.loc[min_t0:, measures], label_list], axis=1)
            dicts = pd.concat([dicts, df])

    return dicts


def clotting(dicts, plot_dir):
    mask0 = (dicts["age"] == "70") & (dicts["dose"] == "0") & (dicts["infection"]=="1") & (dicts["renal"]=="impaired")
    mask1 = (dicts["age"]=="70") & (dicts["dose"]=="5") & (dicts["infection"]=="1")
    mask2 = (dicts["age"]=="20")
    # Continuous IV infusion
    # Initial dose: 5000 units by IV injection
    heparin = 5000 # U/mL
    vD_0 = 30 # ng/mL
    vD = 40 # ng/mL
    Ppap3 = dicts[mask1]["Ppap"] - 0.0008 * heparin - 0.05 * (vD - vD_0)
    Ppc3 = dicts[mask1]["Ppc"] - 0.0008 * heparin - 0.05 * (vD - vD_0)
    t0 = dicts[mask0]["t"].values
    t1 = dicts[mask1]["t"].values
    t2 = dicts[mask2]["t"].values
    Ppap0 = dicts[mask0]["Ppap"]
    Ppap1 = dicts[mask1]["Ppap"]
    Ppap2 = dicts[mask2]["Ppap"]
    Ppc0 = dicts[mask0]["Ppc"]
    Ppc1 = dicts[mask1]["Ppc"]
    Ppc2 = dicts[mask2]["Ppc"]

    t = [t0, t1, t2, t1]
    Ppc = [Ppc0, Ppc1, Ppc2, Ppc3]
    Ppap = [Ppap0, Ppap1, Ppap2, Ppap3]

    sns.set_style("whitegrid")
    plt.figure(figsize=[5, 3])
    for tk, p1, p2 in zip(t, Ppap, Ppc):
        print()
        g = sns.scatterplot(x=p1[tk>4], y=p2[tk>4], alpha=0.6)
    plt.legend(["C+V", "C+V+T", "H", "C+V+3T"], loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.title(f"{title}")
    sns.despine(left=True, bottom=True)
    plt.xlabel("artery [mmHg]")
    plt.ylabel("capillaries [mmHg]")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/heparin.png")
    plt.savefig(f"{plot_dir}/heparin.pdf")
    plt.show()
    plt.clf()
    plt.close()
    gc.collect()

    return dicts


def main():

    aging = True
    out_dir = "data/"
    plot_dir = "plots/"
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    if aging:

        # measures = {
        #     "": ["I", "G"],
        # }
        # for title, measure in measures.items():
        #     dicts = search_data_aging(out_dir, "DIABETES", measure)
        #     make_curve_plot(dicts, measure, plot_dir, title, "insulin [mU/ml]", "glucose [ml/dl]")

        # measures = {
        #     "G": ["", "glucose [ml/dl]"],
        #     # "I": ["", "insulin [ml/dl]"],
        # }
        # for measure, (title, ylabel) in measures.items():
        #     dicts = search_data_aging(out_dir, "DIABETES", measure)
        #     make_lineplot_diabetes(dicts, measure, plot_dir, title, ylabel, "time [days]")

        measures = {
            "diacid": ["", "drug diacid [ng/ml]"],
            # "ang17": ["ANG-(1-7)", "concentration [ng/ml]"],
            # "at1r": ["AT1R", "concentration [ng/ml]"],
            # "at2r": ["AT2R", "concentration [ng/ml]"],
            # "ACE2": ["", "ACE2"],
            "IR": ["", "inflammation"],
        }
        for measure, (title, ylabel) in measures.items():
            dicts = search_data_aging(out_dir, "DKD", measure)
            make_lineplot_age(dicts, measure, plot_dir, title, ylabel, "time [days]")

        # measures = {
        #     "": ["artery", "capillaries", "Ppap", "Ppc"],
        #     # "Proximal pulmonary artery": ["Ppap", "Vpap"],
        #     # "Distal pulmonary artery": ["Ppad", "Vpad"],
        #     # "Pulmonary arterioles": ["Ppa", "Vpa"],
        #     # "Pulmonary capillaries": ["Ppc", "Vpc"],
        #     # "Systemic arteries": ["Psa", "Vsa"],
        #     # "Systemic arterioles": ["Psap", "Vsap"],
        #     # "Systemic capillaries": ["Psc", "Vsc"],
        #     # "Systemic veins": ["Psv", "Vsv"],
        # }
        # for title, items in measures.items():
        #     x_label, y_label, measure = items[0], items[1], items[2:]
        #     dicts = search_data_aging(out_dir, "CARDIO", measure)
        #     clotting(dicts, plot_dir)
        #     make_pvplot(dicts, measure, plot_dir, title, f"{x_label} [mmHg]", f"{y_label} [mmHg]")
        #     make_lineplot_age(dicts, measure[0], plot_dir, title, f"{x_label} [mmHg]")
        #     make_lineplot_age(dicts, measure[1], plot_dir, title, f"{y_label} [mmHg]")

    return


if __name__ == "__main__":
    main()
