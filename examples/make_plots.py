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
    L = dicts["infection"] + dicts["glucose"] + dicts["dose"]
    L.name = "L"
    data = pd.concat([dicts, L], axis=1)
    age = "60"

    sns.set_style("whitegrid")
    plt.figure(figsize=[5, 3])
    lab = list(set(L))
    lab.sort()
    for k, i in enumerate(lab):
        mask = (data["L"] == i) & (data["age"] == age)
        d = data.loc[mask, ["t", measure]].copy()
        g = sns.lineplot(x="t", y=measure, data=d[t0:], alpha=0.6)
    plt.legend(["H", "T", "C+", "T+"], loc='center left', bbox_to_anchor=(1, 0.5))
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


def make_lineplot_age(dicts, measure, plot_dir, title, ylabel):
    # mask1 = (dicts[measure]["infection"] == "0") & (dicts[measure]["glucose"] == "5") & (dicts[measure]["dose"] == "0")
    # mask2 = (dicts[measure]["infection"] == "1") & (dicts[measure]["glucose"] == "25") & (dicts[measure]["dose"] == "0")
    # mask3 = (dicts[measure]["infection"] == "1") & (dicts[measure]["glucose"] == "25") & (dicts[measure]["dose"] == "5")

    # L = dicts["infection"] + dicts["glucose"] + dicts["dose"] + dicts["renal"] + dicts["age"]
    # L.name = "L"
    L = dicts["infection"] + dicts["glucose"] + dicts["dose"]
    L.name = "L"
    data = pd.concat([dicts, L], axis=1)

    ages = set(data["age"])
    std = data[measure].std() / 8
    min_y = data[measure].min() - 3*std
    max_y = data[measure].max()
    for age in ages:
        sns.set_style("whitegrid")
        plt.figure(figsize=[8, 3])
        lab = list(set(L))
        lab.sort()
        for k, i in enumerate(lab):
            mask = (data["L"] == i) & (data["age"] == age)
            d = data.loc[mask, ["t", measure]].copy()
            # d[measure] -= k * std
            g = sns.lineplot(x="t", y=measure, data=d, alpha=0.6)
        plt.legend(["H", "T", "C+", "T+"], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f"{title} - Age {age}")
        plt.ylim([min_y, max_y])
        sns.despine(left=True, bottom=True)
        plt.xlabel("time [sec]")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{measure}_{age}_time.png")
        plt.savefig(f"{plot_dir}/{measure}_{age}_time.pdf")
        plt.show()
        plt.clf()
        plt.close()
        gc.collect()


    # L = dicts["infection"] + dicts["glucose"] + dicts["dose"]
    # L.name = "L"
    # data = pd.concat([dicts, L], axis=1)
    #
    # sns.set_style("whitegrid")
    # plt.figure(figsize=[5, 3])
    # plt.title(title)
    # lab = list(set(L))
    # lab.sort()
    # for i in lab:
    #     g = sns.lineplot(x="age", y=measure, data=data[data["L"] == i], ci=99)
    # sns.despine(left=True, bottom=True)
    # plt.xlabel("age")
    # plt.ylabel(ylabel)
    # plt.legend(["H", "I", "T"], loc='center left', bbox_to_anchor=(1, 0.5))
    # # g.axes.grid(False)
    # plt.tight_layout()
    # plt.savefig(f"{plot_dir}/{measure}_aging.png")
    # plt.savefig(f"{plot_dir}/{measure}_aging.pdf")
    # plt.show()
    # plt.clf()
    # plt.close()
    # gc.collect()


def search_data(out_dir, file_type):
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
    for l in labels:
        age = int(l.split("/")[1])
        drug_dose = int(l.split("/")[2].split("_")[1].split("-")[1])
        glucose = float(l.split("/")[2].split("_")[2].split("-")[1])
        infection = int(l.split("/")[2].split("_")[3].split("-")[1])
        renal = l.split("/")[2].split("_")[4].split("-")[1][:-4]
        if age == 20:
            new_label = "C0"
        elif age == 60 and glucose == 5 and infection == 0 and drug_dose == 0:
            new_label = "C1"
        elif age == 60 and glucose == 5 and infection == 0 and drug_dose == 5:
            new_label = "C2"
        elif age == 60 and glucose == 25 and infection == 0 and drug_dose == 5:
            new_label = "C3"
        elif age == 60 and glucose == 5 and infection == 1 and drug_dose == 5:
            new_label = "C4"
        elif age == 60 and glucose == 25 and infection == 1 and drug_dose == 5:
            new_label = "C5"
        label_list.append(new_label)
    df_label = pd.DataFrame(label_list)

    columns = data[0].columns
    dicts = {}
    for c in columns:
        dicts[c] = pd.DataFrame()
        for d, l in zip(data, df_label.values):
            label_list = [l[0] for i in d.iterrows()]
            df = pd.concat([d.loc[:, c], pd.DataFrame(label_list, columns=["subjects"])], axis=1)
            dicts[c] = pd.concat([dicts[c], df])

    return dicts


def search_data_aging(out_dir, file_type, measure):
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
    for l in labels:
        age = l.split("/")[1]
        drug_dose = l.split("/")[2].split("_")[1].split("-")[1]
        glucose = l.split("/")[2].split("_")[2].split("-")[1][:-2]
        infection = l.split("/")[2].split("_")[3].split("-")[1]
        renal = l.split("/")[2].split("_")[4].split("-")[1][:-4]
        label_list.append([age, drug_dose, glucose, infection, renal])
    df_label = pd.DataFrame(label_list)

    min_t0 = 0
    dicts = pd.DataFrame()
    for d, l in zip(data, df_label.values):
        label_list = pd.DataFrame([l for i in d.iterrows()], columns=["age", "dose", "glucose", "infection", "renal"])
        df = pd.concat([d.loc[min_t0:, "t"], d.loc[min_t0:, measure], label_list], axis=1)
        dicts = pd.concat([dicts, df])

    return dicts


def main():

    aging = True
    out_dir = "data/"
    plot_dir = "plots/"
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    if aging:

        # measures = {
        #     "diacid": "Benazepril",
        #     "ang17": "ANG-(1-7)",
        #     "at1r": "AT1R",
        #     "at2r": "AT2R",
        # }
        # for measure, title in measures.items():
        #     dicts = search_data_aging(out_dir, "DKD", measure)
        #     make_lineplot(dicts, measure, plot_dir, title, "concentration [ng/ml]")

        measures = {
            # "Ppap": "Proximal pulmonary artery",
            # "Ppad": "Distal pulmonary artery",
            # "Ppa": "Pulmonary arterioles",
            "Ppc": "Pulmonary capillaries",
            # "Psa": "Systemic arteries",
            # "Psap": "Systemic arterioles",
            "Psc": "Systemic capillaries",
            "Psv": "Systemic veins",
        }
        for measure, title in measures.items():
            dicts = search_data_aging(out_dir, "CARDIO", measure)
            make_lineplot_age(dicts, measure, plot_dir, title, "pressure [mmHg]")

    # else:
    #     dicts = search_data(out_dir, "DKD")
    #     make_plot(dicts, "diacid", plot_dir)
    #     make_plot(dicts, "ang17", plot_dir)
    #     make_plot(dicts, "at1r", plot_dir)
    #     make_plot(dicts, "at2r", plot_dir)
    #
        # dicts = search_data(out_dir, "CARDIO")
        # make_box_plot(dicts, "Ppap", plot_dir, "Proximal pulmonary artery", "pressure [mmHg]")
        # make_box_plot(dicts, "Ppad", plot_dir, "Distal pulmonary artery", "pressure [mmHg]")
        # make_box_plot(dicts, "Ppa", plot_dir, "Pulmonary arterioles", "pressure [mmHg]")
        # make_box_plot(dicts, "Ppc", plot_dir, "Pulmonary capillaries", "pressure [mmHg]")
        # make_box_plot(dicts, "Psa", plot_dir, "Systemic arteries", "pressure [mmHg]")
        # make_box_plot(dicts, "Psap", plot_dir, "Systemic arterioles", "pressure [mmHg]")
        # make_box_plot(dicts, "Psc", plot_dir, "Systemic capillaries", "pressure [mmHg]")
        # make_box_plot(dicts, "Psv", plot_dir, "Systemic veins", "pressure [mmHg]")

    return


if __name__ == "__main__":
    main()
