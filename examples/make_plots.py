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
    std = data[measure].std() / 8
    min_y = data[measure].min() - 3*std
    max_y = data[measure].max()
    for age in ages:
        sns.set_style("whitegrid")
        plt.figure(figsize=[5, 3])
        lab = list(set(L))
        lab.sort()
        for k, i in enumerate(lab):
            mask = (data["L"] == i) & (data["age"] == age)
            d = data.loc[mask, ["t", *measure]].copy()
            g = sns.scatterplot(x=measure[0], y=measure[1], data=d, alpha=0.6)
        plt.legend(["H", "C+T", "V", "C+V", "C+V+T"], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f"{title} - Age {age}")
        plt.xlim([min_y[0], max_y[0]])
        plt.ylim([min_y[1], max_y[1]])
        sns.despine(left=True, bottom=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_{age}_pv.png")
        plt.savefig(f"{plot_dir}/{measure[0]}_{measure[1]}_{age}_pv.pdf")
        plt.show()
        plt.clf()
        plt.close()
        gc.collect()


def make_lineplot_age(dicts, measure, plot_dir, title, ylabel):
    dicts.reset_index(inplace=True)
    L = []
    for v in range(0, dicts.shape[0]):
        L.append(f'{int(dicts["infection"][v]):2d}_{int(dicts["glucose"][v]):2d}_{int(dicts["dose"][v]):2d}')
    L = pd.Series(L)
    L.name = "L"
    data = pd.concat([dicts, L], axis=1)

    data[[measure, "L"]].groupby(["L"]).max()

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
            g = sns.lineplot(x="t", y=measure, data=d, alpha=0.6)
        plt.legend(["H", "C+T", "V", "C+V", "C+V+T"], loc='center left', bbox_to_anchor=(1, 0.5))
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
        df = pd.concat([d.loc[min_t0:, "t"], d.loc[min_t0:, measures], label_list], axis=1)
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
        #     "diacid": ["Benazepril", "concentration [ng/ml]"],
        #     # "ang17": ["ANG-(1-7)", "concentration [ng/ml]"],
        #     # "at1r": ["AT1R", "concentration [ng/ml]"],
        #     # "at2r": ["AT2R", "concentration [ng/ml]"],
        #     "KS": ["Inflammatory response", "score"],
        # }
        # for measure, (title, ylabel) in measures.items():
        #     dicts = search_data_aging(out_dir, "DKD", measure)
        #     make_lineplot(dicts, measure, plot_dir, title, ylabel)

        measures = {
            "Proximal pulmonary artery": ["Ppap", "Vpap"],
            "Distal pulmonary artery": ["Ppad", "Vpad"],
            "Pulmonary arterioles": ["Ppa", "Vpa"],
            "Pulmonary capillaries": ["Ppc", "Vpc"],
            "Systemic arteries": ["Psa", "Vsa"],
            "Systemic arterioles": ["Psap", "Vsap"],
            "Systemic capillaries": ["Psc", "Vsc"],
            "Systemic veins": ["Psv", "Vsv"],
        }
        for title, measure in measures.items():
            dicts = search_data_aging(out_dir, "CARDIO", measure)
            make_pvplot(dicts, measure, plot_dir, title, "pressure [mmHg]", "volume [mL]")
            # make_lineplot_age(dicts, measure, plot_dir, title, "pressure [mmHg]")

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
