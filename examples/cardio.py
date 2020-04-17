import glob
import os
import sys

import pandas as pd

import pkpd


def main():

    pkpd.covid19_dkd_model()
    #
    # file_list = glob.glob(os.path.join("jsim_data", "**/*.csv"), recursive=True)
    # data = pd.DataFrame()
    # for i, file in enumerate(file_list):
    #     jsim_df = pd.read_csv(file)
    #     # if i > 1:
    #     #     jsim_df.drop(["t"], axis=1, inplace=True)
    #     data = pd.concat([data, jsim_df], axis=1)
    #
    # data.to_csv("jsim_data.csv")

    return


if __name__ == "__main__":
    main()
