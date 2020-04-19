import glob
import os
import sys

import pandas as pd

import msmodel


def main():

    msmodel.covid19_dkd_model(model="cardio")

    return


if __name__ == "__main__":
    main()
