import sys

import msmodel


def main():

    sys.argv.extend(["--age", "20"])
    msmodel.covid19_dkd_model(model="cardio")

    sys.argv.extend(["--age", "60"])
    msmodel.covid19_dkd_model(model="cardio")

    return


if __name__ == "__main__":
    main()
