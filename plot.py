import matplotlib.pyplot as plt
import argparse
import pickle


ALL_METHODS = ["birl", "brex", "avril"]
ALL_NUM_FEATURES = [2, 4, 8]
ALL_PERC_DEMOS = [25, 50, 75, 100]

feature_evolutions = {m: None for m in ALL_METHODS}
demo_evolutions = {m: None for m in ALL_METHODS}


def main(method):
    log_dir = f"./logs/{method}/"
    for num_features in ALL_NUM_FEATURES:
        for perc_demo in ALL_PERC_DEMOS:
            for seed in range(10):
                file_name = f"feat{num_features}_demo{perc_demo}_seed{seed}.pkl"
                with open(log_dir + file_name, "rb") as f:
                    evolution = pickle.load(f)
                    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", "-m", required = True, type = str)
    args = parser.parse_args()
    main(args.method)