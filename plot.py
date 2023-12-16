import matplotlib.pyplot as plt
import argparse
import pickle


ALL_METHODS = ["birl", "brex", "avril"]
ALL_NUM_FEATURES = [2, 4, 8]
ALL_PERC_DEMOS = [25, 50, 75, 100]
COLORS = {"birl": "red", "brex": "blue", "avril": "green"}


def get_method_logs(method, feat, demo):
    evolutions = []
    for seed in range(10):
        file_name = f"./logs/{method}/feat{feat}_demo{demo}_seed{seed}.pkl"
        with open(file_name, "rb") as f:
            evolution = pickle.load(f)
            if not isinstance(evolution, list):
                evolutions.append(evolution.tolist())
            else:
                evolutions.append(evolution)
    return evolutions


def calc_mean_and_std(evolutions):
    sums = {}
    sums_of_squares = {}
    counts = {}
    for evolution in evolutions:
        for i, dist in enumerate(evolution):
            if i not in sums:
                sums[i] = 0
                sums_of_squares[i] = 0
                counts[i] = 0
            sums[i] += value
            sums_of_squares[i] += dist ** 2
            counts[i] += 1
    means = []
    stds = []
    for i in sorted(sums.keys()):
        mean = sums[i] / counts[i]
        std = ((sums_of_squares[i] / counts[i]) - mean ** 2) ** 0.5
        means.append(mean)
        stds.append(std)
    assert len(means) == len(stds), "you fucked up"
    return (means, stds)


def plot_data(data, feat, demo):
    for method in ALL_METHODS:
        mean, std = data[method]
        x = range(len(mean))
        plt.plot(x, mean, label = method.upper(), color = COLORS[method])
        plt.fill_between(x, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                         color = COLORS[method], alpha = 0.3)
    plt.xlabel("Iteration")
    plt.ylabel("Distance to Ground Truth Reward")
    plt.title(f"{feat} Features, {demo}% States as Demos")
    plt.legend()
    plt.savefig(f"./plots/conv_feat{feat}_demo{demo}.png")


def main():
    for num_features in ALL_NUM_FEATURES:
        for perc_demo in ALL_PERC_DEMOS:
            evolutions = {}
            data = {}
            for method in ALL_METHODS:
                evolutions[method] = get_method_logs(method, num_features, perc_demo)
                data[method] = calc_mean_and_std(evolutions[method])
                plot_data(data, num_features, perc_demo)


if __name__ == "__main__":
    main()