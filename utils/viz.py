import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

from ml.utils import boundary_indexes

def plot_bar(df, metric, label, title):
    plt.figure(figsize=(10, 6))

    custom_colors = ["orange", "red", "green", "blue"]
    sns.barplot(data=df, x="latent_space", y=metric, hue="model", palette=custom_colors, ci=None)

    custom_legend_labels = ["kNN", "RBF SVM", "Linear SVM", "Xgboost"]

    custom_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
                    for color in custom_colors]
    plt.legend(custom_legend, custom_legend_labels, title="Models", loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Latent Space", fontsize=16)
    plt.ylabel(label, fontsize=16)

    plt.tight_layout()
    plt.savefig("outputs/viz/{}.png".format(title))



def window_to_index(windex, stride):
  return (windex * stride, (windex * stride) + stride)

def cm_on_signal(clf, data_path, signal, ground_truth, title):

    data = np.load(data_path)
    X, Y = data['latent_space'], data['labels']
    X = X.squeeze(2)
    # X, Y = X[255:], Y[255:]
    bounds = boundary_indexes(Y)*51

    idx_X = np.where(Y != 1)[0]

    window_predictions = list()
    predictions = np.zeros_like(signal)
    

    for idx_x in idx_X:
        # predict
        window_prediction = clf.predict(X[idx_x].reshape(1, -1))

        window_start, window_end = window_to_index(idx_x, 51)
        # create a ground truth window
        ground_truth[window_start:window_end] = Y[idx_x]

        # create a prediction window
        if type(clf) == XGBClassifier:
            predictions[window_start:window_end] = 3 if window_prediction == 1 else 2
            window_predictions.append((3 if window_prediction[0] == 1 else 2, Y[idx_x]))
        else:
            predictions[window_start:window_end] = window_prediction
            window_predictions.append((window_prediction[0], Y[idx_x]))


    pos, neg = 3, 2
    tp = np.where((predictions == pos) & (ground_truth == pos), signal, np.nan)
    tn = np.where((predictions == neg) & (ground_truth == neg), signal, np.nan)
    fn = np.where((predictions == neg) & (ground_truth == pos), signal, np.nan)
    fp = np.where((predictions == pos) & (ground_truth == neg), signal, np.nan)

    # # Plotting
    plt.figure(figsize=(12, 6))
    x = np.arange(len(signal))

    plt.plot(signal, color='gray', label="Baseline")

    # True Positive (TP)
    plt.plot(tp, color='darkblue', linestyle='-', label="True Positive (TP)")

    # True Negative (TN)
    plt.plot(tn, color='green', linestyle='-', label="True Negative (TN)")

    # False Positive (FP)
    plt.plot(fp, color='red', linestyle='-', label="False Positive (FP)")

    # False Negative (FN)
    plt.plot(fn, color='orange', linestyle='-', label="False Negative (FN)")

    plt.axvline(x=bounds[-2], color='black', linestyle='--')
    plt.text(bounds[1]-1000, 0.49, "Test on Training data", color="black", fontsize=16)
    plt.text(bounds[-1] - 5000, 0.49, "Test data", color="black", fontsize=16)



    total_time = len(signal)
    plt.xticks(np.linspace(0, total_time, 5), [f'{int(t/(51*60))}' for t in np.linspace(0, total_time, 5)])  # 5 equally spaced ticks
    plt.xlabel('Time (minutes)', fontsize=16)

    plt.ylabel('GSR Signal (μS)', fontsize=16)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', ncol=2, fontsize=16)

    plt.savefig("outputs/viz/{}.png".format(title))