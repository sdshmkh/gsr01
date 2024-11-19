import matplotlib.pyplot as plt
import seaborn as sns

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
