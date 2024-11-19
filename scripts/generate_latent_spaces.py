import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from dl.autoencoders import AutoEncoder, training_loop, get_arch
from dl.datasets import GSRDataset, GSRLowRankDataset, GSRPhasicDataset, GSRTonicDataset

def dataset_name(dataset):
    if type(dataset) == GSRDataset:
        return "gsr"
    elif type(dataset) == GSRTonicDataset:
        return "tonic"
    elif type(dataset) == GSRPhasicDataset:
        return "phasic"
    else:
        return "low_rank"

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
csv_dataset = 'gsr_data/contractive/gsr_data.csv'


all_datasets = [GSRDataset(csv_dataset, 51), GSRLowRankDataset(csv_dataset, 51), GSRPhasicDataset(csv_dataset, 51), GSRTonicDataset(csv_dataset, 51)]

for dataset in all_datasets:

    test_dataset, train_val_dataset = torch.utils.data.random_split(dataset, [int(0.2 * len(dataset)), len(dataset) - int(0.2 * len(dataset))])
    val_dataset, train_dataset = torch.utils.data.random_split(train_val_dataset, [int(0.2 * len(train_val_dataset)), len(train_val_dataset) - int(0.2 * len(train_val_dataset))])
    dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=32, shuffle=True)

    arch = {
        3: "latent_space_3", 6: "latent_space_6", 96: "latent_space_96", 128: "latent_space_128", 256:"latent_space_256", 512: "latent_space_512"
    }

    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    axes = axes.flatten()

    arch_list = sorted(list(arch.items()))
    decoded_signals = list()
    idx = 0
    lines, labels = list(), list()
    for k, v in arch_list:
        model = AutoEncoder(k)
        losses = training_loop(model, dl, val_dl, 20, 0.001, 0.001)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i in range(len(test_dataset)):
                target = test_dataset[i].reshape(-1, 51)
                output = model(target)
                loss = nn.functional.mse_loss(output, target)
                test_loss += loss.item()
        test_loss /= len(test_dataset)

        print(f"\n Test Loss: {test_loss}")

        ls = model.get_latent_space(dataset.gsr_signal.squeeze(2)).unsqueeze(2).numpy()
        decoded_signal = model(dataset.gsr_signal.squeeze(2)).detach().numpy()
        decoded_signals.append(decoded_signal)
        print(ls.shape, dataset.labels.shape, decoded_signal.shape)
        file_name = "{}_latent_space_{}.npz".format(dataset_name(dataset), k)
        file_path = Path("outputs/{}".format(timestamp))
        if not file_path.exists():
            file_path.mkdir(parents=True)

        file_path = file_path.joinpath(file_name)
        print(file_path)
        np.savez_compressed(str(file_path), latent_space=ls, labels=dataset.labels)

        axes[idx].plot(losses[:, 0], label='Training Loss')
        axes[idx].plot(losses[:, 1], label='Validation Loss')
        axes[idx].set_title("{}".format(k), fontsize=14)
        axes[idx].set_xticklabels([0, 5, 10, 20], fontsize=16)
        min_loss = losses.min()
        max_loss = losses.max()
        yticks = np.linspace(min_loss, max_loss, num=5)
        axes[idx].set_yticks(yticks)
        axes[idx].set_yticklabels(["{:.2f}".format(y) for y in yticks], fontsize=16)

        line, label = axes[idx].get_legend_handles_labels()
        lines.append(line)
        labels.append(label)
        idx += 1



    # Add a single legend for the entire figure
    fig.legend(lines[0], labels[0],fontsize=14, loc='lower right')
    plot_path = Path('outputs/viz/')
    if not plot_path.exists():
        plot_path.mkdir(parents=True)

    fig.supxlabel("Epochs", fontsize=16)
    fig.supylabel("Loss", fontsize=16)
    fig.suptitle("Latent Space Dimension vs Training Loss and Validation Loss", fontsize=16)
    fig.tight_layout()
    plt.savefig(str(plot_path.joinpath('{}_loss.png'.format(dataset_name(dataset)))))