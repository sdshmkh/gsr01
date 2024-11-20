import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from ml.models import compile_grid_search
from utils.viz import plot_bar, cm_on_signal

########### input ###########
parser = argparse.ArgumentParser(description="Run a grid search for ML parameters.")
parser.add_argument("--input_folder", type=str, help="Folder which contains latent spaces")
args = parser.parse_args()

folder = args.input_folder

ls_type = {
    "low_rank": "Low Rank",
    "phasic": "phasic",
    "tonic":"Tonic",
    "gsr": "Gsr", 
}

gsr_files =  {
    3: "gsr_latent_space_3.npz",
    6: "gsr_latent_space_6.npz",
    96: "gsr_latent_space_96.npz",
    128: "gsr_latent_space_128.npz",
    256: "gsr_latent_space_256.npz",
    512: "gsr_latent_space_512.npz"
}

low_rank_files =  {
    3: "low_rank_latent_space_3.npz",
    6: "low_rank_latent_space_6.npz",
    96: "low_rank_latent_space_96.npz",
    128: "low_rank_latent_space_128.npz",
    256: "low_rank_latent_space_256.npz",
    512: "low_rank_latent_space_512.npz"
}

phasic_files =  {
    3: "phasic_latent_space_3.npz",
    6: "phasic_latent_space_6.npz",
    96: "phasic_latent_space_96.npz",
    128: "phasic_latent_space_128.npz",
    256:"phasic_latent_space_256.npz",
    512: "phasic_latent_space_512.npz"
}

sparse_files =  {
    3: "sparse_latent_space_3.npz",
    6: "sparse_latent_space_6.npz",
    96: "sparse_latent_space_96.npz",
    128: "sparse_latent_space_128.npz",
    256:"sparse_latent_space_256.npz",
    512: "sparse_latent_space_512.npz"
}

tonic_files =  {
    3: "tonic_latent_space_3.npz",
    6: "tonic_latent_space_6.npz",
    96: "tonic_latent_space_96.npz",
    128: "tonic_latent_space_128.npz",
    256:"tonic_latent_space_256.npz",
    512: "tonic_latent_space_512.npz"
}

data_files = {
    'phasic': phasic_files,
    'tonic': tonic_files,
    'gsr': gsr_files,
    'low_rank': low_rank_files
}

models_name = {
    'rbf': 'RBF SVM',
    'svm': 'SVM',
    'knn': 'K-NN',
    'xgboost': 'Xgboost'
}

csv_dataset = 'gsr_data/contractive/gsr_data.csv'
data_df = pd.read_csv(csv_dataset, skiprows=1)
data_df = data_df[13000:]# skip the first 13k entries as they are exploratory
signal = data_df['uS'].to_numpy()
ground_truth = data_df['label'].to_numpy()

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
output_path = Path('outputs/results_{}'.format(timestamp))

if not output_path.exists():
    output_path.mkdir(parents=True)

all_models = dict()
for data_type, files in data_files.items():
    ls_df = list()
    for k, v in files.items():
        models, results_df = compile_grid_search(folder, k, v, s_type=data_type)
        if not (models and results_df):
            continue
        res_path = output_path.joinpath('{}_results_{}.csv'.format(data_type, k))
        all_models |= models
        results_df.to_csv(str(res_path))
        results_df['latent_space'] = k
        results_df['full_model'] = results_df.index
        results_df['grid_search'] = results_df['full_model'].apply(lambda x: x.split("-")[2])
        results_df['model'] = results_df['full_model'].apply(lambda x: x.split("-")[3])
    
        ls_df.append(results_df)

    merged_df = pd.concat(ls_df, ignore_index=True)
    merged_df.dropna(inplace=True)
    spec_acc_df = merged_df[merged_df["grid_search"] == "specificity"]
    bal_acc_df = merged_df[merged_df["grid_search"] == "balanced_accuracy"]

    best_model = merged_df[merged_df['balanced_accuracy'] == merged_df['balanced_accuracy'].max()]
    plot_bar(spec_acc_df, "balanced_accuracy", "Balanced Accuracy", " {} Latent Space - Train session 1&2, Test Session 3, Grid Search Metric - Speicificity".format(data_type))
    plot_bar(bal_acc_df, "balanced_accuracy", "Balanced Accuracy", " {} Latent Space - Train session 1&2, Test Session 3, Grid Search Metric - Balanced Accuracy".format(data_type))
    
    clf = best_model['full_model'].iloc[0]
    latent_space_file = best_model['latent_space'].iloc[0]
    title = "Predictions with {}-Dimension {} Latent Space {} model".format(latent_space_file, ls_type[data_type], models_name[best_model['model'].iloc[0]])
    cm_on_signal(all_models[clf], folder + files[latent_space_file], signal, ground_truth, title)
