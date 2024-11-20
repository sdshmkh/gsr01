# Galvanic Skin Response Pain Relief Prediction

This repository contains code for galvaninc skin response pain relief massage predictions.

## Installation

Check out the repository locally with 
```
git clone git@github.com:sdshmkh/gsr01.git
```

Next, install all the required packaged with the conda file provided, to do this run this from the root of the repository,
```
conda env create -f environment.yaml
conda activate gsr
```

## Running the code

First, train autoencoders to create latent spaces of varying dimensions for 1 second windows. To do this run

```
python -m scripts.generate_latent_spaces
```

This will create a new output directory and store the generated latent spaces for the GSR, Tonic, Phasic and Low Rank GSR signal. 

Next, run this after the previuos step to create a grid search with classical ML models like, K-Nearest Neighbours, Support Vector Machines, Linear and Radial Basis Function kernels and Xgboost. To do this run

```
python -m scripts.ml_grid_search
```

This should output results in the output folder as stated above. The plots for visualizing the results are availabe in the viz folder. 