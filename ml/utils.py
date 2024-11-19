import numpy as np
import pandas as pd
import time
from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score, make_scorer

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier


labels_for_cm = ['Pain \nRelieving Massage', 'Non Pain \nRelieving Massage']
pos_label = 3

def specificity_scoring(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).flatten()
    specificity = tn / (tn+fp)
    return specificity

def grid_search(X, y, clf, param_grid,*, cv=5, scoring='f1'):
    """
    Perform a grid search to find the best hyperparameters for a given classifier.

    Parameters:
    - X (array-like): Feature matrix.
    - y (array-like): Target labels.
    - clf (estimator): Classifier instance.
    - param_grid (dict): Dictionary of hyperparameter values to search.
    - cv (int, optional): Number of cross-validation folds. Default is 10.
    - scoring (string, optional): Scoring type to use, i.e Precision, Recall, F-1

    Returns:
    GridSearchCV: Grid search results.
    """
    total_search = 0
    for p in param_grid:
        s = 1
        for k, v in p.items():
            s *= len(v)
        total_search += s
    if scoring == 'specificity':
        scoring = make_scorer(specificity_scoring, greater_is_better=True)
    gs = GridSearchCV(clf, cv=cv, param_grid=param_grid, scoring=scoring)
    print("Starting Grid Search with {} settings...".format(total_search))
    start = time.time()
    gs.fit(X, y)
    print("Finished in {:0.2f}".format(time.time() - start))
    return gs

def top_results(gs_results, top_n=3):
    """
    Get the top N results from grid search.

    Parameters:
    - gs_results (dict): Results from grid search.
    - top_n (int, optional): Number of top results to retrieve. Default is 3.

    Returns:
    list: List of top N results, each containing parameters, rank, and mean test score.
    """
    results = [(gs_results['params'][i], gs_results['rank_test_score'][i], gs_results['mean_test_score'][i]) for i in range(len(gs_results['params']))]
    return sorted(results, key=lambda x: x[1])[0: top_n]

def perform_classifier_gs(clf, param_grid, X, y, scoring='f1'):
    """
    Perform a grid search using Stochastic Gradient Descent (SGD) with SVM.

    Parameters:
    - clf: Classfier
    - param_grid: Param grid to be passed on for grid search.
    - X (array-like): Feature matrix.
    - y (array-like): Target labels.
    - scoring (string): Scoring to be used for grid search, default is recall

    Returns:
    float: The best accuracy achieved.
    """
    gs = grid_search(X, y, clf, param_grid, scoring=scoring)
    # sort the reults based on accuracy
    top_n = top_results(gs.cv_results_)
    # pick the best parameter
    pprint(top_n)
    # return best the accuracy
    return top_n[0]

# Created a common function to plot confusion matrix
def plot_Confusion_Matrix(y_test, pred_test, title='Confusion Matrix - Test Data'):
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
    cm = np.array([[tp, fp], [fn, tn]])
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    plt.xticks([0.5, 1.5], labels_for_cm)
    plt.yticks([0.5, 1.5], labels_for_cm)
    plt.yticks(rotation=0)
    plt.title(title)
    plt.show()


def get_data_random_split(file_name):
  data = np.load(file_name)
  X, Y = data['latent_space'], data['labels']
  X = X.squeeze(2)
  idx_X = np.where(Y != 1)
  filter_X = X[idx_X]
  filter_Y = Y[idx_X]
  test_size = 0.2
  X_train, X_test, y_train, y_test = train_test_split(filter_X, filter_Y, test_size=test_size, random_state=42)
  return X_train, X_test, y_train, y_test

def boundary_indexes(labels):
  res = list()
  for i in range(0, labels.shape[0]-1):
    if labels[i] == labels[i-1] and labels[i] != labels[i+1]:
      if not(labels[i+1] == 3 and labels[i] == 2):
        res.append(i+1)
  
  return np.array(res)

def boundary_indexesv2(labels):
  res = list()
  for i in range(0, labels.shape[0]-1):
    if labels[i] == labels[i-1] and labels[i] != labels[i+1]:
        res.append(i+1)

  return np.array(res)

def get_data_seq(file_name, tr):
  data = np.load(file_name)
  X, Y = data['latent_space'], data['labels']
  X = X.squeeze(2)
  bounds = boundary_indexesv2(Y)

  bounds = bounds.reshape((-1, 3))
  print(bounds.shape)
  # bounds = bounds.reshape((-1, 2))
  # X_train = np.vstack([X[bounds[0, 0]: bounds[0, 1]], X[bounds[1, 0]: bounds[1, 1]]])
  # Y_train = np.concatenate([Y[bounds[0, 0]: bounds[0, 1]], Y[bounds[1, 0]: bounds[1, 1]]])

  # X_test = np.vstack([X[bounds[2, 0]: bounds[2, 1]]])
  # Y_test = np.concatenate([Y[bounds[2, 0]: bounds[2, 1]]])

  train_pos, train_neg, test_pos, test_neg = list(), list(), list(), list()
  test_X, test_Y = list(), list()
  npr_size = 0
  for i in range(3):

    npr_start, npr_end = bounds[i][0], bounds[i][1]
    npr_size = npr_end - npr_start
    pr_start, pr_end = bounds[i][1], bounds[i][2]
    print("****************", bounds[i])
    if i in tr:
      print(i, npr_start, npr_end, pr_start, pr_end)
      train_neg.append(X[npr_start:npr_end])
      test_neg.append(Y[npr_start:npr_end])
      train_pos.append(X[pr_start:pr_end])
      test_pos.append(Y[pr_start:pr_end])
    else:
      test_X.append(X[npr_start:npr_end])
      test_Y.append(Y[npr_start:npr_end])
      pr_start, pr_end = bounds[i][1], bounds[i][2]
      test_X.append(X[pr_start:pr_end])
      test_Y.append(Y[pr_start:pr_end])


  # X_train = np.vstack(train_X)
  # Y_train = np.concatenate(train_Y)

  X_test = np.vstack(test_X)
  Y_test = np.concatenate(test_Y)



  rng = np.random.default_rng(seed=42)
  train_pos, train_neg = np.vstack(train_pos), np.vstack(train_neg)
  train_pos_label, train_neg_label = np.concatenate(test_pos), np.concatenate(test_neg)

  pos_indexes = rng.choice(train_pos.shape[0], size=train_neg.shape[0], replace=False)
  pos_data = train_pos[pos_indexes]
  pos_label = train_pos_label[pos_indexes]

  X = np.vstack([pos_data, train_neg])
  Y = np.concatenate([pos_label, train_neg_label])


  return X, X_test, Y, Y_test

def get_data_seq_samples(file_name, tr, t):
  data = np.load(file_name)
  X, Y = data['latent_space'], data['labels']
  print(X.shape, Y.shape)
  X = X.squeeze(2)

  bounds = boundary_indexes(Y)

  bounds = bounds.reshape((-1, 2))

  X_train = np.vstack([X[bounds[tr[0], 0]: bounds[tr[0], 1]], X[bounds[tr[1], 0]: bounds[tr[1], 1]]])
  Y_train = np.concatenate([Y[bounds[tr[0], 0]: bounds[tr[0], 1]], Y[bounds[tr[1], 0]: bounds[tr[1], 1]]])

  X_test = np.vstack([X[bounds[t, 0]: bounds[t, 1]]])
  Y_test = np.concatenate([Y[bounds[t, 0]: bounds[t, 1]]])

  return X_train, X_test, Y_train, Y_test


def get_accuracy_metrics(clf, x_train, y_train, x_test, y_test):
  if type(clf) == XGBClassifier:
      y_test = LabelBinarizer().fit_transform(y_test)
      y_train = LabelBinarizer().fit_transform(y_train)
      pos_label = 1
      neg_label = 0
  else:
      pos_label = 3
      neg_label = 2

  y_pred = clf.predict(x_test)
  y_pred_train = clf.predict(x_train)

  return [
          # training
          accuracy_score(y_train, y_pred_train),
          balanced_accuracy_score(y_train, y_pred_train),
          recall_score(y_train, y_pred_train, pos_label=pos_label),
          recall_score(y_train, y_pred_train, pos_label=neg_label),
          # training
          accuracy_score(y_test, y_pred),
          balanced_accuracy_score(y_test, y_pred),
          f1_score(y_train, y_pred_train, pos_label=pos_label),
          precision_score(y_train, y_pred_train, pos_label=pos_label),
          precision_score(y_train, y_pred_train, pos_label=neg_label),
          recall_score(y_test, y_pred, pos_label=pos_label),
          recall_score(y_test, y_pred, pos_label=neg_label),
        ]
