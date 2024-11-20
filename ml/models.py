import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import  LabelBinarizer

from sklearn.metrics import  accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.linear_model import  SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .utils import perform_classifier_gs, plot_Confusion_Matrix, labels_for_cm, pos_label, get_accuracy_metrics, get_data_seq_samples

def viz_plots():
  return False

def SVMModels(X_train,y_train, X_test, y_test, scoring='accuracy'):

    param_grid = [{
        "loss": ['hinge'],
        "penalty": ['l1',],
        "alpha": [0.0001, 0.0005, 0.001, 0.01],
        "l1_ratio": [0.15, 0.5],
        "tol": [0.0001, 0.001],
        "shuffle": [True],
    }]

    clf = SGDClassifier()
    params = perform_classifier_gs(clf, param_grid, X_train, y_train, scoring=scoring)[0]



    # Evaluate SVM model with linear kernel  model with Stochastic SVM
    clf = SGDClassifier(**params, random_state=42)
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_train)
    SVM_Score = accuracy_score(y_train, y_pred)
    print("train accuracy_score : {0}".format(SVM_Score))
    print("train Confusion Matrix")
    
    if viz_plots():
      plot_Confusion_Matrix(y_train, y_pred, 'Train Linear SVM')
      if len(labels_for_cm) < 3:
        RocCurveDisplay.from_estimator(clf, X_train, y_train, name="Train RBF SVM", pos_label=pos_label)
        plt.show()


    y_pred = clf.predict(X_test)
    SVM_Score = accuracy_score(y_test, y_pred)
    print("accuracy_score : {0}".format(SVM_Score))
    print("Confusion Matrix")
    
    if viz_plots():
      plot_Confusion_Matrix(y_test, y_pred, 'Linear SVM')
      if len(labels_for_cm) < 3:
        RocCurveDisplay.from_estimator(clf, X_train, y_train, name="Train RBF SVM", pos_label=pos_label)
        plt.show()

    return clf, params

def Radial_SVMModels(X_train,y_train, X_test, y_test, scoring='accuracy'):

    param_grid = [{
      "C": [0.01, 1,10],
      "gamma": ['auto', 'scale', 0.1, 0.001, 10, 100],
      "kernel": ['rbf'],
      "class_weight": ['balanced', None, {2: 1, 3: 15}],
    }]
    # param_grid = [{
    #   "C": [0.001,],
    #   "gamma": ['scale'],
    #   "kernel": ['rbf'],
    #   "max_iter": [10000],
    #   "class_weight": ['balanced']
    # }]
    clf = SVC()

    params = perform_classifier_gs(clf, param_grid, X_train, y_train, scoring=scoring)[0]
    # phasic and normal - params = {'C': 30000, 'class_weight': None, 'gamma': 15000, 'kernel': 'rbf'}
    # params = {'C': 30000, 'class_weight': None, 'gamma': 15000, 'kernel': 'rbf'}



    # Evaluate SVM model with linear kernel  model with Stochastic SVM
    clf = SVC(**params, random_state=42)
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_train)
    SVM_Score = accuracy_score(y_train, y_pred)
    print("train accuracy_score : {0}".format(SVM_Score))
    print("train Confusion Matrix")
    
    if viz_plots():
      plot_Confusion_Matrix(y_train, y_pred, 'Train RBF SVM')
      if len(labels_for_cm) < 3:
        RocCurveDisplay.from_estimator(clf, X_train, y_train, name="Training RBF SVM", pos_label=pos_label)
        plt.show()

    y_pred = clf.predict(X_test)
    SVM_Score = accuracy_score(y_test, y_pred)
    print("accuracy_score : {0}".format(SVM_Score))
    print("Confusion Matrix")
    
    if viz_plots():
      plot_Confusion_Matrix(y_test, y_pred, 'RBF SVM Testing')
      if len(labels_for_cm) < 3:
        RocCurveDisplay.from_estimator(clf, X_test, y_test, name="Testing RBF SVM", pos_label=pos_label)
        plt.show()

    return clf, params

def XGBoostModels(X_train,y_train, X_test, y_test, scoring='accuracy'):

    param_grid = [{
      "learning_rate": [0.001,],
      "n_estimators": [200],
      "max_depth": [12],
      "min_child_weight": [14, 21],
      "gamma": [0.001],
      "subsample": [0.6, 0.3],
      # "colsample_bytree": [0.6, 0.5, 0.3],
      # "reg_alpha": [0.1, 0.001, 0.01],

    }]


    clf = XGBClassifier(random_state=42)
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)


    params = perform_classifier_gs(clf, param_grid, X_train, y_train, scoring=scoring)[0]



    XGBmodel = XGBClassifier(random_state=42)
    XGBmodel.fit(X_train, y_train)
    y_pred = XGBmodel.predict(X_test)

    # y_pred = np.where(y_pred == 1, 3, 2)
    # y_test = np.where(y_test == 1, 3, 2)

    print("Testing Confusion Matrix")
    if viz_plots():
      plot_Confusion_Matrix(y_test, y_pred, 'XgBoost')
      RocCurveDisplay.from_estimator(XGBmodel, X_test, y_test, name="Xgboost", pos_label=1)

    XG_Score = accuracy_score(y_test, y_pred)
    print("accuracy_score : {0}".format(XG_Score))


    return XGBmodel, params

def KNNModels(X_train, y_train, X_test, y_test, scoring='accuracy'):

    #create KNN model and fit the model with train dataset
    clf = KNeighborsClassifier()

    # Define the hyperparameter grid
    param_grid = [{
        'n_neighbors': [12, 24, 48, 96],
        'weights': ['uniform', 'distance'],
        'metric': ['cosine', 'minkowski'],
        'p': [1, 2],
        "n_jobs":[16]
    }]

    params = perform_classifier_gs(clf, param_grid, X_train, y_train, scoring=scoring)[0]


    knn = KNeighborsClassifier(**params)
    knn.fit(X_train, y_train)

    # Accuracy
    y_pred = knn.predict(X_train)
    knn_Score = accuracy_score(y_train, y_pred)
    print("training accuracy_score : {0}".format(knn_Score))
    print("Training Confusion Matrix")
    
    if viz_plots():
      plot_Confusion_Matrix(y_train, y_pred, 'Knn Training')
      if len(labels_for_cm) < 3:
        RocCurveDisplay.from_estimator(knn, X_train, y_train, name="Train KNN", pos_label=pos_label)
        plt.show()

    y_pred = knn.predict(X_test)
    knn_Score = accuracy_score(y_test, y_pred)
    print("accuracy_score : {0}".format(knn_Score))
    print("Confusion Matrix")
    
    if viz_plots():
      plot_Confusion_Matrix(y_test, y_pred, 'Knn Testing')
      if len(labels_for_cm) < 3:
        RocCurveDisplay.from_estimator(knn, X_test, y_test, name="Test KNN", pos_label=pos_label)
        plt.show()

    return knn,params


scoring = ['balanced_accuracy', 'specificity']
cols = ['model', 'training_accuracy', 'training_balanced_accuracy', 'training_recall_pos', 'training_recall_neg', 'accuracy', 'balanced_accuracy', 'f1', 'precision_pos', 'precision_neg', 'recall_pos', 'recall_neg']
classfiers = {'knn':KNNModels, 'rbf':Radial_SVMModels, 'svm':SVMModels, 'xgboost': XGBoostModels}

def compile_grid_search(folder, k, file_name, s_type=''):
  models = dict()
  file_name = folder + file_name

  if not Path(file_name).exists():
    return None, None
  
  X_train, X_test, y_train, y_test = get_data_seq_samples(file_name, [0, 1], 2)
  res = list()
  for score in scoring:
    for name, model_cls in classfiers.items():
      print("*"*20, "Starting - {}".format(str(k)+ " " + score + " " + name), "*"*20)
      clf, _ = model_cls(X_train, y_train, X_test, y_test, scoring=score)
      clf_key = "{}-{}-{}-{}".format(s_type, k, score, name)
      models[clf_key] = clf
      res.append(([clf_key] + get_accuracy_metrics(clf, X_train, y_train, X_test, y_test)))
      print("*" * 20, "Finished - {}".format(score + " " + name), "*" * 0)
      
  df = pd.DataFrame(res, columns=cols)
  df.set_index('model', inplace=True)
  return models, df