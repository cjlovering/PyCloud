import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate
import sklearn.ensemble

from typing import Callable, List

def get_models() -> List:
  """ Returns the list of models to evaluate. """
  return [
    AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor, LinearSVR, DecisionTreeRegressor
  ]

def get_model_names() -> List:
  """Returns the list of model names used. """
  return [
    'AdaBoostRegressor', 'BaggingRegressor', 'GradientBoostingRegressor', 'RandomForestRegressor', 'LinearSVR', 'DecisionTreeRegressor'
  ]

def train(model_class: Callable, X: np.ndarray, y: np.ndarray) -> Callable:
  """Trains a model of the given class on the data. """
  model = model_class()
  model.fit(X, y)
  return model

def evaluate(trained_model: Callable, X: np.ndarray, y_true: np.ndarray) -> float:
  """Evaluates the trained model on the data. """
  y_predict = trained_model.predict(X)
  score = explained_variance_score(y_true, y_predict)
  return score


def main():
  """Runs different regressors on the dataset.

  Sample Output:
    AdaBoostRegressor    BaggingRegressor    GradientBoostingRegressor    RandomForestRegressor    LinearSVR    DecisionTreeRegressor
    -------------------  ------------------  ---------------------------  -----------------------  -----------  -----------------------
              0.995141            0.988741                     0.991962                  0.99019   -0.0626148                 0.992962
  """
  X_raw = pd.read_csv('data/X.csv')
  y_raw = pd.read_csv('data/y.csv')

  X = X_raw.iloc[:,1:].iloc[:,:-1].values
  y = y_raw.iloc[:,1:].values.ravel()

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

  models = get_models()
  trained_models = [train(model, X_train, y_train) for model in models]
  accuracies = [evaluate(model, X_test, y_test) for model in trained_models]
  print(tabulate([accuracies], get_model_names()))

if __name__ == '__main__':
  main()
