import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_log_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
import sklearn.ensemble

from sklearn.model_selection import validation_curve

from typing import Callable, List, Tuple

def get_models() -> List:
  """ Returns the list of models to evaluate. """
  return [
    LinearRegression, LinearSVR, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor, DecisionTreeRegressor
  ]

def get_model_names() -> List:
  """Returns the list of model names used. """
  return [
    'LinearRegression', 'LinearSVR', 'AdaBoostRegressor', 'BaggingRegressor', 'GradientBoostingRegressor', 'RandomForestRegressor', 'DecisionTreeRegressor'
  ]

def train(model_class: Callable, X: np.ndarray, y: np.ndarray) -> Callable:
  """Trains a model of the given class on the data. """
  model = model_class()
  model.fit(X, y)
  return model

def evaluate(trained_model: Callable, X: np.ndarray, y_true: np.ndarray) -> float:
  """Evaluates the trained model on the data. """
  y_predict = trained_model.predict(X)
  explained_variance = explained_variance_score(y_true, y_predict)
  error = mean_squared_error(y_true, y_predict)
  return explained_variance, error

def load_data() -> Tuple[np.ndarray,  np.ndarray, np.ndarray, np.ndarray]:
  """Loads and formats data from csv files. """
  X_raw = pd.read_csv('data/X.csv')
  y_raw = pd.read_csv('data/y.csv')

  X = X_raw.iloc[:,1:].iloc[:,:-1].values
  y = y_raw.iloc[:,1:].values.ravel()

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
  return X_train, X_test, y_train, y_test

def load_data_graph() -> Tuple[np.ndarray, np.ndarray]:
  """Loads and formats data from csv files. """
  X_raw = pd.read_csv('data/X.csv')
  y_raw = pd.read_csv('data/y.csv')

  X = X_raw.iloc[:,1:].values
  y = y_raw.iloc[:,1:].values.ravel()

  return X, y

def print_latex_table(scores: List[Tuple[float, float]]) -> None:
  """Prints results in a format for a latex table :) """
  for model_name, (explained_variance, error) in list(zip(get_model_names(), scores)):
    print("{} & {} & {} \\\\".format(
      model_name, 
      "{0:.3f}".format(explained_variance), 
      "{0:.3f}".format(error)))

def main():
  """Runs different regressors on the dataset.
  """

  X_train, X_test, y_train, y_test = load_data()
  models = get_models()
  trained_models = [train(model, X_train, y_train) for model in models]
  scores = [evaluate(model, X_test, y_test) for model in trained_models]
  print_latex_table(scores)

if __name__ == '__main__':
  main()
