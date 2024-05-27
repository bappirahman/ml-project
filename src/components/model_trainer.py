import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.metrics import r2_score


from src.utils import save_object, evaluate_model
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()

  def initiate_model_trainer(self, train_array, test_array):
    try:
      logging.info('Split training and test input data')
      X_train, y_train, X_test, y_test = (
          train_array[:,:-1],
          train_array[:,-1],
          test_array[:,:-1],
          test_array[:,-1]
      )
      model_hyperparameters = {
    'CatBoostRegressor': {
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 6
    },
    'XGBRegressor': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3
    },
    'LogisticRegression': {
        'C': 1.0,
        'solver': 'lbfgs'
    },
    'KNeighborsClassifier': {
        'n_neighbors': 5,
        'weights': 'uniform'
    },
    'SVC': {
        'C': 1.0,
        'kernel': 'rbf'
    },
    'DecisionTreeClassifier': {
        'max_depth': None,
        'min_samples_split': 2
    },
    'GaussianNB': {},
    'AdaBoostClassifier': {
        'n_estimators': 50,
        'learning_rate': 1.0
    },
    'BaggingClassifier': {
        'n_estimators': 10,
        'max_samples': 1.0
    },
    'ExtraTreesClassifier': {
        'n_estimators': 100,
        'criterion': 'gini'
    },
    'GradientBoostingClassifier': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3
    },
    'RandomForestClassifier': {
        'n_estimators': 100,
        'criterion': 'gini'
    },
}

# Create instances of models with hyperparameters
      models = {
          'CatBoostRegressor': CatBoostRegressor(**model_hyperparameters['CatBoostRegressor']),
          'XGBRegressor': XGBRegressor(**model_hyperparameters['XGBRegressor']),
          'LogisticRegression': LogisticRegression(**model_hyperparameters['LogisticRegression']),
          'KNeighborsClassifier': KNeighborsClassifier(**model_hyperparameters['KNeighborsClassifier']),
          'SVC': SVC(**model_hyperparameters['SVC']),
          'DecisionTreeClassifier': DecisionTreeClassifier(**model_hyperparameters['DecisionTreeClassifier']),
          'GaussianNB': GaussianNB(**model_hyperparameters['GaussianNB']),
          'AdaBoostClassifier': AdaBoostClassifier(**model_hyperparameters['AdaBoostClassifier']),
          'BaggingClassifier': BaggingClassifier(**model_hyperparameters['BaggingClassifier']),
          'ExtraTreesClassifier': ExtraTreesClassifier(**model_hyperparameters['ExtraTreesClassifier']),
          'GradientBoostingClassifier': GradientBoostingClassifier(**model_hyperparameters['GradientBoostingClassifier']),
          'RandomForestClassifier': RandomForestClassifier(**model_hyperparameters['RandomForestClassifier']),
      }

      # Train and evaluate models
      model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

      best_model_score = max(sorted(model_report.values()))

      best_model_name = list(model_report.keys())[
          list(model_report.values()).index(best_model_score)
      ]

      if best_model_score < 0.6:
        raise CustomException('No best model found')

      best_model = models[best_model_name]

      logging.info(f'Best model found on both training and testing dataset: {best_model_name}')

      save_object(
          file_path=self.model_trainer_config.trained_model_file_path,
          obj=best_model
      )
      pred = best_model.predict(X_test)
      r2_square = r2_score(y_test, pred)
      return r2_square

    except Exception as e:
      raise CustomException(e, sys)