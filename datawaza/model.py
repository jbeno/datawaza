# model.py – Model module of Datawaza
#
# Datawaza  Copyright (C) 2024  Jim Beno
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details:
# https://github.com/jbeno/datawaza/blob/main/LICENSE
"""
This module provides tools to streamline data modeling workflows. It contains
functions to set up pipelines, iterate over models, and evaluate and plot results.

Functions:
    - :func:`~datawaza.model.compare_models` - Find the best classification model and hyper-parameters for a dataset.
    - :func:`~datawaza.model.create_nn_binary` - Create a binary classification neural network model.
    - :func:`~datawaza.model.create_nn_multi` - Create a multi-class classification neural network model.
    - :func:`~datawaza.model.create_pipeline` - Create a custom pipeline for data preprocessing and modeling.
    - :func:`~datawaza.model.create_results_df` - Initialize the results_df DataFrame with the columns required for `iterate_model`.
    - :func:`~datawaza.model.eval_model` - Produce a detailed evaluation report for a classification model.
    - :func:`~datawaza.model.iterate_model` - Iterate and evaluate a model pipeline with specified parameters.
    - :func:`~datawaza.model.plot_acf_residuals` - Plot residuals, histogram, ACF, and PACF of a time series ARIMA model.
    - :func:`~datawaza.model.plot_results` - Plot the results of model iterations and select the best metric.
    - :func:`~datawaza.model.plot_train_history` - Plot the training and validation history of a fitted Keras model.
"""

# Metadata
__author__ = "Jim Beno"
__email__ = "jim@jimbeno.net"
__version__ = "0.1.3"
__license__ = "GNU GPLv3"

# Standard library imports
import os
from datetime import datetime
import time
import math

# Data manipulation and analysis
import numpy as np
import pandas as pd
import pytz

# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Scikit-learn imports
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor, VotingRegressor)
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression, Ridge)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, precision_recall_curve, PrecisionRecallDisplay,
                             roc_auc_score, make_scorer, precision_score, recall_score, f1_score, accuracy_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   PolynomialFeatures, RobustScaler, StandardScaler)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# XGBoost
from xgboost import XGBClassifier

# Statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Imbalanced learn - Package: imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Miscellaneous imports
from joblib import dump

# Local Datawaza helper function imports
from datawaza.tools import calc_pfi, calc_vif, extract_coef, log_transform, thousands, DebugPrinter, model_summary

# Typing imports
from typing import Optional, Union, Tuple, List, Dict, Any

# TensorFlow and Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warning on import
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.regularizers import L2

# Functions
def compare_models(
        x: pd.DataFrame,
        y: pd.Series,
        models: List[str],
        config: Dict[str, Any],
        class_map: Optional[Dict[Any, Any]] = None,
        pos_label: Optional[Any] = None,
        test_size: float = 0.25,
        search_type: str = 'grid',
        grid_cv: Union[int, str] = 5,
        plot_perf: bool = False,
        scorer: str = 'accuracy',
        random_state: int = 42,
        decimal: int = 4,
        verbose: int = 4,
        title: Optional[str] = None,
        fig_size: Tuple[int, int] = (12, 6),
        figmulti: float = 1.5,
        multi_class: str = 'ovr',
        average: str = None,
        legend_loc: str = 'best',
        model_eval: bool = False,
        svm_proba: bool = False,
        threshold: float = 0.5,
        class_weight: Optional[Dict[Any, float]] = None,
        stratify: Optional[pd.Series] = None,
        imputer: Optional[str] = None,
        impute_first: bool = True,
        transformers: Optional[List[str]] = None,
        scaler: Optional[str] = None,
        selector: Optional[str] = None,
        cat_columns: Optional[List[str]] = None,
        num_columns: Optional[List[str]] = None,
        max_iter: int = 10000,
        rotation: Optional[int] = None,
        plot_curve: bool = True,
        under_sample: Optional[float] = None,
        over_sample: Optional[float] = None,
        notes: Optional[str] = None,
        svm_knn_resample: Optional[float] = None,
        n_jobs: Optional[int] = None,
        output: bool = True,
        timezone: str = 'UTC',
        debug: bool = False
) -> pd.DataFrame:
    """
    Find the best classification model and hyper-parameters for a dataset by
    automating the workflow for multiple models and comparing results.

    This function integrates a number of steps in a typical classification model
    workflow, and it does this for multiple models, all with one command line:

    * Auto-detecting single vs. multi-class classification problems
    * Option to Under-sample or Over-smple imbalanced data,
    * Option to use a sub-sample of data for SVC or KNN, which can be computation
      intense
    * Ability to split the Train/Test data at a specified ratio,
    * Creation of a multiple-step Pipeline, including Imputation, multiple Column
      Transformer/Encoding steps, Scaling, Feature selection, and the Model,
    * Grid Search of hyper-parameters, either full or random,
    * Calculating performance metrics from the standard Classification Report
      (Accuracy, Precision, Recall, F1) but also with ROC AUC, and if binary, True
      Positive Rate, True Negative Rate, False Positive Rate, False Negative Rate,
    * Evaluating this performance based on a customizable Threshold,
    * Visually showing performance by plotting (a) a Confusion Matrix, and if
      binary, (b) a Histogram of Predicted Probabilities, (c) an ROC Curve, and
      (d) a Precision-Recall Curve.
    * Save all the results in a DataFrame for reference and comparison, and
    * Option to plot the results to visually compare performance of the specified
      metric across multiple model pipelines with their best parameters.

    To use this function, a configuration should be created that defines the
    desired model configurations and parameters you want to search.
    When `compare_models` is run, for each model in the `models` parameter, the
    `create_pipeline` function will be called to create a pipeline from the
    specified parameters. Each model iteration will have the same pipeline
    construction, except for the final model, which will vary. Here are the major
    pipeline parameters, along with the config sections they map to:

    * `imputer` (str) is selected from `config['imputers']`
    * `transformers` (list or str) are selected from `config['transformers']`
    * `scaler` (str) is selected from `config['scalers']`
    * `selector` (str) is selected from `config['selectors']`
    * `models` (list or str) are selected from `config['models']`

    Here is an example of the configuration dictionary structure. It is based on
    what `create_pipeline` requires to assemble the pipeline. But it adds some
    additional configuration parameters referenced by `compare_models`, which
    are `params` (grid search parameters, required) and `cv` (cross-validation
    parameters, optional if `grid_cv` is an integer). The configuration dictionary
    is passed to `compare_models` as the `config` parameter:

    >>> config = {  # doctest: +SKIP
    ...     'models' : {
    ...         'logreg': LogisticRegression(max_iter=max_iter,
    ...                   random_state=random_state, class_weight=class_weight),
    ...         'knn_class': KNeighborsClassifier(),
    ...         'tree_class': DecisionTreeClassifier(random_state=random_state,
    ...                       class_weight=class_weight)
    ...     },
    ...     'imputers': {
    ...         'simple_imputer': SimpleImputer()
    ...     },
    ...     'transformers': {
    ...         'ohe': (OneHotEncoder(drop='if_binary', handle_unknown='ignore'),
    ...                     ohe_columns)
    ...     },
    ...     'scalers': {
    ...         'stand': StandardScaler()
    ...     },
    ...     'selectors': {
    ...         'sfs_logreg': SequentialFeatureSelector(LogisticRegression(
    ...                       max_iter=max_iter, random_state=random_state,
    ...                       class_weight=class_weight))
    ...     },
    ...     'params' : {
    ...         'logreg': {
    ...             'logreg__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    ...             'logreg__solver': ['newton-cg', 'lbfgs', 'saga']
    ...         },
    ...         'knn_class': {
    ...             'knn_class__n_neighbors': [3, 5, 10, 15, 20, 25],
    ...             'knn_class__weights': ['uniform', 'distance'],
    ...             'knn_class__metric': ['euclidean', 'manhattan']
    ...         },
    ...         'tree_class': {
    ...             'tree_class__max_depth': [3, 5, 7],
    ...             'tree_class__min_samples_split': [5, 10, 15],
    ...             'tree_class__criterion': ['gini', 'entropy'],
    ...             'tree_class__min_samples_leaf': [2, 4, 6]
    ...         },
    ...     },
    ...     'cv': {
    ...         'kfold_5': KFold(n_splits=5, shuffle=True, random_state=42)
    ...     },
    ...     'no_scale': ['tree_class'],
    ...     'no_poly': ['knn_class', 'tree_class']
    ... }

    In addition to the configuration file, you will need to define any column
    lists if you want to target certain transformations to a subset of columns.
    For example, you might define a 'ohe' transformer for One-Hot Encoding, and
    reference 'ohe_columns' or 'cat_columns' in its definition in the config.

    Here is an example of how to call this function in an organized manner:

    >>> results_df = dw.compare_models(  # doctest: +SKIP
    ...
    ...     # Data split and sampling
    ...     x=X, y=y, test_size=0.25, stratify=None, under_sample=None,
    ...     over_sample=None, svm_knn_resample=None,
    ...
    ...     # Models and pipeline steps
    ...     imputer=None, transformers=None, scaler='stand', selector=None,
    ...     models=['logreg', 'knn_class', 'svm_proba', 'tree_class',
    ...     'forest_class', 'xgb_class', 'keras_class'], svm_proba=True,
    ...
    ...     # Grid search
    ...     search_type='random', scorer='accuracy', grid_cv='kfold_5', verbose=4,
    ...
    ...     # Model evaluation and charts
    ...     model_eval=True, plot_perf=True, plot_curve=True, fig_size=(12,6),
    ...     legend_loc='lower left', rotation=45, threshold=0.5,
    ...     class_map=class_map, pos_label=1, title='Breast Cancer',
    ...
    ...     # Config, preferences and notes
    ...     config=my_config, class_weight=None, random_state=42, decimal=4,
    ...     n_jobs=None, debug=False, notes='Test Size=0.25, Threshold=0.50'
    ... )

    Use this function when you want to find the best classification model and
    hyper-parameters for a dataset, after doing any required pre-processing or
    cleaning. It is a significant time saver, replacing numerous manual coding
    steps with one command.

    Parameters
    ----------
    x : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The target vector.
    test_size : float, optional (default=0.25)
        The proportion of the dataset to include in the test split.
    models : List[str]
        A list of model names to iterate over.
    config : Dict[str, Any]
        A configuration dictionary that defines the pipeline steps, models,
        grid search parameters, and cross-validation functions. It should have
        the following keys: 'imputers', 'transformers', 'scalers', 'selectors',
        'models', 'params', 'cv', 'no_scale', and 'no_poly'.
    class_map : Dict[Any, Any], optional (default=None)
        A dictionary to map class labels to new values.
    search_type : str, optional (default='grid')
        The type of hyperparameter search to perform. Can be either 'grid'
        for GridSearchCV or 'random' for RandomizedSearchCV.
    grid_cv : Union[int, str], optional (default=5)
        The number of cross-validation folds for GridSearchCV or
        RandomizedSearchCV, or a string to select a cross-validation
        function from config['cv']. Default is 5.
    plot_perf : bool, optional (default=False)
        Whether to plot the model performance.
    scorer : str, optional (default='accuracy')
        The scorer to use for model evaluation.
    pos_label : Any, optional (default=None)
        The positive class label.
    random_state : int, optional (default=42)
        The random state for reproducibility.
    decimal : int, optional (default=4)
        The number of decimal places to round the results to.
    verbose : int, optional (default=4)
        The verbosity level for the search.
    title : str, optional (default=None)
        The title for the plots.
    fig_size : Tuple[int, int], optional (default=(12, 6))
        The figure size for the plots.
    figmulti : float, optional (default=1.5)
        The multiplier for the figure size in multi-class classification.
    multi_class : str, optional
        The method for handling multi-class ROC AUC calculation.
        Can be 'ovr' (one-vs-rest) or 'ovo' (one-vs-one).
        Default is 'ovr'.
    average : str, optional
        The averaging method for multi-class classification metrics.
        Can be 'macro', 'micro', 'weighted', or 'samples'.
        Default is 'macro'.
    legend_loc : str, optional (default='best')
        The location of the legend in the plots.
    model_eval : bool, optional (default=False)
        Whether to perform a detailed model evaluation.
    svm_proba : bool, optional (default=False)
        Whether to enable probability estimates for SVC.
    threshold : float, optional (default=0.5)
        The classification threshold for binary classification.
    class_weight : Dict[Any, float], optional (default=None)
        The class weights for balancing imbalanced classes.
    stratify : pd.Series, optional (default=None)
        The stratification variable for train-test split.
    imputer : str, optional (default=None)
        The imputation strategy.
    impute_first : bool, optional (default=True)
        Whether to impute before other preprocessing steps.
    transformers : List[str], optional (default=None)
        A list of transformers to apply.
    scaler : str, optional (default=None)
        The scaling strategy.
    selector : str, optional (default=None)
        The feature selection strategy.
    config : Dict[str, Any], optional (default=None)
        A configuration dictionary for customizing the pipeline.
    cat_columns : List[str], optional (default=None)
        A list of categorical columns in X.
    num_columns : List[str], optional (default=None)
        A list of numerical columns in X.
    max_iter : int, optional (default=10000)
        The maximum number of iterations for the solvers.
    rotation : int, optional (default=None)
        The rotation angle for the x-axis labels in the plots.
    plot_curve : bool, optional (default=True)
        Whether to plot the learning curve for KerasClassifier.
    under_sample : float, optional (default=None)
        The under-sampling ratio.
    over_sample : float, optional (default=None)
        The over-sampling ratio.
    notes : str, optional (default=None)
        Additional notes or comments.
    svm_knn_resample : float, optional (default=None)
        The resampling ratio for SVC and KNeighborsClassifier.
    n_jobs : int, optional (default=None)
        The number of parallel jobs to run.
    output : bool, optional (default=True)
        Whether to print the progress and results.
    timezone : str, optional
        Timezone to be used for timestamps. Default is 'UTC'.
    debug : bool, optional
        Flag to show debugging information.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the performance metrics and other details for
        each model.

    Examples
    --------
    Prepare the data for the examples:

    >>> pd.set_option('display.max_columns', None)  # For test consistency
    >>> pd.set_option('display.width', None)  # For test consistency
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_classes=2, n_features=20,
    ...                            weights=[0.4, 0.6], random_state=42)
    >>> X = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])
    >>> y = pd.Series(y, name='Target')
    >>> class_map = {0: 'Malignant', 1: 'Benign'}

    Example 1: Define the configuration for the models:

    >>> # Set some variables referenced in the config
    >>> random_state = 42
    >>> class_weight = None
    >>> max_iter = 10000
    >>>
    >>> # Set column lists referenced in the config
    >>> num_columns = list(X.columns)
    >>> cat_columns = []
    >>>
    >>> # Create a custom configuration file with 3 models and grid search params
    >>> my_config = {
    ...     'models' : {
    ...         'logreg': LogisticRegression(max_iter=max_iter,
    ...                   random_state=random_state, class_weight=class_weight),
    ...         'knn_class': KNeighborsClassifier(),
    ...         'tree_class': DecisionTreeClassifier(random_state=random_state,
    ...                       class_weight=class_weight),
    ...         'svm_proba': SVC(random_state=random_state, probability=True,
    ...                      class_weight=class_weight),
    ...     },
    ...     'imputers': {
    ...         'simple_imputer': SimpleImputer()
    ...     },
    ...     'transformers': {
    ...         'ohe': (OneHotEncoder(drop='if_binary', handle_unknown='ignore'),
    ...                     cat_columns),
    ...         'poly2': (PolynomialFeatures(degree=2, include_bias=False), num_columns)
    ...     },
    ...     'scalers': {
    ...         'stand': StandardScaler()
    ...     },
    ...     'selectors': {
    ...         'sfs_logreg': SequentialFeatureSelector(LogisticRegression(
    ...                       max_iter=max_iter, random_state=random_state,
    ...                       class_weight=class_weight))
    ...     },
    ...     'params' : {
    ...         'logreg': {
    ...             'logreg__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    ...             'logreg__solver': ['newton-cg', 'lbfgs', 'saga']
    ...         },
    ...         'knn_class': {
    ...             'knn_class__n_neighbors': [3, 5, 10, 15, 20, 25],
    ...             'knn_class__weights': ['uniform', 'distance'],
    ...             'knn_class__metric': ['euclidean', 'manhattan']
    ...         },
    ...         'tree_class': {
    ...             'tree_class__max_depth': [3, 5, 7],
    ...             'tree_class__min_samples_split': [5, 10, 15],
    ...             'tree_class__criterion': ['gini', 'entropy'],
    ...             'tree_class__min_samples_leaf': [2, 4, 6]
    ...         },
    ...         'svm_proba': {
    ...             'svm_proba__C': [0.01, 0.1, 1, 10, 100],
    ...             'svm_proba__kernel': ['linear', 'poly']
    ...         },
    ...     },
    ...     'cv': {
    ...         'kfold_5': KFold(n_splits=5, shuffle=True, random_state=42)
    ...     },
    ...     'no_scale': ['tree_class'],
    ...     'no_poly': ['knn_class', 'tree_class']
    ... }

    Example 1: Compare models with default parameters:

    >>> results_df = compare_models(
    ...
    ...     # Data split and sampling
    ...     x=X, y=y, test_size=0.25, stratify=None, under_sample=None,
    ...     over_sample=None, svm_knn_resample=None,
    ...
    ...     # Models and pipeline steps
    ...     imputer=None, transformers=None, scaler='stand', selector=None,
    ...     models=['logreg', 'knn_class', 'tree_class'], svm_proba=True,
    ...
    ...     # Grid search
    ...     search_type='random', scorer='accuracy', grid_cv='kfold_5', verbose=1,
    ...
    ...     # Model evaluation and charts
    ...     model_eval=True, plot_perf=True, plot_curve=True, fig_size=(12,6),
    ...     legend_loc='lower left', rotation=45, threshold=0.5,
    ...     class_map=class_map, pos_label=1, title='Breast Cancer',
    ...
    ...     # Config, preferences and notes
    ...     config=my_config, class_weight=None, random_state=42, decimal=2,
    ...     n_jobs=None, notes='Test Size=0.25, Threshold=0.50'
    ... )  #doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    <BLANKLINE>
    -----------------------------------------------------------------------------------------
    Starting Data Processing - ... UTC
    -----------------------------------------------------------------------------------------
    <BLANKLINE>
    Classification type detected: binary
    Unique values in y: [0 1]
    <BLANKLINE>
    Train/Test split, test_size:  0.25
    X_train, X_test, y_train, y_test shapes:  (750, 20) (250, 20) (750,) (250,)
    <BLANKLINE>
    -----------------------------------------------------------------------------------------
    1/3: Starting LogisticRegression Random Search - ... UTC
    -----------------------------------------------------------------------------------------
    <BLANKLINE>
    Fitting 5 folds for each of 10 candidates, totalling 50 fits
    <BLANKLINE>
    Total Time: ... seconds
    Average Fit Time: ... seconds
    Inference Time: ...
    Best CV Accuracy Score: 0.88
    Train Accuracy Score: 0.89
    Test Accuracy Score: 0.86
    Overfit: Yes
    Overfit Difference: 0.03
    Best Parameters: {'logreg__solver': 'saga', 'logreg__C': 0.1}
    <BLANKLINE>
    LogisticRegression Binary Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
       Malignant       0.81      0.82      0.81        92
          Benign       0.89      0.89      0.89       158
    <BLANKLINE>
        accuracy                           0.86       250
       macro avg       0.85      0.85      0.85       250
    weighted avg       0.86      0.86      0.86       250
    <BLANKLINE>
    ROC AUC: 0.92
    <BLANKLINE>
                   Predicted:0         1
    Actual: 0                75        17
    Actual: 1                18        140
    <BLANKLINE>
    True Positive Rate / Sensitivity: 0.89
    True Negative Rate / Specificity: 0.82
    False Positive Rate / Fall-out: 0.18
    False Negative Rate / Miss Rate: 0.11
    <BLANKLINE>
    Positive Class: Benign (1)
    Threshold: 0.5
    <BLANKLINE>
    -----------------------------------------------------------------------------------------
    2/3: Starting KNeighborsClassifier Random Search - ... UTC
    -----------------------------------------------------------------------------------------
    <BLANKLINE>
    Fitting 5 folds for each of 10 candidates, totalling 50 fits
    <BLANKLINE>
    Total Time: ... seconds
    Average Fit Time: ... seconds
    Inference Time: ...
    Best CV Accuracy Score: 0.86
    Train Accuracy Score: 1.00
    Test Accuracy Score: 0.84
    Overfit: Yes
    Overfit Difference: 0.16
    Best Parameters: {'knn_class__weights': 'distance', 'knn_class__n_neighbors': 20, 'knn_class__metric': 'manhattan'}
    <BLANKLINE>
    KNeighborsClassifier Binary Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
       Malignant       0.75      0.84      0.79        92
          Benign       0.90      0.84      0.87       158
    <BLANKLINE>
        accuracy                           0.84       250
       macro avg       0.82      0.84      0.83       250
    weighted avg       0.84      0.84      0.84       250
    <BLANKLINE>
    ROC AUC: 0.91
    <BLANKLINE>
                   Predicted:0         1
    Actual: 0                77        15
    Actual: 1                26        132
    <BLANKLINE>
    True Positive Rate / Sensitivity: 0.84
    True Negative Rate / Specificity: 0.84
    False Positive Rate / Fall-out: 0.16
    False Negative Rate / Miss Rate: 0.16
    <BLANKLINE>
    Positive Class: Benign (1)
    Threshold: 0.5
    <BLANKLINE>
    -----------------------------------------------------------------------------------------
    3/3: Starting DecisionTreeClassifier Random Search - ... UTC
    -----------------------------------------------------------------------------------------
    <BLANKLINE>
    Fitting 5 folds for each of 10 candidates, totalling 50 fits
    <BLANKLINE>
    Total Time: ... seconds
    Average Fit Time: ... seconds
    Inference Time: ...
    Best CV Accuracy Score: 0.88
    Train Accuracy Score: 0.93
    Test Accuracy Score: 0.86
    Overfit: Yes
    Overfit Difference: 0.08
    Best Parameters: {'tree_class__min_samples_split': 15, 'tree_class__min_samples_leaf': 6, 'tree_class__max_depth': 5, 'tree_class__criterion': 'entropy'}
    <BLANKLINE>
    DecisionTreeClassifier Binary Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
       Malignant       0.76      0.89      0.82        92
          Benign       0.93      0.84      0.88       158
    <BLANKLINE>
        accuracy                           0.86       250
       macro avg       0.84      0.86      0.85       250
    weighted avg       0.87      0.86      0.86       250
    <BLANKLINE>
    ROC AUC: 0.92
    <BLANKLINE>
                   Predicted:0         1
    Actual: 0                82        10
    Actual: 1                26        132
    <BLANKLINE>
    True Positive Rate / Sensitivity: 0.84
    True Negative Rate / Specificity: 0.89
    False Positive Rate / Fall-out: 0.11
    False Negative Rate / Miss Rate: 0.16
    <BLANKLINE>
    Positive Class: Benign (1)
    Threshold: 0.5
    >>> results_df.head()  #doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
                        Model  Test Size Over Sample Under Sample Resample  Total Fit Time  Fit Count  Average Fit Time  Inference Time Grid Scorer                                        Best Params  Best CV Score  Train Score  Test Score Overfit  Overfit Difference  Train Accuracy Score  Test Accuracy Score  Train Precision Score  Test Precision Score  Train Recall Score  Test Recall Score  Train F1 Score  Test F1 Score  Train ROC AUC Score  Test ROC AUC Score  Threshold  True Positives  False Positives  True Negatives  False Negatives       TPR       FPR       TNR       FNR  False Rate            Pipeline                           Notes Timestamp
    0      LogisticRegression       0.25        None         None     None               ...         50                 ...    Accuracy       {'logreg__solver': 'saga', 'logreg__C': 0.1}       0.877333        0.888       0.860     Yes               0.028                 0.888                0.860               0.903153              0.891720            0.907240           0.886076        0.905192       0.888889             0.935388            0.922675        0.5             140               17              75               18  0.886076  0.184783  0.815217  0.113924    0.298707     [stand, logreg]  Test Size=0.25, Threshold=0.50...
    1    KNeighborsClassifier       0.25        None         None     None               ...         50                 ...    Accuracy  {'knn_class__weights': 'distance', 'knn_class_...       0.861333        1.000       0.836     Yes               0.164                 1.000                0.836               1.000000              0.897959            1.000000           0.835443        1.000000       0.865574             1.000000            0.911805        0.5             132               15              77               26  0.835443  0.163043  0.836957  0.164557    0.327600  [stand, knn_class]  Test Size=0.25, Threshold=0.50...
    2  DecisionTreeClassifier       0.25        None         None     None               ...         50                 ...    Accuracy  {'tree_class__min_samples_split': 15, 'tree_cl...       0.882667        0.932       0.856     Yes               0.076                 0.932                0.856               0.955711              0.929577            0.927602           0.835443        0.941447       0.880000             0.974926            0.919889        0.5             132               10              82               26  0.835443  0.108696  0.891304  0.164557    0.273253        [tree_class]  Test Size=0.25, Threshold=0.50...

    Example 2: Compare models with more pipeline steps, stratification, under
    sampling, and resampling for SVM, with SVM probabilities enabled:

    >>> results_df = compare_models(
    ...
    ...     # Data split and sampling
    ...     x=X, y=y, test_size=0.25, stratify=y, under_sample=0.8,
    ...     over_sample=None, svm_knn_resample=0.2,
    ...
    ...     # Models and pipeline steps
    ...     imputer='simple_imputer', transformers=None, scaler='stand', selector=None,
    ...     models=['logreg', 'svm_proba'], svm_proba=True,
    ...
    ...     # Grid search
    ...     search_type='random', scorer='accuracy', grid_cv='kfold_5', verbose=1,
    ...
    ...     # Model evaluation and charts
    ...     model_eval=True, plot_perf=True, plot_curve=True, fig_size=(12,6),
    ...     legend_loc='lower left', rotation=45, threshold=0.5,
    ...     class_map=class_map, pos_label=1, title='Breast Cancer',
    ...
    ...     # Config, preferences and notes
    ...     config=my_config, class_weight=None, random_state=42, decimal=2,
    ...     n_jobs=None, notes='Test Size=0.25, Threshold=0.50'
    ... )  #doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    <BLANKLINE>
    -----------------------------------------------------------------------------------------
    Starting Data Processing - ... UTC
    -----------------------------------------------------------------------------------------
    <BLANKLINE>
    Classification type detected: binary
    Unique values in y: [0 1]
    <BLANKLINE>
    Train/Test split, test_size:  0.25
    X_train, X_test, y_train, y_test shapes:  (750, 20) (250, 20) (750,) (250,)
    <BLANKLINE>
    Undersampling via RandomUnderSampler strategy:  0.8
    X_train, y_train shapes before:  (750, 20) (750,)
    y_train value counts before:  Target
    1    450
    0    300
    Name: count, dtype: int64
    Running RandomUnderSampler on X_train, y_train...
    X_train, y_train shapes after:  (675, 20) (675,)
    y_train value counts after:  Target
    1    375
    0    300
    Name: count, dtype: int64
    <BLANKLINE>
    -----------------------------------------------------------------------------------------
    1/2: Starting LogisticRegression Random Search - ... UTC
    -----------------------------------------------------------------------------------------
    <BLANKLINE>
    Fitting 5 folds for each of 10 candidates, totalling 50 fits
    <BLANKLINE>
    Total Time: ... seconds
    Average Fit Time: ... seconds
    Inference Time: ...
    Best CV Accuracy Score: 0.87
    Train Accuracy Score: 0.88
    Test Accuracy Score: 0.86
    Overfit: Yes
    Overfit Difference: 0.01
    Best Parameters: {'logreg__solver': 'saga', 'logreg__C': 0.1}
    <BLANKLINE>
    LogisticRegression Binary Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
       Malignant       0.84      0.82      0.83       100
          Benign       0.88      0.89      0.89       150
    <BLANKLINE>
        accuracy                           0.86       250
       macro avg       0.86      0.86      0.86       250
    weighted avg       0.86      0.86      0.86       250
    <BLANKLINE>
    ROC AUC: 0.92
    <BLANKLINE>
                   Predicted:0         1
    Actual: 0                82        18
    Actual: 1                16        134
    <BLANKLINE>
    True Positive Rate / Sensitivity: 0.89
    True Negative Rate / Specificity: 0.82
    False Positive Rate / Fall-out: 0.18
    False Negative Rate / Miss Rate: 0.11
    <BLANKLINE>
    Positive Class: Benign (1)
    Threshold: 0.5
    <BLANKLINE>
    -----------------------------------------------------------------------------------------
    2/2: Starting SVC Random Search - ... UTC
    -----------------------------------------------------------------------------------------
    <BLANKLINE>
    Training data resampled to 20.0% of original for KNN and SVM speed improvement
    X_train, y_train shapes after:  (135, 20) (135,)
    y_train value counts after:  Target
    1    75
    0    60
    Name: count, dtype: int64
    <BLANKLINE>
    Fitting 5 folds for each of 10 candidates, totalling 50 fits
    <BLANKLINE>
    Total Time: ... seconds
    Average Fit Time: ... seconds
    Inference Time: ...
    Best CV Accuracy Score: 0.87
    Train Accuracy Score: 0.90
    Test Accuracy Score: 0.86
    Overfit: Yes
    Overfit Difference: 0.05
    Best Parameters: {'svm_proba__kernel': 'linear', 'svm_proba__C': 0.01}
    <BLANKLINE>
    SVC Binary Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
       Malignant       0.83      0.85      0.84       100
          Benign       0.90      0.88      0.89       150
    <BLANKLINE>
        accuracy                           0.87       250
       macro avg       0.86      0.86      0.86       250
    weighted avg       0.87      0.87      0.87       250
    <BLANKLINE>
    ROC AUC: 0.92
    <BLANKLINE>
                   Predicted:0         1
    Actual: 0                85        15
    Actual: 1                18        132
    <BLANKLINE>
    True Positive Rate / Sensitivity: 0.88
    True Negative Rate / Specificity: 0.85
    False Positive Rate / Fall-out: 0.15
    False Negative Rate / Miss Rate: 0.12
    <BLANKLINE>
    Positive Class: Benign (1)
    Threshold: 0.5
    """
    # Initialize debugging, controlled via 'debug' parameter
    db = DebugPrinter(debug = debug)
    db.print('-' * 40)
    db.print('START compare_models')
    db.print('-' * 40, '\n')
    db.print('x shape:', x.shape)
    db.print('y shape:', y.shape)
    db.print('models:', models)
    db.print('imputer:', imputer)
    db.print('impute_first:', impute_first)
    db.print('transformers:', transformers)
    db.print('scaler:', scaler)
    db.print('selector:', selector)
    db.print('cat_columns:', cat_columns)
    db.print('num_columns:', num_columns)
    db.print('class_map:', class_map)
    db.print('pos_label:', pos_label)
    db.print('test_size:', test_size)
    db.print('threshold:', threshold)
    db.print('class_weight:', class_weight)
    db.print('stratify:', stratify)
    db.print('search_type:', search_type)
    db.print('cv_folds:', grid_cv)
    db.print('plot_perf:', plot_perf)
    db.print('scorer:', scorer)
    db.print('random_state:', random_state)
    db.print('decimal:', decimal)
    db.print('verbose:', verbose)
    db.print('title:', title)
    db.print('fig_size:', fig_size)
    db.print('figmulti:', figmulti)
    db.print('multi_class:', multi_class)
    db.print('average:', average)
    db.print('legend_loc:', legend_loc)
    db.print('model_eval:', model_eval)
    db.print('svm_proba:', svm_proba)
    db.print('max_iter:', max_iter)
    db.print('rotation:', rotation)
    db.print('plot_curve:', plot_curve)
    db.print('under_sample:', under_sample)
    db.print('over_sample:', over_sample)
    db.print('notes:', notes)
    db.print('svm_knn_resample:', svm_knn_resample)
    db.print('n_jobs:', n_jobs)
    db.print('output:', output)
    db.print('timezone:', timezone)
    db.print('config:', config)

    # Define required parameters
    required_params = {
        'x': x,
        'y': y,
        'models': models,
        'config': config
    }

    # Find which parameters are missing
    db.print('\nChecking for missing parameters...')
    missing_params = [name for name, value in required_params.items() if value is None]

    # Show error message if required parameters are missing
    if missing_params:
        missing_str = ", ".join(missing_params)
        raise ValueError(f"Missing required parameters: {missing_str}.")

    # Define required keys
    required_keys = ['models', 'params']

    # Check for missing keys
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        missing_str = ", ".join(missing_keys)
        raise ValueError(f"Missing required configuration keys: {missing_str}")

    # Create a mapping from model key to class name based on the provided configuration
    # model_map = {key: value.__class__.__name__ for key, value in config['models'].items()}
    model_map = {key: (value, value.__class__.__name__) for key, value in config['models'].items()}
    db.print('model_map:', model_map)

    # Check if all provided model keys exist in the model_map
    missing_models = [model_key for model_key in models if model_key not in model_map]

    # If there are missing models, raise an error now instead of finding out later
    if missing_models:
        known_models = ', '.join(model_map.keys())
        missing_models_str = ', '.join(missing_models)
        raise ValueError(f"'{missing_models_str}' not in config['models']. Please add them to your configuration. Known models are: {known_models}")

    # Store the grid search params from the config in grid_params
    # To-do: Make grid search optional
    grid_params = config['params']
    db.print('grid_params:', grid_params)

    # Configure the cross-validation function for Grid Search
    if isinstance(grid_cv, int):
        db.print(f'\ngrid_cv is int: {grid_cv}. Using KFold cross-validation...')
        cv_func = KFold(n_splits=grid_cv, shuffle=True, random_state=random_state)
    elif isinstance(grid_cv, str):
        db.print(f"\ngrid_cv is str: {grid_cv}. Looking for function in config['cv']...")
        if 'cv' not in config:
            raise ValueError("Key 'cv' not found in config. Please define a cross-validation function in config['cv'] and set grid_cv to that string name. Alternatively, specify an int for the number of folds, or don't specify grid_cv to go with default of 5.")
        elif config['cv'] is None:
            raise ValueError("config['cv'] is None. Please define a cross-validation function in config['cv'] and set grid_cv to that string name. Alternatively, specify an int for the number of folds, or don't specify grid_cv to go with default of 5.")
        # Get the cross-validation function from the config
        elif grid_cv in config['cv']:
            cv_func = config['cv'][grid_cv]
            db.print("grid_cv found in config['cv']. Using specified instance for cross-validation...")
        else:
            raise ValueError(f"Invalid grid_cv: {grid_cv}. Please define a cross-validation function in config['cv'] and set grid_cv to that string name. Alternatively, specify an int for the number of folds, or don't specify grid_cv to go with default of 5.")
    else:
        db.print(f"\ngrid_cv is None or not an int or str. Using default KFold cross-validation with 5 splits...")
        cv_func = KFold(n_splits=5, shuffle=True, random_state=random_state)
    db.print('cv_func:', cv_func)

    # Function to create a scorer and a display name from the scorer param
    def get_scorer_and_name(scorer, pos_label=None):
        # Define valid average types for multi-class/multi-label scenarios
        average_types = ['micro', 'macro', 'weighted', 'samples']

        # Define valid scorers, including those with specific average types
        valid_scorers = [
            'accuracy', 'balanced_accuracy', 'neg_log_loss', 'average_precision',
            'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted',
            *['precision', 'recall', 'f1'],  # Basic forms for binary classification with pos_label
            *[
                f'{metric}_{avg}' for metric in ['precision', 'recall', 'f1']
                for avg in average_types
            ]
        ]

        # Function to build the scoring function and display name
        def build_scoring_function(score_type, pos_label=None, average='macro', zero_division=0):
            if pos_label is not None:
                # For binary classification tasks requiring a pos_label
                return (make_scorer(eval(f'{score_type}_score'), pos_label=pos_label, zero_division=zero_division),
                        f'{score_type.capitalize()} (pos_label={pos_label})')
            elif average in average_types:
                # For multi-class/multi-label tasks specifying an average type
                return make_scorer(eval(f'{score_type}_score'), average=average, zero_division=zero_division), f'{score_type.capitalize()} ({average})'
            else:
                raise ValueError(f"Invalid average type: {average}. Valid options are: {', '.join(average_types)}")

        # Determine the scorer and display name based on input
        db.print('\nCreating scoring function...')
        if scorer in valid_scorers:
            if scorer in ['precision', 'recall', 'f1'] and pos_label is None:
                # Default to 'macro' average for multi-class tasks if pos_label is not specified
                db.print('Using macro average for multi-class tasks...')
                scoring_function, display_name = build_scoring_function(scorer, average='macro')
            elif scorer.startswith(('precision_', 'recall_', 'f1_')):
                # Extract score type and average type from scorer string
                db.print('Extracting score type and average type from scorer string...')
                score_type, avg_type = scorer.split('_')
                scoring_function, display_name = build_scoring_function(score_type, average=avg_type, zero_division=0)
            elif scorer == 'accuracy':
                db.print('Using accuracy as the scoring function...')
                scoring_function, display_name = 'accuracy', 'Accuracy'
            else:
                # Use predefined scikit-learn scorer strings for other cases
                db.print('Using predefined scikit-learn scorer strings...')
                scoring_function, display_name = scorer, scorer.capitalize()
        else:
            # Show an error message if the scorer is invalid
            raise ValueError(f"Unsupported scorer: {scorer}. Valid options are: {', '.join(valid_scorers)}")

        return scoring_function, display_name

    # Define the scorer and display name
    scorer, scorer_name = get_scorer_and_name(scorer=scorer, pos_label=pos_label)
    db.print('scorer:', scorer)
    db.print('scorer_name:', scorer_name)

    # Empty timestamp by default for test cases where we don't want time differences to trigger a failure
    timestamp = ''

    # Set initial timestamp for data processing
    current_time = datetime.now(pytz.timezone(timezone))
    timestamp = current_time.strftime(f'%b %d, %Y %I:%M %p {timezone}')

    if output:
        print(f"\n-----------------------------------------------------------------------------------------")
        print(f"Starting Data Processing - {timestamp}")
        print(f"-----------------------------------------------------------------------------------------\n")

    # Detect the type of classification problem
    unique_y = np.unique(y)
    num_classes = len(unique_y)
    db.print('unique_y:', unique_y)
    db.print('num_classes:', num_classes)
    if num_classes > 2:
        class_type = 'multi'
        if average is None:
            average = 'macro'
    elif num_classes == 2:
        class_type = 'binary'
        average = 'binary'
    else:
        raise ValueError(f"Check data, cannot classify. Number of classes in y_test ({num_classes}) is less than 2: {unique_y}")
    if output:
        print(f"Classification type detected: {class_type}")
        print("Unique values in y:", unique_y)

    # Change data type of y if necessary
    # if y.dtype.kind in 'biufc':  # If y is numeric
    #     y = y.astype(int)  # Convert to int for numeric labels
    # else:
    #     y = y.astype(str)  # Convert to str for categorical labels
    #
    # if output:
    #     print(f"y data type after conversion: {y.dtype}")

    # Make sure y is a Series or a one-dimensional array
    if isinstance(y, pd.DataFrame):
        # Check if y is a DataFrame with only one column
        if y.shape[1] == 1:
            # Convert the single-column DataFrame to a Series
            db.print('\nConverting y from DataFrame to Series...')
            y = y.squeeze()
            db.print('y shape after conversion:', y.shape)
        else:
            # Handle the case where y is a DataFrame with multiple columns
            raise ValueError("y should be a Series or a one-dimensional array, but a DataFrame with multiple columns was provided.")

    # Perform the train/test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=stratify,
                                                        random_state=random_state)

    if output:
        print("\nTrain/Test split, test_size: ", test_size)
        print("X_train, X_test, y_train, y_test shapes: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Over sample with SMOTE, if requested
    if over_sample:
        if output:
            print("\nOversampling via SMOTE strategy: ", over_sample)
            print("X_train, y_train shapes before: ", X_train.shape, y_train.shape)
            print("y_train value counts before: ", y_train.value_counts())
            print("Running SMOTE on X_train, y_train...")
        over = SMOTE(sampling_strategy=over_sample, random_state=random_state)
        X_train, y_train = over.fit_resample(X_train, y_train)
        if output:
            print("X_train, y_train shapes after: ", X_train.shape, y_train.shape)
            print("y_train value counts after: ", y_train.value_counts())

    # Under sample with RandomUnderSampler, if requested
    if under_sample:
        if output:
            print("\nUndersampling via RandomUnderSampler strategy: ", under_sample)
            print("X_train, y_train shapes before: ", X_train.shape, y_train.shape)
            print("y_train value counts before: ", y_train.value_counts())
            print("Running RandomUnderSampler on X_train, y_train...")
        under = RandomUnderSampler(sampling_strategy=under_sample, random_state=random_state)
        X_train, y_train = under.fit_resample(X_train, y_train)
        if output:
            print("X_train, y_train shapes after: ", X_train.shape, y_train.shape)
            print("y_train value counts after: ", y_train.value_counts())

    # Initialized some variables and lists
    timestamp_list = []
    model_name_list = []
    pipeline_list = []

    fit_time_list = []
    fit_count_list = []
    avg_fit_time_list = []
    inference_time_list = []

    train_score_list = []
    test_score_list = []

    overfit_list = []
    overfit_diff_list = []

    best_param_list = []
    best_cv_score_list = []
    best_estimator_list = []

    train_accuracy_list = []
    test_accuracy_list = []

    train_precision_list = []
    test_precision_list = []

    train_recall_list = []
    test_recall_list = []

    train_f1_list = []
    test_f1_list = []

    train_roc_auc_list = []
    test_roc_auc_list = []

    binary_metrics = None
    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []
    tpr_list = []
    fpr_list = []
    tnr_list = []
    fnr_list = []
    fr_list = []

    resample_list = []
    resample_completed = False

    # Function to use a subset of the data for KNN and SVM which can be compute intensive
    def resample_for_knn_svm(X_train, y_train):
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, test_size=1-svm_knn_resample, stratify=y_train, random_state=random_state
        )
        if output:
            print(f"Training data resampled to {svm_knn_resample*100}% of original for KNN and SVM speed improvement")
            print("X_train, y_train shapes after: ", X_train.shape, y_train.shape)
            print("y_train value counts after: ", y_train.value_counts(), "\n")

        return X_train, y_train

    # Function to create the grid search object based on the model_type key
    def create_grid(model_type):
        # Ensure the model type is in the params dictionary
        if model_type not in grid_params:
            raise ValueError(f"Parameters for {model_type} are not defined in the grid_params dictionary")

        # Grab the model params for the grid search
        combined_params = grid_params[model_type]

        # Add optional params for pipeline components, they all need to be in one dict for the search
        if imputer is not None and imputer in grid_params:
            combined_params = {**combined_params, **grid_params[imputer]}
        if selector is not None and selector in grid_params:
            combined_params = {**combined_params, **grid_params[selector]}
        if scaler is not None and scaler in grid_params:
            combined_params = {**combined_params, **grid_params[scaler]}

        # Select the appropriate search method
        if search_type == 'grid':
            grid = GridSearchCV(pipe, param_grid=combined_params, scoring=scorer, verbose=verbose, cv=cv_func, n_jobs=n_jobs)
        elif search_type == 'random':
            grid = RandomizedSearchCV(pipe, param_distributions=combined_params, scoring=scorer, verbose=verbose,
                                      cv=cv_func, random_state=random_state, n_jobs=n_jobs)
        else:
            raise ValueError("search_type should be either 'grid' for GridSearchCV, or 'random' for RandomizedSearchCV")

        return grid

    # Clean up the grid search type for display
    search_string = search_type.capitalize()

    # Set count of total models to iterate through
    total_models = len(models)

    # Model Loop: Iterate through each model in the list and run the workflow for each
    for i, model_key in enumerate(models):

        # Get the model class and a text version of the name from the mapping we did earlier
        model_class, model_name = model_map[model_key]
        db.print(f'\nStarting iteration. i: {i}, total_models: {total_models}, model_key: {model_key}, model_class:{model_class}, model_name: {model_name}:')


        # Create the timestamp for this model's iteration
        current_time = datetime.now(pytz.timezone(timezone))
        timestamp = current_time.strftime(f'%b %d, %Y %I:%M %p {timezone}')
        timestamp_list.append(timestamp)

        # Show a banner with number, model name, search type, timestamp, for this model's iteration
        if output:
            print(f"\n-----------------------------------------------------------------------------------------")
            print(f"{i+1}/{total_models}: Starting {model_name} {search_string} Search - {timestamp}")
            print(f"-----------------------------------------------------------------------------------------\n")

        # Resample the data only for KNN and SVC, if svn_knn_resample is defined
        if svm_knn_resample is not None and model_name in ['KNeighborsClassifier', 'SVC']:
            db.print('\nResampling for KNN and SVM...')
            X_train, y_train = resample_for_knn_svm(X_train, y_train)
            resample_list.append(svm_knn_resample)
        else:
            resample_list.append("None")

        # Set the random seed to random_state for models using TensorFlow
        if model_name == 'KerasClassifier':
            db.print('\nSetting random seed for Keras Classifier:', random_state)
            tf.random.set_seed(random_state)

        db.print('\nCreating pipeline from transformer and model parameters...')
        # Create a pipeline from transformer and model parameters
        pipe = create_pipeline(imputer_key=imputer, transformer_keys=transformers, scaler_key=scaler,
                               selector_key=selector, model_key=model_key, config=config,
                               cat_columns=cat_columns, num_columns=num_columns, class_weight=class_weight,
                               random_state=random_state, max_iter=max_iter, impute_first=impute_first)
        db.print('pipe:', pipe)

        db.print('\nCreating grid search object...')
        grid = create_grid(model_type=model_key)
        db.print('grid:', grid)

        # Append to each list the value from this iteration, starting with model name, pipeline, etc.
        model_name_list.append(model_name)
        pipeline_list.append(list(pipe.named_steps.keys()))

        # Fit the model and measure total fit time, append to list
        start_time = time.time()
        db.print('\nFitting grid...')
        grid.fit(X_train, y_train)
        db.print('\nGrid fit complete.')
        db.print('\nGrid search results:')
        db.print(grid.cv_results_)
        fit_time = time.time() - start_time
        fit_time_list.append(fit_time)
        if output:
            print(f"\nTotal Time: {fit_time:.{decimal}f} seconds")

        # Calculate average fit time (for each fold in the CV search) and append to list
        db.print('\nCalculating average fit time...')
        n_splits = cv_func.get_n_splits()
        db.print('n_splits:', n_splits)
        n_folds = len(grid.cv_results_['params'])
        db.print('n_folds:', n_folds)
        fit_count = n_splits * n_folds
        db.print('fit_count:', fit_count)
        fit_count_list.append(fit_count)
        db.print('fit_time:', fit_time)
        avg_fit_time = fit_time / fit_count
        avg_fit_time_list.append(avg_fit_time)
        if output:
            print(f"Average Fit Time: {avg_fit_time:.{decimal}f} seconds")

        # Function to apply different thresholds for binary classification
        def apply_threshold(probs, threshold):
            return np.where(probs >= threshold, 1, 0)

        # Debugging data for detecting support of predict_proba
        db.print("grid.best_estimator_:", grid.best_estimator_)
        db.print("hasattr(grid.best_estimator_, 'predict_proba'):", hasattr(grid.best_estimator_, 'predict_proba'))
        db.print("hasattr(grid.best_estimator_, 'decision_function'):", hasattr(grid.best_estimator_, 'decision_function'))

        # Generate train predictions based on class type and threshold
        db.print('\nGenerating train predictions based on class type and threshold...')
        if class_type == 'binary':
            if hasattr(grid.best_estimator_, 'predict_proba'):
                # Model supports probability estimates
                if threshold != 0.5:
                    db.print(f'Class: {class_type}, Method: predict_proba, Threshold: {threshold}, Data: Train')
                    # Get probabilities for the positive class
                    probabilities_train = grid.predict_proba(X_train)[:, 1]
                    # Apply the custom threshold to get binary predictions
                    y_train_pred = apply_threshold(probabilities_train, threshold)
                else:
                    db.print(f'Class: {class_type}, Method: predict, Threshold: {threshold}, Data: Train')
                    # Use default predictions for binary classification
                    y_train_pred = grid.predict(X_train)
            elif hasattr(grid.best_estimator_, 'decision_function'):
                db.print(f'Class: {class_type}, Method: decision_function, Threshold: {threshold}, Data: Train')
                # Model does not support probability estimates but has a decision function (ex: SVC without probability)
                decision_values_train = grid.decision_function(X_train)
                # Apply the custom threshold to the decision function values
                y_train_pred = apply_threshold(decision_values_train, threshold)
            else:
                db.print(f'Class: {class_type}, Method: predict, Threshold: {threshold}, Data: Train')
                # Use default predictions if neither predict_proba nor decision_function are available
                y_train_pred = grid.predict(X_train)
        elif class_type == 'multi':
            db.print(f'Class: {class_type}, Method: predict, Threshold: {threshold}, Data: Train')
            # Use default predictions for multi-class classification
            y_train_pred = grid.predict(X_train)

        # Start tracking the inference time, or test predictions time
        start_time = time.time()

        # Generate test predictions based on class type and threshold
        db.print('\nGenerating test predictions based on class type and threshold...')
        if class_type == 'binary':
            if hasattr(grid.best_estimator_, 'predict_proba'):
                if threshold != 0.5:
                    db.print(f'Class: {class_type}, Method: predict_proba, Threshold: {threshold}, Data: Test')
                    probabilities_test = grid.predict_proba(X_test)[:, 1]
                    y_test_pred = apply_threshold(probabilities_test, threshold)
                else:
                    db.print(f'Class: {class_type}, Method: predict, Threshold: {threshold}, Data: Test')
                    y_test_pred = grid.predict(X_test)
            elif hasattr(grid.best_estimator_, 'decision_function'):
                db.print(f'Class: {class_type}, Method: decision_function, Threshold: {threshold}, Data: Test')
                decision_values_test = grid.decision_function(X_test)
                y_test_pred = apply_threshold(decision_values_test, threshold)
            else:
                db.print(f'Class: {class_type}, Method: predict, Threshold: {threshold}, Data: Test')
                y_test_pred = grid.predict(X_test)
        elif class_type == 'multi':
            db.print(f'Class: {class_type}, Method: predict, Threshold: {threshold}, Data: Test')
            y_test_pred = grid.predict(X_test)

        # Capture the inference time, or test predictions time
        inference_time = time.time() - start_time
        inference_time_list.append(inference_time)
        if output:
            print(f"Inference Time: {inference_time:.{decimal}f}")

        # Calculate ROC AUC, based on class type and predict_proba support
        def calculate_roc_auc(grid, X, y, class_type, note):
            try:
                # Attempt to use predict_proba or decision_function based on class_type
                if class_type == 'multi':
                    # Ensure predict_proba is available for the grid (model)
                    if hasattr(grid, 'predict_proba'):
                        db.print(f'Class: {class_type}, Method: predict_proba(X), Threshold: {threshold}, Data: {note}, Score: ROC AUC')
                        pred_proba = grid.predict_proba(X)
                        # Check if predict_proba output is 2D and correct shape, adjust if necessary
                        if pred_proba.ndim == 1:
                            db.print(f'pred_proba.ndim == 1, Before: {pred_proba.shape}')
                            db.print('pred_proba:', pred_proba)
                            pred_proba = np.expand_dims(pred_proba, axis=1)
                            db.print(f'After: {pred_proba.shape}')
                            db.print('pred_proba:', pred_proba)
                        return roc_auc_score(y, pred_proba, multi_class='ovr')
                    else:
                        print(f"Model does not support 'predict_proba' for multi-class ROC AUC calculation.")
                        return None
                else:
                    # For binary classification, directly use predict_proba or decision_function
                    if hasattr(grid, 'predict_proba'):
                        db.print(f'Class: {class_type}, Method: predict_proba(X)[:, 1], Threshold: {threshold}, Data: {note}, Score: ROC AUC')
                        pred_proba = grid.predict_proba(X)[:, 1]
                        db.print('pred_proba:', pred_proba)
                        return roc_auc_score(y, pred_proba)
                    elif hasattr(grid, 'decision_function'):
                        db.print(f'Class: {class_type}, Method: decision_function(X), Threshold: {threshold}, Data: {note}, Score: ROC AUC')
                        decision_values = grid.decision_function(X)
                        db.print('decision_values:', decision_values)
                        return roc_auc_score(y, decision_values)
                    else:
                        print(f"Model does not support 'predict_proba' or 'decision_function' for binary ROC AUC calculation.")
                        return None
            except Exception as e:
                print(f"An error occurred during ROC AUC calculation: {str(e)}")
                return None

        # Calculate the train and test ROC AUC
        db.print('\nCalculating ROC AUC...')
        train_roc_auc = calculate_roc_auc(grid, X_train, y_train, class_type=class_type, note='Train')
        test_roc_auc = calculate_roc_auc(grid, X_test, y_test, class_type=class_type, note='Test')

        # Calculate train metrics
        db.print('\nCalculating train metrics...')
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average=average, zero_division=0, pos_label=pos_label)
        train_recall = recall_score(y_train, y_train_pred, average=average, pos_label=pos_label)
        train_f1 = f1_score(y_train, y_train_pred, average=average, pos_label=pos_label)

        # Calculate test metrics
        db.print('\nCalculating test metrics...')
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average=average, zero_division=0, pos_label=pos_label)
        test_recall = recall_score(y_test, y_test_pred, average=average, pos_label=pos_label)
        test_f1 = f1_score(y_test, y_test_pred, average=average, pos_label=pos_label)

        # Append train metrics to lists
        db.print('\nAppending train metrics to lists...')
        train_accuracy_list.append(train_accuracy)
        train_precision_list.append(train_precision)
        train_recall_list.append(train_recall)
        train_f1_list.append(train_f1)
        train_roc_auc_list.append(train_roc_auc)

        # Append test metrics to lists
        db.print('\nAppending test metrics to lists...')
        test_accuracy_list.append(test_accuracy)
        test_precision_list.append(test_precision)
        test_recall_list.append(test_recall)
        test_f1_list.append(test_f1)
        test_roc_auc_list.append(test_roc_auc)

        # Get the best Grid Search CV score and append to list
        db.print('\nGetting the best Grid Search CV score...')
        best_cv_score = grid.best_score_
        best_cv_score_list.append(best_cv_score)
        if output:
            print(f"Best CV {scorer_name} Score: {best_cv_score:.{decimal}f}")

        # Get the best Grid Search Train score and append to list
        db.print('\nGetting the best Grid Search Train score...')
        train_score = grid.score(X_train, y_train)
        train_score_list.append(train_score)
        if output:
            print(f"Train {scorer_name} Score: {train_score:.{decimal}f}")

        # Get the best Grid Search Test score and append to list
        db.print('\nGetting the best Grid Search Test score...')
        test_score = grid.score(X_test, y_test)
        test_score_list.append(test_score)
        if output:
            print(f"Test {scorer_name} Score: {test_score:.{decimal}f}")

        # Assess the degree of overfit (train score higher than test score)
        db.print('\nAssessing the degree of overfit...')
        overfit_diff = train_score - test_score
        overfit_diff_list.append(overfit_diff)
        if train_score > test_score:
            overfit = 'Yes'
        else:
            overfit = 'No'
        overfit_list.append(overfit)
        if output:
            print(f"Overfit: {overfit}")
            print(f"Overfit Difference: {overfit_diff:.{decimal}f}")

        # Capture the best model and params from grid search
        db.print('\nCapturing the best model and params from grid search...')
        best_estimator = grid.best_estimator_
        best_estimator_list.append(best_estimator)
        best_params = grid.best_params_
        best_param_list.append(best_params)
        if output:
            print(f"Best Parameters: {best_params}")

        # Output the neural network layers for KerasClassifier
        if model_name == 'KerasClassifier':
            db.print('\nOutputting the neural network layers for KerasClassifier...')
            keras_classifier = grid.best_estimator_.named_steps['keras_class']
            keras_model = keras_classifier.model_
            if output:
                print('') # Empty line for spacing
                # Access the Keras model from the best estimator in the grid search
                keras_model.summary()

        # Display model evaluation metrics and plots by calling 'eval_model' function
        # Note: Some of this duplicates what we just calculated, room for future optimization
        if model_eval:
            db.print('\nDisplaying model evaluation metrics and plots...')

            # Handle binary vs. multi-class, and special case for SVC that requires svm_proba=True
            if model_name != 'SVC' or (model_name == 'SVC' and svm_proba == True):
                if class_type == 'binary':
                    # Capture binary metrics for processing later, only in the binary case
                    binary_metrics = eval_model(y_test=y_test, y_pred=y_test_pred, x_test=X_test, estimator=grid,
                                                class_map=class_map, pos_label=pos_label, debug=debug,
                                                class_type=class_type, model_name=model_name, threshold=threshold,
                                                decimal=decimal, plot=True, figsize=(12,11), class_weight=class_weight,
                                                return_metrics=True, output=output)
                elif class_type == 'multi':
                    multi_metrics = eval_model(y_test=y_test, y_pred=y_test_pred, x_test=X_test, estimator=grid,
                                               class_map=class_map, pos_label=pos_label, debug=debug,
                                               class_type=class_type, model_name=model_name, average=average,
                                               decimal=decimal, plot=True, figmulti=figmulti, class_weight=class_weight,
                                               return_metrics=True, output=output, multi_class=multi_class)

            # For neural network, if plot_curves=True, plot training history
            if model_name == 'KerasClassifier' and plot_curve:

                # Access the training history
                db.print('best_estimator:', best_estimator)
                db.print('keras_classifier:', keras_classifier)
                db.print('keras_model:', keras_model)
                db.print('keras_classifier.history_:', keras_classifier.history_)
                history = keras_classifier.history_

                # Plot the training history
                plot_train_history(history=history)

        # Set the binary metric values based on the list of binary metrics, if it was produced by 'eval_model'
        if binary_metrics is not None:
            db.print('\nSetting the binary metric values based on the list of binary metrics...')
            tp = binary_metrics['True Positives']
            fp = binary_metrics['False Positives']
            tn = binary_metrics['True Negatives']
            fn = binary_metrics['False Negatives']
            tpr = binary_metrics['TPR']
            fpr = binary_metrics['FPR']
            tnr = binary_metrics['TNR']
            fnr = binary_metrics['FNR']
            fr = fnr + fpr
        # If no binary metrics, set the values as NaN (better than string, allows numeric formatting from 'format_df')
        else:
            db.print('\nSetting the binary metric values as NaN...')
            tp = np.nan
            fp = np.nan
            tn = np.nan
            fn = np.nan
            tpr = np.nan
            fpr = np.nan
            tnr = np.nan
            fnr = np.nan
            fr = np.nan

        # Append the binary metrics to the list
        db.print('\nAppending the binary metrics to the list...')
        tp_list.append(tp)
        fp_list.append(fp)
        tn_list.append(tn)
        fn_list.append(fn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        tnr_list.append(tnr)
        fnr_list.append(fnr)
        fr_list.append(fr)

        # To debug lists not being the same length, print the lengths
        db.print('\nLength of each list:')
        db.print('Model', len(model_name_list))
        db.print('Test Size', len([test_size] * len(model_name_list)))
        db.print('Over Sample', len([over_sample] * len(model_name_list)))
        db.print('Under Sample', len([under_sample] * len(model_name_list)))
        db.print('Resample', len(resample_list))
        db.print('Total Fit Time', len(fit_time_list))
        db.print('Fit Count', len(fit_count_list))
        db.print('Average Fit Time', len(avg_fit_time_list))
        db.print('Inference Time', len(inference_time_list))
        db.print('Grid Scorer', len([scorer_name] * len(model_name_list)))
        db.print('Best Params', len(best_param_list))
        db.print('Best CV Score', len(best_cv_score_list))
        db.print('Train Score', len(train_score_list))
        db.print('Test Score', len(test_score_list))
        db.print('Overfit', len(overfit_list))
        db.print('Overfit Difference', len(overfit_diff_list))
        db.print('Train Accuracy Score', len(train_accuracy_list))
        db.print('Test Accuracy Score', len(test_accuracy_list))
        db.print('Train Precision Score', len(train_precision_list))
        db.print('Test Precision Score', len(test_precision_list))
        db.print('Train Recall Score', len(train_recall_list))
        db.print('Test Recall Score', len(test_recall_list))
        db.print('Train F1 Score', len(train_f1_list))
        db.print('Test F1 Score', len(test_f1_list))
        db.print('Train ROC AUC Score', len(train_roc_auc_list))
        db.print('Test ROC AUC Score', len(test_roc_auc_list))
        db.print('Threshold', len([threshold] * len(model_name_list)))
        db.print('True Positives', len(tp_list))
        db.print('False Positives', len(fp_list))
        db.print('True Negatives', len(tn_list))
        db.print('False Negatives', len(fn_list))
        db.print('TPR', len(tpr_list))
        db.print('TNR', len(tnr_list))
        db.print('FNR', len(fnr_list))
        db.print('False Rate', len(fr_list))
        db.print('Pipeline', len(pipeline_list))
        db.print('Notes', len([notes] * len(model_name_list)))
        db.print('Timestamp', len(timestamp_list))

        # Create the results DataFrame with each list as a column, with a row for model iteration in this run
        db.print('\nCreating the results DataFrame...')
        results_df = pd.DataFrame({'Model': model_name_list,
                                   'Test Size': [test_size] * len(model_name_list),
                                   'Over Sample': [over_sample] * len(model_name_list),
                                   'Under Sample': [under_sample] * len(model_name_list),
                                   'Resample': resample_list,
                                   'Total Fit Time': fit_time_list,
                                   'Fit Count': fit_count_list,
                                   'Average Fit Time': avg_fit_time_list,
                                   'Inference Time': inference_time_list,
                                   'Grid Scorer': [scorer_name] * len(model_name_list),
                                   'Best Params': best_param_list,
                                   'Best CV Score': best_cv_score_list,
                                   'Train Score': train_score_list,
                                   'Test Score': test_score_list,
                                   'Overfit': overfit_list,
                                   'Overfit Difference': overfit_diff_list,
                                   'Train Accuracy Score': train_accuracy_list,
                                   'Test Accuracy Score': test_accuracy_list,
                                   'Train Precision Score': train_precision_list,
                                   'Test Precision Score': test_precision_list,
                                   'Train Recall Score': train_recall_list,
                                   'Test Recall Score': test_recall_list,
                                   'Train F1 Score': train_f1_list,
                                   'Test F1 Score': test_f1_list,
                                   'Train ROC AUC Score': train_roc_auc_list,
                                   'Test ROC AUC Score': test_roc_auc_list,
                                   'Threshold': [threshold] * len(model_name_list),
                                   'True Positives': tp_list,
                                   'False Positives': fp_list,
                                   'True Negatives': tn_list,
                                   'False Negatives': fn_list,
                                   'TPR': tpr_list,
                                   'FPR': fpr_list,
                                   'TNR': tnr_list,
                                   'FNR': fnr_list,
                                   'False Rate': fr_list,
                                   'Pipeline': pipeline_list,
                                   'Notes': [notes] * len(model_name_list),
                                   'Timestamp': timestamp_list
                                   })

    # Plot a chart showing the performance of each model, if requested
    if plot_perf:
        db.print('\nPlotting a chart showing the performance of each model...')
        # Melt the results_df so we can plot the scores for each model
        db.print('Melting the results_df so we can plot the scores for each model...')
        score_df = results_df.melt(id_vars=['Model'],
                                   value_vars=[f'Best CV Score', f'Train Score', f'Test Score'],
                                   var_name='Split', value_name=f'{scorer_name}')

        # Create the bar plot of Scores by Model and Data Split
        plt.figure(figsize=fig_size)
        sns.barplot(data=score_df, x='Model', y=f'{scorer_name}', hue='Split')
        plt.title(f'{title} {scorer_name} Scores by Model and Data Split', fontsize=18, pad=15)
        plt.yticks(np.arange(0,1.1,0.1))
        plt.xticks(rotation=rotation)
        plt.xlabel('Model', fontsize=14, labelpad=10)
        plt.ylabel(f'{scorer_name}', fontsize=14, labelpad=10)
        plt.legend(loc=legend_loc)
        plt.show()

        # Create the bar plot of Fit Time by Model
        plt.figure(figsize=fig_size)
        sns.barplot(data=results_df, x='Model', y='Average Fit Time')
        plt.title(f'{title} Average Fit Time by Model', fontsize=18, pad=15)
        plt.xticks(rotation=rotation)
        plt.xlabel('Model', fontsize=14, labelpad=10)
        plt.ylabel('Average Fit Time (seconds)', fontsize=14, labelpad=10)
        plt.show()

    # Return the results as a DataFrame
    return results_df


def create_nn_binary(
        hidden_layer_dim: int,
        dropout_rate: float,
        l2_reg: float,
        second_layer_dim: Optional[int] = None,
        third_layer_dim: Optional[int] = None,
        meta: Dict[str, Any] = None
) -> keras.models.Sequential:
    """
    Create a binary classification neural network model.

    This function allows for flexible configuration of the neural network
    structure for binary classification using the KerasClassifier in scikit-learn.
    It supports adding up to three hidden layers with customizable dimensions,
    dropout regularization, and L2 regularization.

    Use this function to create a neural network model with a specific structure
    and regularization settings for binary classification tasks. It is set as the
    `model` parameter of a KerasClassifier instance referenced in the configuration
    file for `compare_models`.

    Parameters
    ----------
    hidden_layer_dim : int
        The number of neurons in the first hidden layer.
    dropout_rate : float
        The dropout rate to be applied after each hidden layer.
    l2_reg : float
        The L2 regularization strength. If greater than 0, L2 regularization is
        applied to the kernel weights of the dense layers.
    second_layer_dim : Optional[int], optional
        The number of neurons in an additional hidden layer. If not None, an
        additional hidden layer is added. Default is None.
    third_layer_dim : Optional[int], optional
        The number of neurons in a third hidden layer. If not None, a third hidden
        layer is added. Default is None.
    meta : Dict[str, Any], optional
        A dictionary containing metadata about the input features and shape.
        Default is None.

    Returns
    -------
    keras.models.Sequential
        The constructed neural network model for binary classification.

    Examples
    --------
    >>> pd.set_option('display.max_columns', None)  # For test consistency
    >>> pd.set_option('display.width', None)  # For test consistency
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    ...                                                     random_state=42)
    >>> meta = {"n_features_in_": 10, "X_shape_": (80, 10)}

    Example 1: Create a basic neural network with default settings:

    >>> model = create_nn_binary(hidden_layer_dim=32, dropout_rate=0.2, l2_reg=0.01,
    ...                       meta=meta)
    >>> model_summary(model)  #doctest: +NORMALIZE_WHITESPACE
            Item                  Name         Type Activation Output Shape  Parameters   Bytes
    0      Model            Sequential   Sequential       None         None         NaN     NaN
    1      Input                 Input  KerasTensor       None   (None, 10)         0.0     0.0
    2      Layer              Hidden_1        Dense       relu   (None, 32)       352.0  1408.0
    3      Layer             Dropout_1      Dropout       None   (None, 32)         0.0     0.0
    4      Layer                Output        Dense    sigmoid    (None, 1)        33.0   132.0
    5  Statistic          Total Params         None       None         None       385.0  1540.0
    6  Statistic      Trainable Params         None       None         None       385.0  1540.0
    7  Statistic  Non-Trainable Params         None       None         None         0.0     0.0

    Example 2: Create a neural network with additional layers and regularization:

    >>> model = create_nn_binary(hidden_layer_dim=64, dropout_rate=0.3, l2_reg=0.05,
    ...                       second_layer_dim=32, third_layer_dim=16, meta=meta)
    >>> model_summary(model)  #doctest: +NORMALIZE_WHITESPACE
             Item                  Name         Type Activation Output Shape  Parameters    Bytes
    0       Model            Sequential   Sequential       None         None         NaN      NaN
    1       Input                 Input  KerasTensor       None   (None, 10)         0.0      0.0
    2       Layer              Hidden_1        Dense       relu   (None, 64)       704.0   2816.0
    3       Layer             Dropout_1      Dropout       None   (None, 64)         0.0      0.0
    4       Layer              Hidden_2        Dense       relu   (None, 32)      2080.0   8320.0
    5       Layer             Dropout_2      Dropout       None   (None, 32)         0.0      0.0
    6       Layer              Hidden_3        Dense       relu   (None, 16)       528.0   2112.0
    7       Layer             Dropout_3      Dropout       None   (None, 16)         0.0      0.0
    8       Layer                Output        Dense    sigmoid    (None, 1)        17.0     68.0
    9   Statistic          Total Params         None       None         None      3329.0  13316.0
    10  Statistic      Trainable Params         None       None         None      3329.0  13316.0
    11  Statistic  Non-Trainable Params         None       None         None         0.0      0.0
    """
    # Capture parameters from metadata
    n_features_in_ = meta["n_features_in_"]
    X_shape_ = meta["X_shape_"]
    n_classes_ = 1  # For binary classification

    # Adjust L2 regularization based on the parameter
    reg = L2(l2_reg) if l2_reg > 0 else None

    # Create a sequential model
    model = keras.models.Sequential(name='Sequential')

    # Create the input layer
    input_shape = (X_shape_[1],)
    model.add(Input(shape=input_shape, name='Input'))

    # Add the first hidden layer
    model.add(Dense(hidden_layer_dim, activation='relu', kernel_regularizer=reg, name='Hidden_1'))
    model.add(Dropout(dropout_rate, name='Dropout_1'))

    # Add a second hidden layer if specified
    if second_layer_dim is not None:
        model.add(Dense(second_layer_dim, activation='relu', kernel_regularizer=reg, name='Hidden_2'))
        model.add(Dropout(dropout_rate, name='Dropout_2'))

    # Add a third hidden layer if specified
    if third_layer_dim is not None:
        model.add(Dense(third_layer_dim, activation='relu', kernel_regularizer=reg, name='Hidden_3'))
        model.add(Dropout(dropout_rate, name='Dropout_3'))

    # Add the output layer for binary classification
    model.add(Dense(n_classes_, activation='sigmoid', name='Output'))

    return model


def create_nn_multi(
        hidden_layer_dim: int,
        dropout_rate: float,
        l2_reg: float,
        second_layer_dim: Optional[int] = None,
        third_layer_dim: Optional[int] = None,
        meta: Dict[str, Any] = None
) -> keras.models.Sequential:
    """
    Create a multi-class classification neural network model.

    This function allows for flexible configuration of the neural network
    structure for multi-class classification using the KerasClassifier in
    scikit-learn. It supports adding an optional hidden layer with customizable
    dimensions, dropout regularization, and L2 regularization.

    Use this function to create a neural network model with a specific structure
    and regularization settings for multi-class classification tasks. It is set as
    the `model` parameter of a KerasClassifier instance referenced in the
    configuration file for `compare_models`.

    Parameters
    ----------
    hidden_layer_dim : int
        The number of neurons in the hidden layer.
    dropout_rate : float
        The dropout rate to be applied after the hidden layer.
    l2_reg : float
        The L2 regularization strength applied to the kernel weights of the dense
        layers.
    second_layer_dim : Optional[int], optional
        The number of neurons in an additional hidden layer. If not None, an
        additional hidden layer is added. Default is None.
    third_layer_dim : Optional[int], optional
        The number of neurons in a third hidden layer. If not None, a third hidden
        layer is added. Default is None.
    meta : Dict[str, Any], optional
        A dictionary containing metadata about the input features, shape, and
        number of classes. Default is None.

    Returns
    -------
    keras.models.Sequential
        The constructed neural network model for multi-class classification.

    Examples
    --------
    >>> pd.set_option('display.max_columns', None)  # For test consistency
    >>> pd.set_option('display.width', None)  # For test consistency
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    ...                                                     random_state=42)
    >>> meta = {"n_features_in_": 4, "X_shape_": (120, 4), "n_classes_": 3}

    Example 1: Create a basic neural network with default settings:

    >>> model = create_nn_multi(hidden_layer_dim=64, dropout_rate=0.2, l2_reg=0.01,
    ...                         meta=meta)
    >>> model_summary(model)  #doctest: +NORMALIZE_WHITESPACE
            Item                  Name         Type Activation Output Shape  Parameters   Bytes
    0      Model            Sequential   Sequential       None         None         NaN     NaN
    1      Input                 Input  KerasTensor       None    (None, 4)         0.0     0.0
    2      Layer              Hidden_1        Dense       relu   (None, 64)       320.0  1280.0
    3      Layer             Dropout_1      Dropout       None   (None, 64)         0.0     0.0
    4      Layer                Output        Dense    softmax    (None, 3)       195.0   780.0
    5  Statistic          Total Params         None       None         None       515.0  2060.0
    6  Statistic      Trainable Params         None       None         None       515.0  2060.0
    7  Statistic  Non-Trainable Params         None       None         None         0.0     0.0

    Example 2: Create a neural network with an additional hidden layer:

    >>> model = create_nn_multi(hidden_layer_dim=128, dropout_rate=0.3, l2_reg=0.05,
    ...                         second_layer_dim=64, meta=meta)
    >>> model_summary(model)  #doctest: +NORMALIZE_WHITESPACE
            Item                  Name         Type Activation Output Shape  Parameters    Bytes
    0      Model            Sequential   Sequential       None         None         NaN      NaN
    1      Input                 Input  KerasTensor       None    (None, 4)         0.0      0.0
    2      Layer              Hidden_1        Dense       relu  (None, 128)       640.0   2560.0
    3      Layer             Dropout_1      Dropout       None  (None, 128)         0.0      0.0
    4      Layer              Hidden_2        Dense       relu   (None, 64)      8256.0  33024.0
    5      Layer             Dropout_2      Dropout       None   (None, 64)         0.0      0.0
    6      Layer                Output        Dense    softmax    (None, 3)       195.0    780.0
    7  Statistic          Total Params         None       None         None      9091.0  36364.0
    8  Statistic      Trainable Params         None       None         None      9091.0  36364.0
    9  Statistic  Non-Trainable Params         None       None         None         0.0      0.0
    """
    # Capture parameters from metadata
    n_features_in_ = meta["n_features_in_"]
    X_shape_ = meta["X_shape_"]
    n_classes_ = meta["n_classes_"]  # Number of classes for multi-class classification

    # Adjust L2 regularization based on the parameter
    reg = L2(l2_reg) if l2_reg > 0 else None

    # Create a sequential model
    model = keras.models.Sequential(name='Sequential')

    # Create the input layer
    input_shape = (X_shape_[1],)  # Tuple representing the shape of a single sample
    model.add(Input(shape=input_shape, name='Input'))

    # Add the first hidden layer
    model.add(Dense(hidden_layer_dim, activation='relu', kernel_regularizer=reg, name='Hidden_1'))
    model.add(Dropout(dropout_rate, name='Dropout_1'))

    # Add a second hidden layer if specified
    if second_layer_dim is not None:
        model.add(Dense(second_layer_dim, activation='relu', kernel_regularizer=reg, name='Hidden_2'))
        model.add(Dropout(dropout_rate, name='Dropout_2'))

    # Add a third hidden layer if specified
    if third_layer_dim is not None:
        model.add(Dense(third_layer_dim, activation='relu', kernel_regularizer=reg, name='Hidden_3'))
        model.add(Dropout(dropout_rate, name='Dropout_3'))

    # Output layer for multi-class classification
    model.add(Dense(n_classes_, activation='softmax', name='Output'))

    return model


def create_pipeline(
        imputer_key: Optional[str] = None,
        transformer_keys: Optional[Union[List[str], str]] = None,
        scaler_key: Optional[str] = None,
        selector_key: Optional[str] = None,
        model_key: Optional[str] = None,
        impute_first: bool = True,
        config: Optional[Dict[str, Any]] = None,
        cat_columns: Optional[List[str]] = None,
        num_columns: Optional[List[str]] = None,
        random_state: int = 42,
        class_weight: Optional[Dict[int, float]] = None,
        max_iter: int = 10000,
        debug: bool = False
) -> Pipeline:
    """
    Create a custom pipeline for data preprocessing and modeling.

    This function allows you to define a custom pipeline by specifying the
    desired preprocessing steps (imputation, transformation, scaling, feature
    selection) and the model to use for predictions. Provide the keys
    for the steps you want to include in the pipeline. If a step is not
    specified, it will be skipped. The definition of the keys are defined in
    a configuration dictionary that is passed to the function. If no external
    configuration is provided, a default one will be used.

    * `imputer_key` (str) is selected from `config['imputers']`
    * `transformer_keys` (list or str) are selected from `config['transformers']`
    * `scaler_key` (str) is selected from `config['scalers']`
    * `selector_key` (str) is selected from `config['selectors']`
    * `model_key` (str) is selected from `config['models']`
    * `config['no_scale']` lists model keys that should not be scaled.
    * `config['no_poly']` lists models that should not be polynomial transformed.

    By default, the sequence of the Pipeline steps are: Imputer > Column
    Transformer > Scaler > Selector > Model. However, if `impute_first` is False,
    the data will be imputed after the column transformations. Scaling will not
    be done for any Model that is listed in `config['no_scale']` (ex: for decision
    trees, which don't require scaling).

    A column transformer will be created based on the specified
    `transformer_keys`. Any number of column transformations can be defined here.
    For example, you can define `transformer_keys = ['ohe', 'poly2', 'log']` to
    One-Hot Encode some columns, Polynomial transform some columns, and Log
    transform others. Just define each of these in your config file to
    reference the appropriate column lists. By default, these will transform the
    columns passed in as `cat_columns` or `num_columns`. But you may want to
    apply different transformations to your categorical features. For example,
    if you One-Hot Encode some, but Ordinal Encode others, you could define
    separate column lists for these as 'ohe_columns' and 'ord_columns', and then
    define `transformer_keys` in your config dictionary that reference them.

    Here is an example of the configuration dictionary structure:

    >>> config = {  # doctest: +SKIP
    ...     'imputers': {
    ...         'knn_imputer': KNNImputer().set_output(transform='pandas'),
    ...         'simple_imputer': SimpleImputer()
    ...     },
    ...     'transformers': {
    ...         'ohe': (OneHotEncoder(drop='if_binary', handle_unknown='ignore'),
    ...                 cat_columns),
    ...         'ord': (OrdinalEncoder(), cat_columns),
    ...         'poly2': (PolynomialFeatures(degree=2, include_bias=False),
    ...                   num_columns),
    ...         'log': (FunctionTransformer(np.log1p, validate=True),
    ...                 num_columns)
    ...     },
    ...     'scalers': {
    ...         'stand': StandardScaler(),
    ...         'minmax': MinMaxScaler()
    ...     },
    ...     'selectors': {
    ...         'rfe_logreg': RFE(LogisticRegression(max_iter=max_iter,
    ...                                         random_state=random_state,
    ...                                         class_weight=class_weight)),
    ...         'sfs_linreg': SequentialFeatureSelector(LinearRegression())
    ...     },
    ...     'models': {
    ...         'linreg': LinearRegression(),
    ...         'logreg': LogisticRegression(max_iter=max_iter,
    ...                                      random_state=random_state,
    ...                                      class_weight=class_weight),
    ...         'tree_class': DecisionTreeClassifier(random_state=random_state),
    ...         'tree_reg': DecisionTreeRegressor(random_state=random_state)
    ...     },
    ...     'no_scale': ['tree_class', 'tree_reg'],
    ...     'no_poly': ['tree_class', 'tree_reg'],
    ... }


    Use this function to quickly create a pipeline during model iteration and
    evaluation. You can easily experiment with different combinations of
    preprocessing steps and models to find the best performing pipeline. This
    function is utilized by `iterate_model`, `compare_models`, and
    `compare_reg_models` to dynamically build pipelines as part of that
    larger modeling workflow.

    Parameters
    ----------
    imputer_key : str, optional
        The key corresponding to the imputer to use for handling missing values.
        If not provided, no imputation will be performed.
    transformer_keys : list of str, str, or None, optional
        The keys corresponding to the transformers to apply to the data. This
        can be a list of string keys or a single string key. If not provided,
        no transformers will be applied.
    scaler_key : str or None, optional
        The key corresponding to the scaler to use for scaling the data. If not
        provided, no scaling will be performed.
    selector_key : str or None, optional
        The key corresponding to the feature selector to use for selecting
        relevant features. If not provided, no feature selection will be
        performed.
    model_key : str, optional
        The key corresponding to the model to use for predictions.
    impute_first : bool, default=True
        Whether to perform imputation before applying the transformers. If
        False, imputation will be performed after the transformers.
    config : dict or None, optional
        A dictionary containing the configuration for the pipeline components.
        If not provided, a default configuration will be used.
    cat_columns : list-like, optional
        List of categorical columns from the input dataframe. This is used in
        the default configuration for the relevant transformers.
    num_columns : list-like, optional
        List of numeric columns from the input dataframe. This is used in the
        default configuration for the relevant transformers.
    random_state : int, default=42
        The random state to use for reproducibility.
    class_weight : dict or None, optional
        A dictionary mapping class labels to weights for imbalanced
        classification problems. If not provided, equal weights will be used.
    max_iter : int, default=10000
        The maximum number of iterations for iterative models.
    debug : bool, optional
        Flag to show debugging information.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        The constructed pipeline based on the specified components and
        configuration.

    Examples
    --------
    Prepare sample data for the examples:

    >>> from sklearn.datasets import fetch_california_housing
    >>> X, y = fetch_california_housing(return_X_y=True)
    >>> cat_columns = ['ocean_proximity']
    >>> num_columns = ['longitude', 'latitude', 'housing_median_age',
    ...                  'total_rooms', 'total_bedrooms', 'population',
    ...                  'households', 'median_income']

    Example 1: Create a pipeline with Standard Scaler and Linear Regression:

    >>> pipeline = create_pipeline(scaler_key='stand', model_key='linreg',
    ...                            cat_columns=cat_columns,
    ...                            num_columns=num_columns)
    >>> pipeline.steps
    [('stand', StandardScaler()), ('linreg', LinearRegression())]

    Example 2: Create a pipeline with One-Hot Encoding, Standard Scaler, and a
    Logistic Regression model:

    >>> pipeline = create_pipeline(transformer_keys=['ohe'],
    ...                            scaler_key='stand',
    ...                            model_key='logreg',
    ...                            cat_columns=cat_columns,
    ...                            num_columns=num_columns)
    >>> pipeline.steps
    [('ohe', ColumnTransformer(force_int_remainder_cols=False, remainder='passthrough',
                      transformers=[('ohe',
                                     OneHotEncoder(drop='if_binary',
                                                   handle_unknown='ignore'),
                                     ['ocean_proximity'])])), ('stand', StandardScaler()), ('logreg', LogisticRegression(max_iter=10000, random_state=42))]

    Example 3: Create a pipeline with KNN Imputer, One-Hot Encoding, Polynomial
    Transformation, Log Transformation, Standard Scaler, and Gradient Boost
    Regressor for the model:

    >>> pipeline = create_pipeline(imputer_key='knn_imputer',
    ...                            transformer_keys=['ohe', 'poly2', 'log'],
    ...                            scaler_key='stand',
    ...                            model_key='boost_reg',
    ...                            cat_columns=cat_columns,
    ...                            num_columns=num_columns)
    >>> pipeline.steps
    [('knn_imputer', KNNImputer()), ('ohe_poly2_log', ColumnTransformer(force_int_remainder_cols=False, remainder='passthrough',
                      transformers=[('ohe',
                                     OneHotEncoder(drop='if_binary',
                                                   handle_unknown='ignore'),
                                     ['ocean_proximity']),
                                    ('poly2',
                                     PolynomialFeatures(include_bias=False),
                                     ['longitude', 'latitude', 'housing_median_age',
                                      'total_rooms', 'total_bedrooms', 'population',
                                      'households', 'median_income']),
                                    ('log',
                                     FunctionTransformer(func=<ufunc 'log1p'>,
                                                         validate=True),
                                     ['longitude', 'latitude', 'housing_median_age',
                                      'total_rooms', 'total_bedrooms', 'population',
                                      'households', 'median_income'])])), ('stand', StandardScaler()), ('boost_reg', GradientBoostingRegressor(random_state=42))]
    """
    # Check for configuration file parameter, if none, use default in library
    if config is None:
        # If no column lists are provided, raise an error
        if not cat_columns and not num_columns:
            raise ValueError("If no config is provided, cat_columns and num_columns must be passed.")
        config = {
            'imputers': {
                'knn_imputer': KNNImputer().set_output(transform='pandas'),
                'knn20_imputer': KNNImputer().set_output(transform='pandas'),
                'simple_imputer': SimpleImputer(),
                'zero_imputer': SimpleImputer(),
                'mean_imputer': SimpleImputer()
            },
            'transformers': {
                'ohe': (OneHotEncoder(drop='if_binary', handle_unknown='ignore'),
                        cat_columns),
                'ord': (OrdinalEncoder(), cat_columns),
                'poly2': (PolynomialFeatures(degree=2, include_bias=False),
                          num_columns),
                'poly2_bias': (PolynomialFeatures(degree=2, include_bias=True),
                               num_columns),
                'poly3': (PolynomialFeatures(degree=3, include_bias=False),
                          num_columns),
                'poly3_bias': (PolynomialFeatures(degree=3, include_bias=True),
                               num_columns),
                'log': (FunctionTransformer(np.log1p, validate=True),
                        num_columns)
            },
            'scalers': {
                'stand': StandardScaler(),
                'robust': RobustScaler(),
                'minmax': MinMaxScaler()
            },
            'selectors': {
                'rfe_logreg': RFE(LogisticRegression(max_iter=max_iter,
                                                     random_state=random_state,
                                                     class_weight=class_weight)),
                'sfs_logreg': SequentialFeatureSelector(
                    LogisticRegression(max_iter=max_iter,
                                       random_state=random_state,
                                       class_weight=class_weight)),
                'sfs_linreg': SequentialFeatureSelector(LinearRegression()),
                'sfs_7': SequentialFeatureSelector(LinearRegression(),
                                                   n_features_to_select=7),
                'sfs_6': SequentialFeatureSelector(LinearRegression(),
                                                   n_features_to_select=6),
                'sfs_5': SequentialFeatureSelector(LinearRegression(),
                                                   n_features_to_select=5),
                'sfs_4': SequentialFeatureSelector(LinearRegression(),
                                                   n_features_to_select=4),
                'sfs_3': SequentialFeatureSelector(LinearRegression(),
                                                   n_features_to_select=3),
                'sfs_bw': SequentialFeatureSelector(LinearRegression(),
                                                    direction='backward')
            },
            'models': {
                'linreg': LinearRegression(),
                'knn_reg': KNeighborsRegressor(),
                'ttr_log': TransformedTargetRegressor(
                    regressor=LinearRegression(), func=np.log, inverse_func=np.exp),
                'svr': SVR(),
                'logreg': LogisticRegression(max_iter=max_iter,
                                             random_state=random_state,
                                             class_weight=class_weight),
                'ridge': Ridge(random_state=random_state),
                'lasso': Lasso(random_state=random_state),
                'tree_class': DecisionTreeClassifier(random_state=random_state),
                'tree_reg': DecisionTreeRegressor(random_state=random_state),
                'knn': KNeighborsClassifier(),
                'svm': SVC(random_state=random_state, class_weight=class_weight),
                'svm_proba': SVC(random_state=random_state, probability=True,
                                 class_weight=class_weight),
                'forest_reg': RandomForestRegressor(random_state=random_state),
                'forest_class': RandomForestClassifier(random_state=random_state,
                                                       class_weight=class_weight),
                'vot_reg': VotingRegressor([('linreg', LinearRegression()),
                                            ('knn_reg', KNeighborsRegressor()),
                                            ('tree_reg',
                                             DecisionTreeRegressor(
                                                 random_state=random_state)),
                                            ('ridge', Ridge(
                                                random_state=random_state)),
                                            ('svr', SVR())]),
                'bag_reg': BaggingRegressor(random_state=random_state),
                'bag_class': BaggingClassifier(random_state=random_state),
                'boost_reg': GradientBoostingRegressor(
                    random_state=random_state),
                'boost_class': GradientBoostingClassifier(
                    random_state=random_state),
                'ada_class': AdaBoostClassifier(random_state=random_state),
                'ada_reg': AdaBoostRegressor(random_state=random_state)
            },
            'no_scale': ['tree_class', 'tree_reg', 'forest_reg', 'forest_class'],
            'no_poly': ['knn', 'tree_reg', 'tree_class', 'forest_reg', 'forest_class']
        }

    # Initialize an empty list for the transformation steps
    steps = []

    # Function to add imputer to the pipeline steps
    def add_imputer_step():
        if imputer_key is not None:
            imputer_obj = config['imputers'][imputer_key]
            steps.append((imputer_key, imputer_obj))

    # Add imputer step before column transformers if impute_first is True
    if impute_first:
        add_imputer_step()

    # If transformers are provided, add them to the steps
    if transformer_keys is not None:
        transformer_steps = []

        for key in (transformer_keys if isinstance(transformer_keys, list) else [transformer_keys]):
            transformer, cols = config['transformers'][key]
            if key in ['poly2', 'poly2_bias', 'poly3', 'poly3_bias'] and model_key in config['no_poly']:
                continue  # Skip polynomial transformers if the model is in 'no_poly'
            transformer_steps.append((key, transformer, cols))

        # Create column transformer
        col_trans = ColumnTransformer(transformer_steps, remainder='passthrough', force_int_remainder_cols=False)
        transformer_name = '_'.join(transformer_keys) \
            if isinstance(transformer_keys, list) else transformer_keys
        steps.append((transformer_name, col_trans))
        if debug:
            print('col_trans:', col_trans)
            print('transformer_name:', transformer_name)
            print('steps:', steps)


    # Add imputer step after column transformers if impute_first is False
    if not impute_first:
        add_imputer_step()

    # If a scaler is provided, add it to the steps, unless model listed in
    # no_scale config
    if scaler_key is not None and model_key not in config['no_scale']:
        scaler_obj = config['scalers'][scaler_key]
        steps.append((scaler_key, scaler_obj))

    # If a selector is provided, add it to the steps
    if selector_key is not None:
        selector_obj = config['selectors'][selector_key]
        steps.append((selector_key, selector_obj))

    # If a model is provided, add it to the steps
    if model_key is not None:
        model_obj = config['models'][model_key]
        steps.append((model_key, model_obj))

    if debug:
        print('steps:', steps)
    # Create and return pipeline
    return Pipeline(steps)


def create_results_df() -> pd.DataFrame:
    """
    Initialize the results_df DataFrame with the columns required for
    `iterate_model`.

    This function creates a new DataFrame with the following columns:
    'Iteration', 'Train MSE', 'Test MSE', 'Train RMSE', 'Test RMSE',
    'Train MAE', 'Test MAE', 'Train R^2 Score', 'Test R^2 Score',
    'Pipeline', 'Best Grid Params', 'Note', 'Date'.

    Create a `results_df` with this function, and then pass it as a parameter
    to `iterate_model`. The results of each model iteration will be appended
    to `results_df`.

    Returns
    -------
    pd.DataFrame
        The initialized results_df DataFrame.

    Examples
    --------
    Create a DataFrame with the columns required for `iterate_model`:

    >>> results_df = create_results_df()
    >>> results_df.columns
    Index(['Iteration', 'Train MSE', 'Test MSE', 'Train RMSE', 'Test RMSE',
           'Train MAE', 'Test MAE', 'Train R^2 Score', 'Test R^2 Score',
           'Pipeline', 'Best Grid Params', 'Note', 'Date'],
          dtype='object')
    """
    columns = [
        'Iteration', 'Train MSE', 'Test MSE', 'Train RMSE', 'Test RMSE',
        'Train MAE', 'Test MAE', 'Train R^2 Score', 'Test R^2 Score',
        'Pipeline', 'Best Grid Params', 'Note', 'Date'
    ]

    return pd.DataFrame(columns=columns)



def eval_model(
        *,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        class_map: Dict[Any, Any] = None,
        estimator: Optional[Any] = None,
        x_test: Optional[np.ndarray] = None,
        class_type: Optional[str] = None,
        pos_label: Optional[Any] = 1,
        threshold: float = 0.5,
        multi_class: str = 'ovr',
        average: str = 'macro',
        title: Optional[str] = None,
        model_name: str = 'Model',
        class_weight: Optional[str] = None,
        decimal: int = 2,
        bins: int = 10,
        bin_strategy: str = None,
        plot: bool = False,
        figsize: Tuple[int, int] = (12, 11),
        figmulti: float = 1.7,
        conf_fontsize: int = 14,
        return_metrics: bool = False,
        output: bool = True,
        debug: bool = False
) -> Optional[Dict[str, Union[int, float]]]:
    """
    Evaluate a classification model's performance and plot results.

    This function provides a comprehensive evaluation of a binary or multi-class
    classification model based on `y_test` (the actual target values) and `y_pred`
    (the predicted target values). It displays a text-based classification report
    enhanced with True/False Positives/Negatives (if binary), and 4 charts if
    `plot` is True: Confusion Matrix, Histogram of Predicted Probabilities, ROC
    Curve, and Precision-Recall Curve.

    If `class_type` is 'binary', it will treat this as a binary classification.
    If `class_type` is 'multi', it will treat this as a multi-class problem. If
    `class_type` is not specified, it will be detected based on the number of
    unique values in `y_test`. To plot the curves or adjust the `threshold`
    (default 0.5), both `x_test` and `estimator` must be provided so that
    proababilities can be calculated.

    For binary classification, `pos_label` is required. This defaults to 1 as an
    integer, but can be set to any value that matches one of the values in
    `y_test` and `y_pred`. The `class_map` can be used to provide display names
    for the classes. If not provided, the actual class values will be used.

    A number of classification metrics are shown in the report: Accuracy,
    Precision, Recall, F1, and ROC AUC. In addition, for binary classification,
    True Positive Rate, False Positive Rate, True Negative Rate, and False
    Negative Rate are shown. The metrics are calculated at the default threshold
    of 0.5, but can be adjusted with the `threshold` parameter.

    You can customize the `title` of the report completely, or pass the
    `model_name` and it will be displayed in a dynamically generated title. You
    can also specify the number of `decimal` places to show, and size of the
    figure (`fig_size`). For multi-class, you can set a `figmulti` scaling factor
    for the plot.

    You can set the `class_weight` as a display only string that is not used in
    any functions within `eval_model`. This is useful if you trained the model
    with a 'balanced' class_weight, and now want to pass that to this report to
    see the effects.

    A dictionary of metrics can be returned if `return_metrics` is True, and
    the output can be disabled by setting `output` to False. These are used by
    parent functions (ex: `compare_models`) to gather the data into a DataFrame
    of the results.

    Use this function to assess the performance of a trained classification
    model. You can experiment with different thresholds to see how they affect
    metrics like Precision, Recall, False Positive Rate and False Negative
    Rate. The plots make it easy to see if you're getting good separation and
    maximum area under the curve.

    Parameters
    ----------
    y_test : np.ndarray
        The true labels of the test set.
    y_pred : np.ndarray
        The predicted labels of the test set.
    class_map : Dict[Any, Any], optional
        A dictionary mapping class labels to their string representations.
        Default is None.
    estimator : Any, optional
        The trained estimator object used for prediction. Required for
        generating probabilities. Default is None.
    x_test : np.ndarray, optional
        The test set features. Required for generating probabilities.
        Default is None.
    class_type : str, optional
        The type of classification problem. Can be 'binary' or 'multi'.
        If not provided, it will be inferred from the number of unique labels.
        Default is None.
    pos_label : Any, optional
        The positive class label for binary classification.
        Default is 1.
    threshold : float, optional
        The threshold for converting predicted probabilities to class labels.
        Default is 0.5.
    multi_class : str, optional
        The method for handling multi-class ROC AUC calculation.
        Can be 'ovr' (one-vs-rest) or 'ovo' (one-vs-one).
        Default is 'ovr'.
    average : str, optional
        The averaging method for multi-class classification metrics.
        Can be 'macro', 'micro', 'weighted', or 'samples'.
        Default is 'macro'.
    title : str, optional
        The title for the plots. Default is None.
    model_name : str, optional
        The name of the model for labeling the plots. Default is 'Model'.
    class_weight : str, optional
        The class weight settings used for training the model.
        Default is None.
    decimal : int, optional
        The number of decimal places to display in the output and plots.
        Default is 4.
    bins : int, optional
        The number of bins for the predicted probabilities histogram when
        `bin_strategy` is None. Default is 10.
    bin_strategy : str, optional
        The strategy for determining the number of bins for the predicted
        probabilities histogram. Can be 'sqrt', 'sturges', 'rice', 'freed',
        'scott', or 'doane'. Default is None.
    plot : bool, optional
        Whether to display the evaluation plots. Default is False.
    figsize : Tuple[int, int], optional
        The figure size for the plots in inches. Default is (12, 11).
    figmulti : float, optional
        The multiplier for the figure size in multi-class classification.
        Default is 1.7.
    conf_fontsize : int, optional
        The font size for the numbers in the confusion matrix. Default is 14.
    return_metrics : bool, optional
        Whether to return the evaluation metrics as a dictionary.
        Default is False.
    output : bool, optional
        Whether to print the evaluation results. Default is True.
    debug : bool, optional
        Whether to print debug information. Default is False.

    Returns
    -------
    metrics : Dict[str, Union[int, float]], optional
        A dictionary containing the evaluation metrics. Returned only if
        `return_metrics` is True and the classification type is binary.

    Examples
    --------
    Prepare data and model for the examples:

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import SVC
    >>> X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.4, 0.6],
    ...                            random_state=42)
    >>> class_map = {0: 'Malignant', 1: 'Benign'}
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    ...                                                     random_state=42)
    >>> model = SVC(kernel='linear', probability=True, random_state=42)
    >>> model.fit(X_train, y_train)
    SVC(kernel='linear', probability=True, random_state=42)
    >>> y_pred = model.predict(X_test)

    Example 1: Basic evaluation with default settings:

    >>> eval_model(y_test=y_test, y_pred=y_pred)  #doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    Binary Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
               0       0.76      0.74      0.75        72
               1       0.85      0.87      0.86       128
    <BLANKLINE>
        accuracy                           0.82       200
       macro avg       0.81      0.80      0.80       200
    weighted avg       0.82      0.82      0.82       200
    <BLANKLINE>
                   Predicted:0         1
    Actual: 0                53        19
    Actual: 1                17        111
    <BLANKLINE>
    True Positive Rate / Sensitivity: 0.87
    True Negative Rate / Specificity: 0.74
    False Positive Rate / Fall-out: 0.26
    False Negative Rate / Miss Rate: 0.13
    <BLANKLINE>
    Positive Class: 1 (1)
    Threshold: 0.5

    Example 2: Evaluation with custom settings:

    >>> eval_model(y_test=y_test, y_pred=y_pred, estimator=model, x_test=X_test,
    ...            class_type='binary', class_map=class_map, pos_label=0,
    ...            threshold=0.35, model_name='SVM', class_weight='balanced',
    ...            decimal=4, plot=True, figsize=(13, 13), conf_fontsize=18,
    ...            bins=20)   #doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    SVM Binary Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
          Benign     0.9545    0.8203    0.8824       128
       Malignant     0.7444    0.9306    0.8272        72
    <BLANKLINE>
        accuracy                         0.8600       200
       macro avg     0.8495    0.8754    0.8548       200
    weighted avg     0.8789    0.8600    0.8625       200
    <BLANKLINE>
    ROC AUC: 0.9220
    <BLANKLINE>
                   Predicted:1         0
    Actual: 1                105       23
    Actual: 0                5         67
    <BLANKLINE>
    True Positive Rate / Sensitivity: 0.9306
    True Negative Rate / Specificity: 0.8203
    False Positive Rate / Fall-out: 0.1797
    False Negative Rate / Miss Rate: 0.0694
    <BLANKLINE>
    Positive Class: Malignant (0)
    Class Weight: balanced
    Threshold: 0.35

    Example 3: Evaluate model with no output and return a dictionary:

    >>> metrics = eval_model(y_test=y_test, y_pred=y_pred, estimator=model,
    ...            x_test=X_test, class_map=class_map, pos_label=0,
    ...            return_metrics=True, output=False)
    >>> print(metrics)
    {'True Positives': 53, 'False Positives': 17, 'True Negatives': 111, 'False Negatives': 19, 'TPR': 0.7361111111111112, 'TNR': 0.8671875, 'FPR': 0.1328125, 'FNR': 0.2638888888888889, 'Benign': {'precision': 0.8538461538461538, 'recall': 0.8671875, 'f1-score': 0.8604651162790697, 'support': 128.0}, 'Malignant': {'precision': 0.7571428571428571, 'recall': 0.7361111111111112, 'f1-score': 0.7464788732394366, 'support': 72.0}, 'accuracy': 0.82, 'macro avg': {'precision': 0.8054945054945055, 'recall': 0.8016493055555556, 'f1-score': 0.8034719947592532, 'support': 200.0}, 'weighted avg': {'precision': 0.819032967032967, 'recall': 0.82, 'f1-score': 0.819430068784802, 'support': 200.0}, 'ROC AUC': 0.9219835069444444, 'Threshold': 0.5, 'Class Type': 'binary', 'Class Map': {0: 'Malignant', 1: 'Benign'}, 'Positive Label': 0, 'Title': None, 'Model Name': 'Model', 'Class Weight': None, 'Multi-Class': 'ovr', 'Average': 'macro'}

    Prepare multi-class example data:

    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> X = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length',
    ...                              'petal_width'])
    >>> y = pd.Series(y)
    >>> class_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    ...                                    random_state=42)
    >>> model = SVC(kernel='linear', probability=True, random_state=42)
    >>> model.fit(X_train, y_train)
    SVC(kernel='linear', probability=True, random_state=42)
    >>> y_pred = model.predict(X_test)

    Example 4: Evaluate multi-class model with default settings:

    >>> metrics = eval_model(y_test=y_test, y_pred=y_pred, class_map=class_map,
    ...               return_metrics=True)   #doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    Multi-Class Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
          Setosa       1.00      1.00      1.00        10
      Versicolor       1.00      1.00      1.00         9
       Virginica       1.00      1.00      1.00        11
    <BLANKLINE>
        accuracy                           1.00        30
       macro avg       1.00      1.00      1.00        30
    weighted avg       1.00      1.00      1.00        30
    <BLANKLINE>
    Predicted   Setosa  Versicolor  Virginica
    Actual
    Setosa          10           0          0
    Versicolor       0           9          0
    Virginica        0           0         11
    <BLANKLINE>
    >>> print(metrics)
    {'Setosa': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 10.0}, 'Versicolor': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 9.0}, 'Virginica': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 11.0}, 'accuracy': 1.0, 'macro avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 30.0}, 'weighted avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 30.0}, 'ROC AUC': None, 'Threshold': 0.5, 'Class Type': 'multi', 'Class Map': {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}, 'Positive Label': None, 'Title': None, 'Model Name': 'Model', 'Class Weight': None, 'Multi-Class': 'ovr', 'Average': 'macro'}
    """
    # Initialize debugging, controlled via 'debug' parameter
    db = DebugPrinter(debug = debug)
    db.print('-' * 40)
    db.print('START eval_model')
    db.print('-' * 40, '\n')
    db.print('y_test shape:', y_test.shape)
    db.print('y_pred shape:', y_pred.shape)
    db.print('class_map:', class_map)
    db.print('pos_label:', pos_label)
    db.print('class_type:', class_type)
    db.print('estimator:', estimator)
    if x_test is not None:
        db.print('x_test shape:', x_test.shape)
    else:
        db.print('x_test:', x_test)
    db.print('threshold:', threshold)

    # Convert y_test DataFrame to a Series if it's not already
    if isinstance(y_test, pd.DataFrame):
        db.print('\nConverting y_test DataFrame to Series...')
        db.print('y_test shape before:', y_test.shape)
        y_test = y_test.squeeze()
        db.print('y_test shape after:', y_test.shape)

    # Convert y_test DataFrame to a Series if it's not already
    if isinstance(y_pred, pd.DataFrame):
        db.print('\nConverting y_pred DataFrame to Series...')
        db.print('y_pred shape before:', y_pred.shape)
        y_pred = y_pred.squeeze()
        db.print('y_pred shape after:', y_pred.shape)

    # Get the unique labels and display labels for the confusion matrix
    if class_map is not None:
        # Make sure class_map is a dictionary
        if isinstance(class_map, dict):
            db.print('\nGetting labels from class_map...')
            unique_labels = list(class_map.keys())
            display_labels = list(class_map.values())
        else:
            raise TypeError("class_map must be a dictionary")

        # Make sure every unique_label has a corresponding entry in y_test
        missing_labels = set(np.unique(y_test)) - set(unique_labels)
        if missing_labels:
            db.print('y_test[:5]:', list(y_test[:5]))
            db.print('set(unique_labels):', set(unique_labels))
            db.print('set(np.unique(y_test)):', set(np.unique(y_test)))
            db.print('missing_labels:', missing_labels)
            raise ValueError(f"The following labels in y_test are missing from class_map: {missing_labels}")
    else:
        db.print('\nGetting labels from unique values in y_test...')
        unique_labels = np.unique(y_test)
        display_labels = [str(label) for label in unique_labels]
        db.print('Creating class_map...')
        class_map = {label: str(label) for label in unique_labels}
        db.print('class_map:', class_map)
    db.print('unique_labels:', unique_labels)
    db.print('display_labels:', display_labels)

    # Count the number of classes
    num_classes = len(unique_labels)
    db.print('num_classes:', num_classes)

    # If class_type is not passed, auto-detect based on unique values of y_test
    if class_type is None:
        if num_classes > 2:
            class_type = 'multi'
        elif num_classes == 2:
            class_type = 'binary'
        else:
            raise ValueError(f"Check data, cannot classify. Number of classes in y_test ({num_classes}) is less than 2: {unique_labels}")
        db.print(f"\nClassification type detected: {class_type}")
        db.print("Unique values in y:", num_classes)
    elif class_type not in ['binary', 'multi']:
        # If class type is invalid, raise an error
        raise ValueError(f"Class type '{class_type}' is invalid, must be 'binary' or 'multi'. Number of classes in y_test: {num_classes}, unique labels: {unique_labels}")

    # Check to ensure num_classes matches the passed class_type
    if class_type == 'binary' and num_classes != 2:
        raise ValueError(f"Class type is {class_type}, but number of classes in y_test ({num_classes}) is not 2: {unique_labels}")
    elif class_type == 'multi' and num_classes < 3:
        raise ValueError(f"Class type is {class_type}, but number of classes in y_test ({num_classes}) is less than 3: {unique_labels}")
    elif num_classes < 2:
        raise ValueError(f"Check data, cannot classify. Class type is {class_type}, and number of classes in y_test ({num_classes}) is less than 2: {unique_labels}")

    # Evaluation for multi-class classification
    if class_type == 'multi':

        # Set pos_label to None for multi-class
        pos_label = None

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Run the classification report
        db.print('\nRun the Classification Report...')
        class_report = classification_report(y_test, y_pred, digits=decimal, target_names=display_labels,
                                             zero_division=0, output_dict=True)
        db.print('class_report:', class_report)

        # Calculate ROC AUC if we have x_test and estimator
        if x_test is not None and estimator is not None:
            db.print('\nCalculating ROC AUC...')
            roc_auc = roc_auc_score(y_test, estimator.predict_proba(x_test), multi_class=multi_class, average=average)
        else:
            roc_auc = None
        db.print('roc_auc:', roc_auc)

        if output:
            # Display the best title we can create
            if title is not None:
                print(f"\n{title}\n")
            elif model_name != 'Model':
                print(f"\n{model_name} Multi-Class Classification Report\n")
            else:
                print(f"\nMulti-Class Classification Report\n")
            # Display the classification report
            print(classification_report(y_test, y_pred, digits=decimal, target_names=display_labels, zero_division=0))

            # Display the ROC AUC
            if roc_auc is not None:
                if isinstance(roc_auc, float):
                    print(f'ROC AUC: {round(roc_auc, decimal)}\n')
                elif isinstance(roc_auc, np.ndarray):
                    # It's an array, handle different cases
                    if roc_auc.size == 1:
                        print(f'ROC AUC: {round(roc_auc[0], decimal)}\n')
                    else:
                        # If it's an array with multiple elements, print the mean value, rounded
                        mean_roc_auc = np.mean(roc_auc)
                        print(f'ROC AUC (mean): {round(mean_roc_auc, decimal)}\n')
                else:
                    # Print it raw
                    print(f'ROC AUC: {roc_auc}\n')

            # Display the class weight for reference only
            if class_weight is not None:
                print(f'Class Weight: {class_weight}\n')

            # Create a DataFrame from the confusion matrix
            df_cm = pd.DataFrame(cm, index=display_labels, columns=display_labels)
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'
            print(f'{df_cm}\n')

    # Pre-processing for binary classification
    if class_type == 'binary':

        # Check if pos_label is in unique_labels
        if pos_label not in unique_labels:
            db.print('pos_label:', pos_label)
            db.print('type(pos_label):', type(pos_label).__name__)
            db.print('unique_labels:', unique_labels)
            db.print('unique_labels[0]:', unique_labels[0])
            db.print('unique_labels[1]:', unique_labels[1])
            db.print('type(unique_labels[0]):', type(unique_labels[0]).__name__)
            db.print('type(unique_labels[1]):', type(unique_labels[1]).__name__)
            raise ValueError(f"Positive label: {pos_label} ({type(pos_label).__name__}) is not in y_test unique values: {unique_labels}. Please specify the correct 'pos_label'.")

        # Encode labels if binary classification problem
        db.print('\nEncoding labels for binary classification...')

        # Assign neg_label based on pos_label
        neg_label = np.setdiff1d(unique_labels, [pos_label])[0]
        db.print('pos_label:', pos_label)
        db.print('neg_label:', neg_label)

        # Create a label_map for encoding
        label_map = {neg_label: 0, pos_label: 1}
        db.print('label_map:', label_map)

        # Encode new labels as 0 and 1
        db.print('\nEncoding y_test and y_pred...')
        y_test_enc = np.array([label_map[label] for label in y_test])
        y_pred_enc = np.array([label_map[label] for label in y_pred])
        db.print('y_test[:5]:', list(y_test[:5]))
        db.print('y_test_enc[:5]:', y_test_enc[:5])
        db.print('y_pred[:5]:', y_pred[:5])
        db.print('y_pred_enc[:5]:', y_pred_enc[:5])
        db.print('Overwriting y_test and y_pred...')
        y_test = y_test_enc
        y_pred = y_pred_enc
        db.print('y_test[:5]:', list(y_test[:5]))
        db.print('y_pred[:5]:', y_pred[:5])

        # Create a map for the new labels
        db.print('\nGetting the display labels...')
        pos_display = class_map[pos_label]
        neg_display = class_map[neg_label]
        db.print('pos_display:', pos_display)
        db.print('neg_display:', neg_display)
        if class_map is not None:
            display_map = {0: neg_display, 1: pos_display}
        else:
            display_map = {0: str(neg_label), 1: str(pos_label)}
        db.print('display_map:', display_map)

        # Update the unique labels and display labels for the confusion matrix
        db.print('\nUpdating labels from display_map...')
        unique_labels = list(display_map.keys())
        display_labels = list(display_map.values())
        db.print('New unique_labels:', unique_labels)
        db.print('New display_labels:', display_labels)

    # Calculate the probabilities
    if class_type == 'binary' and x_test is not None and estimator is not None:
        db.print('\nCalculating probabilities...')
        pos_class_index = np.where(estimator.classes_ == pos_label)[0][0]
        db.print('estimator.classes_:', estimator.classes_)
        db.print('pos_label:', pos_label)
        db.print('pos_class_index:', pos_class_index)
        probabilities = estimator.predict_proba(x_test)[:, pos_class_index]
        all_probabilities = estimator.predict_proba(x_test)
        db.print('probabilities[:5]:', probabilities[:5])
        db.print('all_probabilities[:5]:', all_probabilities[:5])
        db.print('all_probabilities shape:', np.shape(all_probabilities))

        # Apply the threshold to the probabilities
        if plot or threshold != 0.5:
            db.print(f'\nApplying threshold {threshold} to probabilities...')
            y_pred_thresh = (probabilities >= threshold).astype(int)
            db.print('y_pred[:5]:', y_pred[:5])
            db.print('y_pred_thresh[:5]:', y_pred_thresh[:5])
            db.print('Overwriting y_pred with y_pred_thres...')
            y_pred = y_pred_thresh
            db.print('y_pred[:5]:', y_pred[:5])
        else:
            db.print(f'\nUsing default threshold of {threshold}...')
        db.print('plot:', plot)
    else:
        probabilities = None
        db.print(f'\nSkipping probabilities. class_type: {class_type}, x_test shape: {np.shape(x_test)}, estimator: {estimator.__class__.__name__}')

    # Evaluation for binary classification
    if class_type == 'binary':
        if output:
            # Display the best title we can create
            if title is not None:
                print(f"\n{title}\n")
            elif model_name != 'Model':
                print(f"\n{model_name} Binary Classification Report\n")
            else:
                print(f"\nBinary Classification Report\n")

        # Run the classification report
        db.print('\nRun the Classification Report...')
        class_report = classification_report(y_test, y_pred, labels=unique_labels, target_names=display_labels,
                                             digits=decimal, zero_division=0, output_dict=True)
        db.print('class_report:', class_report)
        if output:
            print(classification_report(y_test, y_pred, labels=unique_labels, target_names=display_labels,
                                        digits=decimal, zero_division=0))

        # Calculate the confusion matrix
        db.print('\nCalculating confusion matrix and metrics...')
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

        # Calculate the binary metrics
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tnr = tn / (tn + fp)
        fnr = fn / (fn + tp)
        db.print('cm:\n', cm)
        db.print('\ncm.ravel:', cm.ravel())
        db.print(f'TN: {tn}')
        db.print(f'FP: {fp}')
        db.print(f'FN: {fn}')
        db.print(f'TP: {tp}')

        binary_metrics = {
            "True Positives": tp,
            "False Positives": fp,
            "True Negatives": tn,
            "False Negatives": fn,
            "TPR": tpr,
            "TNR": tnr,
            "FPR": fpr,
            "FNR": fnr,
        }

    # Calculate the ROC AUC score if binary classification with probabilities
    if class_type == 'binary' and probabilities is not None:

        # Calculate ROC AUC score
        db.print('\nCalculating ROC AUC score...')
        roc_auc = roc_auc_score(y_test, probabilities, labels=unique_labels)
        db.print('y_test[:5]:', y_test[:5])
        db.print('probabilities[:5]:', probabilities[:5])
        db.print('unique_labels:', unique_labels)
        if output:
            print(f'ROC AUC: {roc_auc:.{decimal}f}\n')

        # Calculate false positive rate, true positive rate, and thresholds for ROC curve
        db.print('\nCalculating ROC curve...')
        fpr_array, tpr_array, thresholds = roc_curve(y_test, probabilities, pos_label=1)
        if len(thresholds) == 0 or len(fpr_array) == 0 or len(tpr_array) == 0:
            raise ValueError(f"Error in ROC curve calculation, at least one empty array. fpr_array length: {len(fpr_array)}, tpr_array length: {len(tpr_array)}, thresholds length: {len(thresholds)}.")
        db.print('y_test[:5]:', y_test[:5])
        db.print('probabilities[:5]:', probabilities[:5])
        db.print('Arrays from roc_curve:')
        db.print('fpr_array[:5]:', fpr_array[:5])
        db.print('tpr_array[:5]:', tpr_array[:5])
        db.print('thresholds[:5]:', thresholds[:5])

    # Print the binary classification output
    if class_type == 'binary' and output:

        # Print confusion matrix
        print(f"{'':<15}{'Predicted:':<10}{neg_label:<10}{pos_label:<10}")
        print(f"{'Actual: ' + str(neg_label):<25}{cm[0][0]:<10}{cm[0][1]:<10}")
        print(f"{'Actual: ' + str(pos_label):<25}{cm[1][0]:<10}{cm[1][1]:<10}")

        # Print evaluation metrics
        print("\nTrue Positive Rate / Sensitivity:", round(tpr, decimal))
        print("True Negative Rate / Specificity:", round(tnr, decimal))
        print("False Positive Rate / Fall-out:", round(fpr, decimal))
        print("False Negative Rate / Miss Rate:", round(fnr, decimal))
        print(f"\nPositive Class: {pos_display} ({pos_label})")
        if class_weight is not None:
            print("Class Weight:", class_weight)
        print("Threshold:", threshold)

    # Plot the evaluation metrics
    if plot and output:

        # Define a blue color for plots
        blue = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)

        # Just plot a confusion matrix for multi-class
        if class_type == 'multi':

            # Calculate the figure size for multi-class plots
            multiplier = figmulti
            max_size = 20
            size = min(len(unique_labels) * multiplier, max_size)
            figsize = (size, size)

            # Create a figure and axis for multi-class confusion matrix
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

            # Plot the confusion matrix
            cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            cm_display.plot(cmap='Blues', ax=ax1, colorbar=False)
            for text in cm_display.text_:
                for t in text:
                    t.set_fontsize(conf_fontsize - 2)  # Reduce font size for multi-class
            ax1.set_title(f'Confusion Matrix', fontsize=18, pad=15)
            ax1.set_xlabel('Predicted Label', fontsize=14, labelpad=15)
            ax1.set_ylabel('True Label', fontsize=14, labelpad=10)
            ax1.tick_params(axis='both', which='major', labelsize=10)

            plt.tight_layout()
            plt.show()

        # Just plot a confusion matrix for binary classification without probabilities
        elif class_type == 'binary' and probabilities is None:

            # Calculate the figure size for a single-chart plot
            multiplier = figmulti
            max_size = 20
            size = min(len(unique_labels) * multiplier, max_size) + 1.5  # Extra size for just 2 classes
            figsize = (size, size)

            # Create a figure and axis for a confusion matrix
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

            # Plot the confusion matrix
            cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            cm_display.plot(cmap='Blues', ax=ax1, colorbar=False)
            for text in cm_display.text_:
                for t in text:
                    t.set_fontsize(conf_fontsize)
            ax1.set_title(f'Confusion Matrix', fontsize=18, pad=15)
            ax1.set_xlabel('Predicted Label', fontsize=14, labelpad=15)
            ax1.set_ylabel('True Label', fontsize=14, labelpad=10)
            ax1.tick_params(axis='both', which='major', labelsize=10)

            plt.tight_layout()
            plt.show()

        # Plot 4 charts for binary classification
        elif class_type == 'binary' and probabilities is not None:

            # Calculate the number of bins
            if bin_strategy is not None:
                # Calculate the number of bins based on the specified strategy
                data_len = len(probabilities)
                if bin_strategy == 'sqrt':
                    num_bins = int(np.sqrt(data_len))
                elif bin_strategy == 'sturges':
                    num_bins = int(np.ceil(np.log2(data_len)) + 1)
                elif bin_strategy == 'rice':
                    num_bins = int(2 * data_len ** (1/3))
                elif bin_strategy == 'freed':
                    iqr = np.subtract(*np.percentile(probabilities, [75, 25]))
                    bin_width = 2 * iqr * data_len ** (-1/3)
                    num_bins = int(np.ceil((probabilities.max() - probabilities.min()) / bin_width))
                elif bin_strategy == 'scott':
                    std_dev = np.std(probabilities)
                    bin_width = 3.5 * std_dev * data_len ** (-1/3)
                    num_bins = int(np.ceil((probabilities.max() - probabilities.min()) / bin_width))
                elif bin_strategy == 'doane':
                    std_dev = np.std(probabilities)
                    skewness = ((np.mean(probabilities) - np.median(probabilities)) / std_dev)
                    sigma_g1 = np.sqrt(6 * (data_len - 2) / ((data_len + 1) * (data_len + 3)))
                    num_bins = int(np.ceil(np.log2(data_len) + 1 + np.log2(1 + abs(skewness) / sigma_g1)))
                else:
                    raise ValueError("Invalid bin strategy, possible values of 'bin_strategy' are 'sqrt', 'sturges', 'rice', 'freed', 'scott', and 'doane'")
            else:
                # Use default behavior of bins=10 for X axis range of 0 to 1.0
                num_bins = bins

            # Create a figure and subplots for binary classification plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

            # 1. Confusion Matrix
            cm_matrix = ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred, labels=unique_labels,
                                                                display_labels=display_labels, cmap='Blues', colorbar=False, normalize=None, ax=ax1)
            for text in cm_matrix.text_:
                for t in text:
                    t.set_fontsize(conf_fontsize)
            ax1.set_title(f'Confusion Matrix', fontsize=18, pad=15)
            ax1.set_xlabel('Predicted Label', fontsize=14, labelpad=15)
            ax1.set_ylabel('True Label', fontsize=14, labelpad=10)
            ax1.tick_params(axis='both', which='major', labelsize=11)

            # 2. Histogram of Predicted Probabilities
            ax2.hist(probabilities, color=blue, edgecolor='black', alpha=0.7, bins=num_bins, label=f'{model_name} Probabilities')
            ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold: {threshold:.{decimal}f}')
            ax2.set_title('Histogram of Predicted Probabilities', fontsize=18, pad=15)
            ax2.set_xlabel('Probability', fontsize=14, labelpad=15)
            ax2.set_ylabel('Frequency', fontsize=14, labelpad=10)
            ax2.set_xticks(np.arange(0, 1.1, 0.1))
            ax2.legend()

            # 3. ROC Curve
            ax3.plot([0, 1], [0, 1], color='grey', linestyle=':', label='Chance Baseline')
            ax3.plot(fpr_array, tpr_array, color=blue, marker='.', lw=2, label=f'{model_name} ROC Curve')
            ax3.scatter(fpr, tpr, color='red', s=80, zorder=5, label=f'Threshold {threshold:.{decimal}f}')
            ax3.axvline(x=fpr, ymax=tpr-0.027, color='red', linestyle='--', lw=1,
                        label=f'TPR: {tpr:.{decimal}f}, FPR: {fpr:.{decimal}f}')
            ax3.axhline(y=tpr, xmax=fpr+0.04, color='red', linestyle='--', lw=1)
            ax3.set_xticks(np.arange(0, 1.1, 0.1))
            ax3.set_yticks(np.arange(0, 1.1, 0.1))
            ax3.set_ylim(0,1.05)
            ax3.set_xlim(-0.05,1.0)
            ax3.grid(which='both', color='lightgrey', linewidth=0.5)
            ax3.set_title('ROC Curve', fontsize=18, pad=15)
            ax3.set_xlabel('False Positive Rate', fontsize=14, labelpad=15)
            ax3.set_ylabel('True Positive Rate', fontsize=14, labelpad=10)
            ax3.legend(loc='lower right')

            # 4. Precision-Recall Curve
            db.print('\nCalculating precision-recall curve...')
            db.print('y_test[:5]:', y_test[:5])
            db.print('probabilities[:5]:', probabilities[:5])
            db.print('pos_label:', pos_label)
            precision_array, recall_array, _ = precision_recall_curve(y_test, probabilities, pos_label=1)
            db.print('precision_array[:5]:', precision_array[:5])
            db.print('recall_array[:5]:', recall_array[:5])
            precision = class_report[pos_display]['precision']
            recall = class_report[pos_display]['recall']
            db.print('precision:', precision)
            db.print('recall:', recall)

            # Plot the Precision-Recall curve
            ax4.plot(recall_array, precision_array, marker='.', label=f'{model_name} Precision-Recall', color=blue)
            ax4.scatter(recall, precision, color='red', s=80, zorder=5, label=f'Threshold: {threshold:.{decimal}f}')
            ax4.axvline(x=recall, ymax=precision-0.025, color='red', linestyle='--', lw=1,
                        label=f'Precision: {precision:.{decimal}f}, Recall: {recall:.{decimal}f}')
            ax4.axhline(y=precision, xmax=recall-0.025, color='red', linestyle='--', lw=1)
            ax4.set_xticks(np.arange(0, 1.1, 0.1))
            ax4.set_yticks(np.arange(0, 1.1, 0.1))
            ax4.set_ylim(0,1.05)
            ax4.set_xlim(0,1.05)
            ax4.grid(which='both', color='lightgrey', linewidth=0.5)
            ax4.set_title('Precision-Recall Curve', fontsize=18, pad=15)
            ax4.set_xlabel('Recall', fontsize=14, labelpad=15)
            ax4.set_ylabel('Precision', fontsize=14, labelpad=10)
            ax4.legend(loc='lower left')

            plt.tight_layout()
            plt.show()

    # Package up the metrics if requested
    if return_metrics:

        # Custom metrics dictionary
        db.print('\nPackaging metrics dictionary...')
        custom_metrics = {
            "ROC AUC": roc_auc,
            "Threshold": threshold,
            "Class Type": class_type,
            "Class Map": class_map,
            "Positive Label": pos_label,
            "Title": title,
            "Model Name": model_name,
            "Class Weight": class_weight,
            "Multi-Class": multi_class,
            "Average": average
        }

        # Assemble the final metrics based on class type
        if class_type == 'binary':
            metrics = {**binary_metrics, **class_report, **custom_metrics}
        else:
            metrics = {**class_report, **custom_metrics}
        db.print('metrics:', metrics)

        # Return a dictionary of metrics
        return metrics

def iterate_model(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model: Optional[str] = None,
        imputer: Optional[str] = None,
        transformers: Optional[Union[List[str], str]] = None,
        scaler: Optional[str] = None,
        selector: Optional[str] = None,
        drop: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        iteration: str = '1',
        note: str = '',
        save: bool = False,
        save_df: Optional[pd.DataFrame] = None,
        export: bool = False,
        plot: bool = False,
        coef: bool = False,
        perm: bool = False,
        vif: bool = False,
        cross: bool = False,
        cv_folds: int = 5,
        grid: bool = False,
        grid_params: Optional[str] = None,
        grid_cv: Optional[str] = None,
        grid_score: str = 'r2',
        grid_verbose: int = 1,
        search_type: str = 'grid',
        random_state: int = 42,
        n_jobs: Optional[int] = None,
        decimal: int = 2,
        lowess: bool = False,
        timezone: str = 'UTC',
        debug: bool = False
) -> Tuple[pd.DataFrame, Pipeline, Optional[Dict[str, Any]]]:
    """
    Iterate and evaluate a model pipeline with specified parameters.

    This function creates a pipeline from specified parameters for imputers,
    column transformers, scalers, feature selectors, and models. Parameters must
    be defined in a configuration dictionary containing the sections described
    below. If `config` is not defined, the `create_pipeline` function will revert
    to the default config embedded in its code. After creating the pipeline, it
    fits the pipeline to the passed training data, and evaluates performance with
    both test and training data. There are options to see plots of residuals and
    actuals vs. predicted, save results to a save_df with user-defined note,
    display coefficients, calculate permutation feature importance, variance
    inflation factor (VIF), and perform cross-validation.

    If `grid` is set to True, a Grid Search CV will run to find the best hyper-
    parameters. You must also specify a `grid_params` string that matches a key
    in the `config['params']` dictionary. This needs to point to a dictionary
    whose keys exactly match the name of the pipeline steps and parameter you
    want to search. See the example config. You can also specify a different
    `grid_score` and control the `grid_verbose` level (set it to 4 to see a
    full log). If you want to do a Randomized Grid Search, set `search_type` to
    'random'. `random_state` defaults to 42. `n_jobs` are None by default, but
    you can increase the number (however, you may not see the real-time output
    of the search if you have `grid_verbose` set high).

    When `iterate_model` is run, the `create_pipeline` function is called to
    create a pipeline from the specified parameters:

    * `imputer_key` (str) is selected from `config['imputers']`
    * `transformer_keys` (list or str) are selected from `config['transformers']`
    * `scaler_key` (str) is selected from `config['scalers']`
    * `selector_key` (str) is selected from `config['selectors']`
    * `model_key` (str) is selected from `config['models']`
    * `config['no_scale']` lists model keys that should not be scaled.
    * `config['no_poly']` lists models that should not be polynomial transformed.

    Here is an example of the configuration dictionary structure. It is based on
    what `create_pipeline` requires to assemble the pipeline. But it adds some
    additional configuration parameters only required by `iterate_model`, which
    are `params` (grid search parameters) and `cv` (cross-validation parameters):

    >>> config = {  # doctest: +SKIP
    ...     'imputers': {
    ...         'knn_imputer': KNNImputer().set_output(transform='pandas'),
    ...         'simple_imputer': SimpleImputer()
    ...     },
    ...     'transformers': {
    ...         'ohe': (OneHotEncoder(drop='if_binary', handle_unknown='ignore'),
    ...                 cat_columns),
    ...         'ord': (OrdinalEncoder(), cat_columns),
    ...         'poly2': (PolynomialFeatures(degree=2, include_bias=False),
    ...                   num_columns),
    ...         'log': (FunctionTransformer(np.log1p, validate=True),
    ...                 num_columns)
    ...     },
    ...     'scalers': {
    ...         'stand': StandardScaler(),
    ...         'minmax': MinMaxScaler()
    ...     },
    ...     'selectors': {
    ...         'rfe_logreg': RFE(LogisticRegression(max_iter=max_iter,
    ...                                         random_state=random_state,
    ...                                         class_weight=class_weight)),
    ...         'sfs_linreg': SequentialFeatureSelector(LinearRegression())
    ...     },
    ...     'models': {
    ...         'linreg': LinearRegression(),
    ...         'logreg': LogisticRegression(max_iter=max_iter,
    ...                                      random_state=random_state,
    ...                                      class_weight=class_weight),
    ...         'tree_class': DecisionTreeClassifier(random_state=random_state),
    ...         'tree_reg': DecisionTreeRegressor(random_state=random_state)
    ...     },
    ...     'no_scale': ['tree_class', 'tree_reg'],
    ...     'no_poly': ['tree_class', 'tree_reg'],
    ...     'params': {
    ...         'sfs': {
    ...             'Selector: sfs__n_features_to_select': np.arange(3, 13, 1),
    ...         },
    ...         'linreg': {
    ...             'Model: linreg__fit_intercept': [True],
    ...         },
    ...         'ridge': {
    ...             'Model: ridge__alpha': np.array([0.001, 0.1, 1, 10, 100, 1000, 10000, 100000]),
    ...         }
    ...     },
    ...     'cv': {
    ...         'kfold_5': KFold(n_splits=5, shuffle=True, random_state=42),
    ...         'kfold_10': KFold(n_splits=10, shuffle=True, random_state=42),
    ...         'skf_5': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    ...         'skf_10': StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    ...     }
    ... }

    In addition to the configuration file, you will need to define any column
    lists if you want to target certain transformations to a subset of columns.
    For example, you might define a 'ohe' transformer for One-Hot Encoding, and
    reference 'ohe_columns' or 'cat_columns' in its definition in the config.

    When `iterate_model` completes, it will print out the results and performance
    metrics, as well as any requested charts. It will return the best model, and
    also the grid search results (if a grid search was ran). In addition, if
    `save = True` it will append the results to a global variable `results_df`.
    This should be created using `create_results_df` beforehand. If `export=True`
    it will save the best model to disk using joblib dump with a timestamp.

    Use this function to iterate and evaluate different model pipeline
    configurations, analyze their performance, and select the best model. With one
    line of code, you can quickly explore a change to the model pipeline, or
    grid search parameters, and see how it impacts performance. You can also
    track the results of these iterations in a `results_df` DataFrame that can
    be used to evaluate the best model, or to plot the progress you made from
    each iteration.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training feature set.
    x_test : pd.DataFrame
        Test feature set.
    y_train : pd.Series
        Training target set.
    y_test : pd.Series
        Test target set.
    model : str, optional
        Key for the model to be used (ex: 'linreg', 'lasso', 'ridge').
    imputer : str, optional
        Key for the imputer to be applied (ex: 'simple_imputer').
    transformers : List[str], optional
        List of transformation keys to apply (ex: ['ohe', 'poly2']).
    scaler : str, optional
        Key for the scaler to be applied (ex: 'stand').
    selector : str, optional
        Key for the selector to be applied (ex: 'sfs').
    drop : List[str], optional
        List of columns to be dropped from the training and test sets.
    iteration : str, optional
        A string identifier for the iteration (default '1').
    note : str, optional
        Any note or comment to be added for the iteration.
    save : bool, optional
        Boolean flag to save the results to the global results dataframe.
    save_df : pd.DataFrame, optional
        DataFrame to store the results of each iteration.
    export : bool, optional
        Boolean flag to export the trained model.
    plot : bool, optional
        Flag to plot residual and actual vs predicted for train/test data.
    coef : bool, optional
        Flag to print and plot model coefficients.
    perm : bool, optional
        Flag to compute and display permutation feature importance.
    vif : bool, optional
        Flag to calculate and display Variance Inflation Factor.
    cross : bool, optional
        Flag to perform cross-validation and print results.
    cv_folds : int, optional
        Number of folds for cross-validation if cross=True (default 5).
    config : Dict[str, Any], optional
        Configuration dictionary for pipeline construction.
    grid : bool, optional
        Flag to perform grid search for hyperparameter tuning.
    grid_params : str, optional
        Key for the grid search parameters in the config dictionary.
    grid_cv : str, optional
        Key for the grid search cross-validation in the config dictionary.
    grid_score : str, optional
        Scoring metric for grid search (default 'r2').
    grid_verbose : int, optional
        Verbosity level for grid search (default 1).
    search_type : str, optional
        Choose type of grid search: 'grid' for GridSearchCV, or 'random' for
        RandomizedSearchCV. Default is 'grid'.
    random_state : int, optional
        Random state seed, necessary for reproducability with RandomizedSearchCV.
        Default is 42.
    n_jobs : int, optional
        Number of jobs to run in parallel for Grid Search or Randomized Search.
        Default is None.
    decimal : int, optional
        Number of decimal places for displaying metrics (default 2).
    lowess : bool, optional
        Flag to display lowess curve in residual plots (default False).
    timezone : str, optional
        Timezone to be used for timestamps. Default is 'UTC'.
    debug : bool, optional
        Flag to show debugging information.

    Returns
    -------
    Tuple[DataFrame, Pipeline, Optional[Dict[str, Any]]]
        A tuple containing the save_df DataFrame, the best model pipeline, and
        the grid search results (if grid=True, else None).

    Examples
    --------
    Prepare some sample data for the examples:

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_regression(n_samples=100, n_features=5, noise=0.5,
    ...                        random_state=42)
    >>> X_df = pd.DataFrame(X,
    ...                     columns=[f"Feature {i+1}" for i in range(X.shape[1])])
    >>> y_df = pd.DataFrame(y, columns=['Target'])
    >>> X_train, X_test, y_train, y_test = train_test_split(X_df, y_df,
    ...     test_size=0.2, random_state=42)

    Create column lists and set some variables:

    >>> num_columns = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5']
    >>> cat_columns = []
    >>> random_state = 42

    Create a dataframe to store the results of each iteration (optional):

    >>> results_df = create_results_df()

    Create a custom configuration file:

    >>> my_config = {
    ...     'imputers': {
    ...         'simple_imputer': SimpleImputer()
    ...     },
    ...     'transformers': {
    ...         'poly2': (PolynomialFeatures(degree=2, include_bias=False),
    ...                   num_columns)
    ...     },
    ...     'scalers': {
    ...         'stand': StandardScaler()
    ...     },
    ...     'selectors': {
    ...         'sfs_linreg': SequentialFeatureSelector(LinearRegression())
    ...     },
    ...     'models': {
    ...         'linreg': LinearRegression(),
    ...         'ridge': Ridge(random_state=random_state)
    ...     },
    ...     'no_scale': [],
    ...     'no_poly': [],
    ...     'params': {
    ...         'linreg': {
    ...             'linreg__fit_intercept': [True],
    ...         },
    ...         'ridge': {
    ...             'ridge__alpha': np.array([0.1, 1, 10, 100]),
    ...         }
    ...     },
    ...     'cv': {
    ...         'kfold_5': KFold(n_splits=5, shuffle=True, random_state=42)
    ...     }
    ... }

    Example 1: Iterate a linear regression model with default parameters:

    >>> model = iterate_model(X_train, X_test, y_train, y_test,
    ...                       model='linreg')  #doctest: +ELLIPSIS
    <BLANKLINE>
    ITERATION 1 RESULTS
    <BLANKLINE>
    Pipeline: linreg
    ...UTC
    <BLANKLINE>
    Predictions:
                              Train            Test
    MSE:                       0.20            0.28
    RMSE:                      0.45            0.53
    MAE:                       0.36            0.42
    R^2 Score:                 1.00            1.00

    Example 2: Iterate a pipeline with transformers and scalers

    >>> results_df, model, grid = iterate_model(X_train, X_test, y_train, y_test,
    ...     transformers=['poly2'], scaler='stand', model='ridge', iteration='2',
    ...     grid=True, grid_params='ridge', grid_cv='kfold_5', plot=True,
    ...     coef=True, perm=True, vif=True, config=my_config,
    ...     save=True, save_df=results_df)  #doctest: +ELLIPSIS
    <BLANKLINE>
    ITERATION 2 RESULTS
    <BLANKLINE>
    Pipeline: poly2 -> stand -> ridge
    ...UTC
    <BLANKLINE>
    Grid Search:
    <BLANKLINE>
    Fitting 5 folds for each of 4 candidates, totalling 20 fits
    <BLANKLINE>
    Best Grid mean score (r2): 1.00
    Best Grid parameters: ridge__alpha: 0.1
    <BLANKLINE>
    Predictions:
                              Train            Test
    MSE:                       0.20            0.43
    RMSE:                      0.45            0.66
    MAE:                       0.37            0.50
    R^2 Score:                 1.00            1.00
    <BLANKLINE>
    Permutation Feature Importance:
      Feature Importance Mean Importance Std
    Feature 2            0.83           0.14
    Feature 1            0.47           0.03
    Feature 4            0.33           0.03
    Feature 3            0.31           0.03
    Feature 5            0.11           0.01
    <BLANKLINE>
    Variance Inflation Factor:
     Features  VIF Multicollinearity
    Feature 1 1.03               Low
    Feature 4 1.03               Low
    Feature 5 1.02               Low
    Feature 3 1.02               Low
    Feature 2 1.01               Low
    <BLANKLINE>
    <BLANKLINE>
    Coefficients:
                    Feature Coefficient
    1             Feature 1       65.68
    2             Feature 2       90.96
    3             Feature 3       53.72
    4             Feature 4       56.56
    5             Feature 5       33.85
    6           Feature 1^2        0.02
    7   Feature 1 Feature 2        0.03
    8   Feature 1 Feature 3       -0.16
    9   Feature 1 Feature 4       -0.08
    10  Feature 1 Feature 5        0.03
    11          Feature 2^2       -0.03
    12  Feature 2 Feature 3       -0.03
    13  Feature 2 Feature 4        0.07
    14  Feature 2 Feature 5       -0.05
    15          Feature 3^2       -0.06
    16  Feature 3 Feature 4        0.03
    17  Feature 3 Feature 5       -0.07
    18          Feature 4^2        0.01
    19  Feature 4 Feature 5       -0.04
    20          Feature 5^2       -0.05
    """
    # Drop specified columns from Xn_train and Xn_test
    if drop is not None:
        x_train = x_train.drop(columns=drop)
        x_test = x_test.drop(columns=drop)
        if debug:
            print('Drop:', drop)
            print('x_train.columns', x_train.columns)
            print('x_test.columns', x_test.columns)

    # Check for configuration file parameter, if none, use default in library
    if config is None:
        num_columns = x_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_columns = x_train.select_dtypes(exclude=[np.number]).columns.tolist()
        if debug:
            print('Config:', config)
            print('num_columns:', num_columns)
            print('cat_columns:', cat_columns)
    else:
        num_columns = None
        cat_columns = None

    # Create a pipeline from transformer and model parameters
    if debug:
        print('BEFORE create_pipeline')
        print('transformers:', transformers)
    pipe = create_pipeline(imputer_key=imputer, transformer_keys=transformers, scaler_key=scaler,
                           selector_key=selector, model_key=model, config=config,
                           cat_columns=cat_columns, num_columns=num_columns, debug=debug)
    if debug:
        print('AFTER create_pipeline')
        print('Pipeline:', pipe)
        print('Pipeline Parameters:', pipe.get_params())

    # Construct format string
    format_str = f',.{decimal}f'

    # Print some metadata
    print(f'\nITERATION {iteration} RESULTS\n')
    pipe_steps = " -> ".join(pipe.named_steps.keys())
    print(f'Pipeline: {pipe_steps}')
    if note: print(f'Note: {note}')
    # Get the current date and time
    current_time = datetime.now(pytz.timezone(timezone))
    timestamp = current_time.strftime(f'%b %d, %Y %I:%M %p {timezone}')
    print(f'{timestamp}\n')

    if cross:
        print('Cross Validation:\n')
    # Before fitting the pipeline, check if cross-validation is desired:
    if cross:
        # Flatten yn_train for compatibility
        yn_train_flat = y_train.values.flatten() if isinstance(y_train, pd.Series) else np.array(y_train).flatten()
        cv_scores = cross_val_score(pipe, x_train, yn_train_flat, cv=cv_folds, scoring='r2')

        print(f'Cross-Validation (R^2) Scores for {cv_folds} Folds:')
        for i, score in enumerate(cv_scores, 1):
            print(f'Fold {i}: {score:{format_str}}')
        print(f'Average: {np.mean(cv_scores):{format_str}}')
        print(f'Standard Deviation: {np.std(cv_scores):{format_str}}\n')

    if grid:

        # Select the appropriate search method
        if search_type == 'grid':
            print('Grid Search:\n')
            grid = GridSearchCV(pipe, param_grid=config['params'][grid_params], scoring=grid_score, verbose=grid_verbose, cv=config['cv'][grid_cv], n_jobs=n_jobs)
        elif search_type == 'random':
            print('Randomized Grid Search:\n')
            grid = RandomizedSearchCV(pipe, param_distributions=config['params'][grid_params], scoring=grid_score, verbose=grid_verbose, cv=config['cv'][grid_cv], random_state=random_state, n_jobs=n_jobs)
        else:
            raise ValueError("search_type should be either 'grid' for GridSearchCV, or 'random' for RandomizedSearchCV")

        if debug:
            print('Grid: ', grid)
            print('Grid Parameters: ', grid.get_params())
        # Fit the grid and predict
        grid.fit(x_train, y_train)
        #best_model = grid.best_estimator_
        best_model = grid
        yn_train_pred = grid.predict(x_train)
        yn_test_pred = grid.predict(x_test)
        if debug:
            print("First 10 actual train values:", y_train[:10])
            print("First 10 predicted train values:", yn_train_pred[:10])
            print("First 10 actual test values:", y_test[:10])
            print("First 10 predicted test values:", yn_test_pred[:10])
        best_grid_params = grid.best_params_
        best_grid_score = grid.best_score_
        best_grid_estimator = grid.best_estimator_
        best_grid_index = grid.best_index_
        grid_results = grid.cv_results_
    else:
        best_grid_params = np.nan
        best_grid_score = np.nan
        # Fit the pipeline and predict
        pipe.fit(x_train, y_train)
        best_model = pipe
        yn_train_pred = pipe.predict(x_train)
        yn_test_pred = pipe.predict(x_test)

    # MSE
    yn_train_mse = mean_squared_error(y_train, yn_train_pred)
    yn_test_mse = mean_squared_error(y_test, yn_test_pred)

    # RMSE
    yn_train_rmse = np.sqrt(yn_train_mse)
    yn_test_rmse = np.sqrt(yn_test_mse)

    # MAE
    yn_train_mae = mean_absolute_error(y_train, yn_train_pred)
    yn_test_mae = mean_absolute_error(y_test, yn_test_pred)

    # R^2 Score
    if grid:
        if grid_score == 'r2':
            train_score = grid.score(x_train, y_train)
            test_score = grid.score(x_test, y_test)
        else:
            train_score = 0
            test_score = 0
    else:
        train_score = pipe.score(x_train, y_train)
        test_score = pipe.score(x_test, y_test)

    # Print Grid best parameters
    if grid:
        print(f'\nBest Grid mean score ({grid_score}): {best_grid_score:{format_str}}')
        #print(f'Best Grid parameters: {best_grid_params}\n')
        param_str = ', '.join(f"{key}: {value}" for key, value in best_grid_params.items())
        print(f"Best Grid parameters: {param_str}\n")
        #print(f'Best Grid estimator: {best_grid_estimator}')
        #print(f'Best Grid index: {best_grid_index}')
        #print(f'Grid results: {grid_results}')

    # Print the results
    print('Predictions:')
    print(f'{"":<15} {"Train":>15} {"Test":>15}')
    #print('-'*55)
    print(f'{"MSE:":<15} {yn_train_mse:>15{format_str}} {yn_test_mse:>15{format_str}}')
    print(f'{"RMSE:":<15} {yn_train_rmse:>15{format_str}} {yn_test_rmse:>15{format_str}}')
    print(f'{"MAE:":<15} {yn_train_mae:>15{format_str}} {yn_test_mae:>15{format_str}}')
    print(f'{"R^2 Score:":<15} {train_score:>15{format_str}} {test_score:>15{format_str}}')

    # Save the results if save=True
    if save:
        if save_df is not None:
            results_df = save_df
        else:
            # Create results_df if it doesn't exist with predefined columns
            results_df = pd.DataFrame(columns=['Iteration', 'Train MSE', 'Test MSE', 'Train RMSE', 'Test RMSE',
                                               'Train MAE', 'Test MAE', 'Train R^2 Score', 'Test R^2 Score',
                                               'Best Grid Mean Score', 'Best Grid Params', 'Pipeline',
                                               'Note', 'Date'])

        # Store results in a dictionary
        results = {
            'Iteration': iteration,
            'Train MSE': yn_train_mse,
            'Test MSE': yn_test_mse,
            'Train RMSE': yn_train_rmse,
            'Test RMSE': yn_test_rmse,
            'Train MAE': yn_train_mae,
            'Test MAE': yn_test_mae,
            'Train R^2 Score': train_score,
            'Test R^2 Score': test_score,
            'Best Grid Mean Score': best_grid_score,
            'Best Grid Params': best_grid_params,
            'Pipeline': pipe_steps,
            'Note': note,
            'Date': timestamp
        }

        # Convert the results dictionary to a pd.Series
        results_series = pd.Series(results)

        # Append the series to the DataFrame
        results_df = pd.concat([results_df, results_series.to_frame().T], ignore_index=True)

    # Permutation Feature Importance
    if perm:
        print("\nPermutation Feature Importance:")
        if grid:
            pfi_df = calc_pfi(grid, x_train, y_train)
        else:
            pfi_df = calc_pfi(pipe, x_train, y_train)
        print(pfi_df.to_string(index=False))

    # Variance Inflation Factor
    if vif:
        print("\nVariance Inflation Factor:")

        if pipe is not None:
            if debug:
                print(type(pipe))
                print(pipe.steps)
                print(hasattr(pipe, '_final_estimator'))

            if pipe.steps:
                last_step = pipe.steps[-1][1]
                if hasattr(last_step, 'transform'):
                    vif_data = pipe.transform(x_train)
                else:
                    vif_data = x_train
            else:
                vif_data = x_train

            # Convert vif_data to a DataFrame if it's a NumPy array
            if isinstance(vif_data, np.ndarray):
                vif_df = pd.DataFrame(vif_data, columns=[f"Feature_{i}" for i in range(vif_data.shape[1])])
            else:
                vif_df = vif_data

            vif_results = calc_vif(vif_df)
            print(vif_results.to_string(index=False))
        elif grid is not None:
            if grid.best_estimator_.steps:
                last_step = grid.best_estimator_.steps[-1][1]
                if hasattr(last_step, 'transform'):
                    vif_data = grid.best_estimator_.transform(x_train)
                else:
                    vif_data = x_train
            else:
                vif_data = x_train

            # Convert vif_data to a DataFrame if it's a NumPy array
            if isinstance(vif_data, np.ndarray):
                vif_df = pd.DataFrame(vif_data, columns=[f"Feature_{i}" for i in range(vif_data.shape[1])])
            else:
                vif_df = vif_data

            vif_results = calc_vif(vif_df)
            print(vif_results.to_string(index=False))
        else:
            print("No pipeline or grid found. Skipping VIF calculation.")

    if plot:
        print('')
        y_train = y_train.values.flatten() if isinstance(y_train, pd.Series) else np.array(y_train).flatten()
        y_test = y_test.values.flatten() if isinstance(y_test, pd.Series) else np.array(y_test).flatten()

        yn_train_pred = yn_train_pred.flatten()
        yn_test_pred = yn_test_pred.flatten()

        # Generate residual plots
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        sns.residplot(x=y_train, y=yn_train_pred, lowess=lowess, scatter_kws={'s': 30, 'edgecolor': 'white'}, line_kws={'color': 'red', 'lw': '1'})
        plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands))
        plt.title(f'Training Residuals - Iteration {iteration}')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')

        plt.subplot(1, 2, 2)
        sns.residplot(x=y_test, y=yn_test_pred, lowess=lowess, scatter_kws={'s': 30, 'edgecolor': 'white'}, line_kws={'color': 'red', 'lw': '1'})
        plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands))
        plt.title(f'Test Residuals - Iteration {iteration}')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')

        plt.tight_layout()
        plt.show()

        # Generate predicted vs actual plots
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=y_train, y=yn_train_pred, s=30, edgecolor='white')
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linewidth=1)
        plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands))
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Training Predicted vs. Actual - Iteration {iteration}')

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=y_test, y=yn_test_pred, s=30, edgecolor='white')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=1)
        plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands))
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Test Predicted vs. Actual - Iteration {iteration}')

        plt.tight_layout()
        plt.show()

    # Calculate coefficients if model supports
    if coef:
        # Extract features and coefficients using the function
        coefficients_df = extract_coef(
            grid.best_estimator_ if grid else pipe, x_train, format=False, debug=debug
        )

        # Check if there are any non-NaN coefficients
        if coefficients_df['Coefficient'].notna().any():
            # Ensure the coefficients are shaped as a 2D numpy array
            coefficients = coefficients_df[['Coefficient']].values
        else:
            coefficients = None

        # Debugging information
        if debug:
            print("Coefficients: ", coefficients)
            # Print the number of coefficients and selected rows
            print(f"Number of coefficients: {len(coefficients)}")

        if coefficients is not None:
            print("\nCoefficients:")
            with pd.option_context('display.float_format', lambda x: f'{x:,.{decimal}f}'.replace('-0.00', '0.00')):
                coefficients_df.index = coefficients_df.index + 1
                coefficients_df = coefficients_df.rename(columns={'feature_name': 'Feature', 'coefficients': 'Value'})
                print(coefficients_df)

            if plot:
                # Flatten the coefficients array for plotting
                coefficients = coefficients_df['Coefficient'].values.flatten()
                feature_names = coefficients_df['Feature'].values.flatten()

                plt.figure(figsize=(12, 4))
                x_values = range(len(feature_names))
                plt.bar(x_values, coefficients, align='center')

                # Set the x-ticks labels to be the feature names
                plt.xticks(x_values, feature_names, rotation=90, ha='right')

                plt.xlabel('')
                plt.ylabel('')
                plt.title('Coefficients')
                plt.axhline(y=0, color='black', linestyle='dotted', lw=1)
                plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands))
                plt.tight_layout()
                plt.show()

    if export:
        filestamp = current_time.strftime('%Y%m%d_%H%M%S')
        filename = f'iteration_{iteration}_model_{filestamp}.joblib'
        dump(best_model, filename)

        # Check if file exists and display a message
        if os.path.exists(filename):
            print(f"\nModel saved successfully as {filename}")
        else:
            print(f"\nFAILED to save the model as {filename}")

    if save:
        if grid:
            return results_df, best_model, grid_results
        else:
            return results_df, best_model
    else:
        if grid:
            return best_model, grid_results
        else:
            return best_model


def plot_acf_residuals(
        results: Any,
        figsize: Tuple[float, float] = (12, 8),
        rotation: int = 45,
        bins: int = 30,
        lags: int = 40,
        legend_loc: str = 'best',
        show_std: bool = True,
        pacf_method: str = 'ywm',
        alpha: float = 0.7
) -> None:
    """
    Plot residuals, histogram, ACF, and PACF of a time series ARIMA model.

    This function takes the results of an ARIMA model and creates a 2x2 grid of
    plots to visualize the residuals, their histogram, autocorrelation function
    (ACF), and partial autocorrelation function (PACF). The residuals are plotted
    with lines indicating standard deviations from the mean if `show_std` is True.

    Use this function in time series analysis to assess the residuals of an ARIMA
    model and check for any patterns or autocorrelations that may indicate
    inadequacies in the model.

    Parameters
    ----------
    results : Any
        The result object typically obtained after fitting an ARIMA model.
        This object should have a `resid` attribute containing the residuals.
    figsize : Tuple[float, float], optional
        The size of the figure in inches, specified as (width, height).
        Default is (12, 7).
    rotation : int, optional
        The rotation angle for the x-axis tick labels in degrees. Default is 45.
    bins : int, optional
        The number of bins to use in the histogram of residuals. Default is 30.
    lags : int, optional
        The number of lags to plot in the ACF and PACF plots. Default is 40.
    legend_loc : str, optional
        The location of the legend in the residual plot and histogram.
        Default is 'best'.
    show_std : bool, optional
        Whether to display the standard deviation lines in the residual plot and
        histogram. Default is True.
    pacf_method : str, optional
        The method to use for the partial autocorrelation function (PACF) plot.
        Default is 'ywm'. Other options include 'ywadjusted', 'ywmle' and 'ols'.
    alpha : float, optional
        The transparency of the histogram bars, between 0 and 1. Default is 0.7.

    Returns
    -------
    None
        The function displays a 2x2 grid of plots using matplotlib.

    Examples
    --------
    Prepare the necessary data and model:

    >>> from statsmodels.tsa.arima.model import ARIMA
    >>> import numpy as np
    >>> data = np.random.random(100)
    >>> model = ARIMA(data, order=(1, 1, 1))
    >>> results = model.fit()

    Example 1: Plot residuals with default parameters:

    >>> plot_acf_residuals(results)

    Example 2: Plot residuals without standard deviation lines:

    >>> plot_acf_residuals(results, show_std=False)

    Example 3: Plot residuals with custom figsize, bins, and PACF method:

    >>> plot_acf_residuals(results, figsize=(12, 10), bins=20, pacf_method='ols')
    """
    residuals = results.resid
    std_dev = residuals.std()

    fig, ax = plt.subplots(2, 2, figsize=figsize)

    # Plot residuals
    ax[0, 0].axhline(y=0, color='lightgrey', linestyle='-', lw=1)
    if show_std:
        ax[0, 0].axhline(y=std_dev, color='red', linestyle='--', lw=1,
                         label=f'1 STD (±{std_dev:.2f})')
        ax[0, 0].axhline(y=2*std_dev, color='red', linestyle=':', lw=1,
                         label=f'2 STD (±{2*std_dev:.2f})')
        ax[0, 0].axhline(y=-std_dev, color='red', linestyle='--', lw=1)
        ax[0, 0].axhline(y=2*-std_dev, color='red', linestyle=':', lw=1)
        ax[0, 0].legend(loc=legend_loc)
    ax[0, 0].plot(residuals, label='Residuals')
    ax[0, 0].tick_params(axis='x', rotation=rotation)
    ax[0, 0].set_title('Residuals from ARIMA Model', fontsize=15, pad=10)
    ax[0, 0].set_xlabel("Time", fontsize=12, labelpad=10)
    ax[0, 0].set_ylabel("Residual Value", fontsize=12, labelpad=10)

    # Plot histogram of residuals
    ax[0, 1].hist(residuals, bins=bins, edgecolor='k', alpha=alpha)
    if show_std:
        ax[0, 1].axvline(x=std_dev, color='red', linestyle='--', lw=1,
                         label=f'1 STD (±{std_dev:.2f})')
        ax[0, 1].axvline(x=2*std_dev, color='red', linestyle=':', lw=1,
                         label=f'2 STD (±{2*std_dev:.2f})')
        ax[0, 1].axvline(x=-std_dev, color='red', linestyle='--', lw=1)
        ax[0, 1].axvline(x=2*-std_dev, color='red', linestyle=':', lw=1)
        ax[0, 1].legend(loc=legend_loc)
    ax[0, 1].set_title("Histogram of Residuals", fontsize=15, pad=10)
    ax[0, 1].set_xlabel("Residual Value", fontsize=12, labelpad=10)
    ax[0, 1].set_ylabel("Frequency", fontsize=12, labelpad=10)

    # Plot ACF of residuals
    plot_acf(residuals, lags=lags, ax=ax[1, 0])
    ax[1, 0].set_title("ACF of Residuals", fontsize=15, pad=10)
    ax[1, 0].set_xlabel("Lag", fontsize=12, labelpad=10)
    ax[1, 0].set_ylabel("Autocorrelation", fontsize=12, labelpad=10)

    # Plot PACF of residuals
    plot_pacf(residuals, lags=lags, ax=ax[1, 1], method=pacf_method)
    ax[1, 1].set_title("PACF of Residuals", fontsize=15, pad=10)
    ax[1, 1].set_xlabel("Lag", fontsize=12, labelpad=10)
    ax[1, 1].set_ylabel("Partial Autocorrelation", fontsize=12, labelpad=10)

    plt.tight_layout(pad=2)
    plt.show()


def plot_results(
        df: pd.DataFrame,
        metrics: Optional[Union[str, List[str]]] = None,
        select_metric: Optional[str] = None,
        select_criteria: str = 'max',
        chart_type: str = 'line',
        decimal: int = 2,
        return_df: bool = False,
        x_column: str = 'Iteration',
        y_label: str = None,
        rotation: int = 45,
        title: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Plot the results of model iterations and select the best metric.

    This function creates line plots to visualize the performance of a model over
    multiple iterations, or to compare the performance of multiple models. Specify
    one or more `metrics` columns to plot (ex: 'Train MAE', 'Test MAE') in a list,
    and specify the name of the `x_column` whose values will become the X axis of
    the plot. The default is 'Iteration', which aligns with the format of the
    'results_df' DataFrame created by the `create_results_df` function. But this
    could be any column in the provided `df` that you want to compare across
    (for example, 'Model', 'Epoch', 'Dataset').

    In addition, if you specify `select_metric` (any metric column in the `df`)
    and `select_criteria` ('min' or 'max'), the best result will be selected
    and plotted on the chart with a vertical line, dot, and a legend label that
    describes the value. The number of decimal places can be controlled by
    setting `decimal` (default is 2).

    The title of the chart will be dynamically generated if `y_label` and
    `x_column` are defined. The title will be constructed in this format:
    '{y_label} over {x_column}' (ex: 'MSE over Iteration'). However, you can
    always pass a customer title by setting `title` to any string of text. If
    none of these are defined, there will be no title on the chart.

    Use this function to easily visualize and compare the performance of a model
    across different metrics, and identify the best iteration based on a chosen
    metric and criteria.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the model evaluation results.
    metrics : Optional[Union[str, List[str]]], optional
        The metric(s) to plot. If a single string is provided, it will be converted
        to a list. If None, an error will be raised. Default is None.
    select_metric : Optional[str], optional
        The metric to use for selecting the best result. If None, then no
        best result will be selected. Default is None.
    select_criteria : str, optional
        The criteria for selecting the best result. Can be either 'max' or 'min'.
        Required if `select_metric` is specified. Default is 'max'.
    chart_type : str, optional
        The type of chart to plot. Currently only 'line' or 'bar' is supported.
        Default is 'line'.
    decimal : int, optional
        The number of decimal places to display in the plot and legend.
        Default is 2.
    return_df : bool, optional
        Whether to return the melted DataFrame used for plotting. Default is False.
    x_column : str, optional
        The column in `df` to use as the x-axis. Default is 'Iteration'.
    y_label : str, optional
        The text to display as the label for the y-axis, and to also include in
        the dynamically generated title of the chart. Default is None.
    title : Optional[str], optional
        The title of the plot. If None, a default title will be generated
        from `select_metric` and `x_column`. If `select_metric` is also None, the
        title will be blank. Default is None.
    rotation : int, optional
        The rotation angle for the x-axis tick labels in degrees. Default is 45.

    Returns
    -------
    Optional[pd.DataFrame]
        If `return_df` is True, returns the melted DataFrame used for plotting.
        Otherwise, returns None.

    Examples
    --------
    Prepare some example data:

    >>> df = pd.DataFrame({
    ...     'Iteration': [1, 2, 3, 4, 5],
    ...     'Train Accuracy': [0.8510, 0.9017, 0.8781, 0.9209, 0.8801],
    ...     'Test Accuracy': [0.8056, 0.8509, 0.8232, 0.8889, 0.8415]
    ... })

    Example 1: Plot a single metric with default parameters:

    >>> plot_results(df, metrics='Test Accuracy')

    Example 2: Plot multiple metrics, select the best result based on the
    minimum value of 'Test Accuracy', and customize the Y-axis label:

    >>> plot_results(df, metrics=['Train Accuracy', 'Test Accuracy'],
    ...              select_metric='Test Accuracy', select_criteria='max',
    ...              y_label='Accuracy')

    Example 3: Plot multiple metrics, customize the title and decimal, and
    return the melted DataFrame:

    >>> long_df = plot_results(df, metrics=['Train Accuracy', 'Test Accuracy'],
    ...              select_metric='Test Accuracy', select_criteria='max',
    ...              title='Train vs. Test Accuracy by Model Iteration',
    ...              return_df=True, decimal=4)
    >>> long_df
       Iteration          Metric   Value
    0          1  Train Accuracy  0.8510
    1          2  Train Accuracy  0.9017
    2          3  Train Accuracy  0.8781
    3          4  Train Accuracy  0.9209
    4          5  Train Accuracy  0.8801
    5          1   Test Accuracy  0.8056
    6          2   Test Accuracy  0.8509
    7          3   Test Accuracy  0.8232
    8          4   Test Accuracy  0.8889
    9          5   Test Accuracy  0.8415

    Example 4: Plot a single metric as a bar chart:

    >>> plot_results(df, metrics='Test Accuracy', chart_type='bar')

    Example 5: Plot multiple metrics as a bar chart:

    >>> plot_results(df, metrics=['Train Accuracy', 'Test Accuracy'],
    ...              select_metric='Test Accuracy', select_criteria='max',
    ...              y_label='Accuracy', chart_type='bar')
    """
    # Check if metrics are provided
    if metrics is None:
        raise ValueError("At least one metric must be provided.")

    # Convert metrics to a list if it's a single string
    if isinstance(metrics, str):
        metrics = [metrics]

    # Melt dataframe to long format
    df_long = df.melt(id_vars=[x_column], value_vars=metrics, var_name='Metric', value_name='Value')

    # Start the plot
    plt.figure(figsize=(12, 6))
    plt.grid(linestyle='-', linewidth=0.5, color='#DDDDDD', zorder=0)

    # Decide between lineplot and barplot
    if chart_type == 'line':
        sns.lineplot(data=df_long, x=x_column, y='Value', hue='Metric', zorder=2)
    elif chart_type == 'bar':
        sns.barplot(data=df_long, x=x_column, y='Value', hue='Metric', zorder=2)

    # Plot the best result if select_metric is specified
    if select_metric is not None:

        # Check if select_criteria is valid
        if select_criteria not in ['max', 'min']:
            raise ValueError("To select a best result, select_criteria must be either 'max' or 'min'.")

        # Find iteration with min/max metric value
        if select_criteria == 'max':
            best_iter = df[df[select_metric] == df[select_metric].max()][x_column].values[0]
            best_val = df[df[x_column] == best_iter][select_metric].values[0]
        else:
            best_iter = df[df[select_metric] == df[select_metric].min()][x_column].values[0]
            best_val = df[df[x_column] == best_iter][select_metric].values[0]

        # Get y-coordinate of the vertical line to position the dot
        y_coord = df_long[(df_long[x_column] == best_iter) & (df_long['Metric'] == select_metric)]['Value'].values[0]

        # Format the best_val with decimal places and commas
        best_val_formatted = f'{best_val:,.{decimal}f}'

        # Plot the best result
        if chart_type == 'line':
            # Plot the vertical dotted line
            plt.axvline(x=best_iter, color='green', linestyle='--', zorder=3,
                        label=f"{x_column} {best_iter}: {select_metric}: {best_val_formatted}")
            # Plot the dot
            plt.scatter(best_iter, y_coord, color='green', s=60, zorder=3)
        elif chart_type == 'bar':
            # Plot the horizontal dotted line
            plt.axhline(y=best_val, color='green', linestyle='--', zorder=3,
                        label=f"{x_column} {best_iter}: {select_metric}: {best_val_formatted}")

    # Continue the plot
    plt.legend(loc='best')

    # Format the X axis
    plt.xticks(df[x_column].unique(), rotation=rotation)
    plt.xlabel(x_column, fontsize=14, labelpad=10)

    # Plot the title, with whatever parameters we have
    if title is None:
        if y_label is not None:
            plt.title(f'{y_label} over {x_column}', fontsize=18, pad=15)
        else:
            plt.title('')
    else:
        plt.title(f'{title}', fontsize=18, pad=15)

    # Custom formatter that adds commas and respects decimal parameter
    def format_tick(value, pos):
        return f'{value:,.{decimal}f}'

    # Format the Y axis
    if y_label is not None:
        plt.ylabel(y_label, fontsize=14, labelpad=10)
    else:
        plt.ylabel('Value', fontsize=14, labelpad=10)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_tick))

    plt.show()

    # Return the long format df if requested
    if return_df:
        return df_long.reset_index(drop=True)


def plot_train_history(
        model=None,
        history=None,
        metrics: Optional[List[str]] = None,
        plot_loss: bool = True
) -> None:
    """
    Visualize the training history of a fitted Keras model or history dictionary.

    This function creates a grid of subplots to display the training and validation
    metrics over the epochs. You can pass a fitted model, in which case the history
    will be extracted from it. Alternatively, you can pass the history dictionary
    itself. This function will automatically detect the metrics present in the
    history and plot them all, unless a specific list of metrics is provided.
    The loss is plotted by default, but can be excluded by setting `plot_loss` to
    False.

    Use this function to quickly analyze the model's performance during training
    and identify potential issues such as overfitting or underfitting.

    Parameters
    ----------
    model : keras.Model, optional
        The fitted Keras model whose training history will be plotted. Default is None.
    history : dict, optional
        A direct history dictionary obtained from the fitting process. Default is None.
    metrics : List[str], optional
        A list of metric names to plot. If None, all metrics found in the history will be plotted,
        excluding 'loss' unless explicitly listed. Default is None.
    plot_loss : bool, optional
        Whether to plot the training and validation loss. Default is True.

    Returns
    -------
    None
        The function displays the plot and does not return any value.

    Examples
    --------
    Prepare a simple example model:

    >>> model = Sequential([
    ...     Input(shape=(8,)),
    ...     Dense(10, activation='relu'),
    ...     Dense(1, activation='sigmoid')
    ... ])
    >>> model.compile(optimizer='adam', loss='binary_crossentropy',
    ...               metrics=['accuracy', 'precision', 'recall'])

    Fit the model on some random data:

    >>> import numpy as np
    >>> X = np.random.rand(100, 8)
    >>> y = np.random.randint(0, 2, size=(100, 1))
    >>> model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2,
    ...           verbose=0)  #doctest: +ELLIPSIS
    <keras...callbacks.history.History object at 0x...>
    >>> history = model.history.history

    Example 1: Plot all metrics in the training history from a model:

    >>> plot_train_history(model)

    Example 2: Plot the training history with specific metrics:

    >>> plot_train_history(model, metrics=['accuracy', 'precision'])

    Example 3: Plot the training history without the loss:

    >>> plot_train_history(model, plot_loss=False)

    Example 4: Plot the training history of a model without validation data:

    >>> model.fit(X, y, epochs=10, batch_size=32, verbose=0)  #doctest: +ELLIPSIS
    <keras...callbacks.history.History object at 0x...>
    >>> plot_train_history(model)

    Example 5: Plot the training history from a history dictionary:

    >>> plot_train_history(history=history)
    """
    # Determine the history source
    if model is not None:
        if not hasattr(model, 'history') or model.history is None:
            raise ValueError("The model has not been fitted yet. Please fit the model before plotting.")
        history_data = model.history.history
    elif history is not None:
        if not isinstance(history, dict):
            raise TypeError("The 'history' parameter must be a dictionary.")
        history_data = history
    else:
        raise ValueError("Either a fitted 'model' or 'history' dictionary is required for plotting.")

    # Auto-detect metrics if not provided, excluding loss
    if metrics is None:
        metrics = [key for key in history_data.keys() if not key.startswith('val_') and key != 'loss']

    # Filter out metrics not in history
    metrics = [metric for metric in metrics if metric in history_data or 'val_' + metric in history_data]

    # Calculate the number of plots
    total_plots = (1 if plot_loss and 'loss' in history_data else 0) + len(metrics)
    rows = math.ceil(total_plots / 2)
    cols = 2 if total_plots > 1 else 1

    # Create subplots
    fig, axs = plt.subplots(rows, cols, figsize=(12, 5.5 * rows))
    axs = np.array(axs).reshape(-1) if total_plots > 1 else np.array([axs])

    plot_index = 0

    # Plot Loss if required
    if plot_loss and 'loss' in history_data:
        axs[plot_index].plot(history_data['loss'], label='Training Loss', marker='.')
        if 'val_loss' in history_data:
            axs[plot_index].plot(history_data['val_loss'], label='Validation Loss', marker='.')
        axs[plot_index].set_title('Loss', fontsize=18, pad=15)
        axs[plot_index].set_xlabel('Epoch', fontsize=14, labelpad=15)
        axs[plot_index].set_ylabel('Loss', fontsize=14, labelpad=10)
        axs[plot_index].grid(which='both', color='lightgrey', linewidth=0.5)
        axs[plot_index].legend()
        plot_index += 1

    # Plot specified metrics and their validation counterparts if present
    for metric in metrics:
        if metric in history_data:
            axs[plot_index].plot(history_data[metric], label=f'Training {metric.capitalize()}', marker='.')
        val_metric = 'val_' + metric
        if val_metric in history_data:
            axs[plot_index].plot(history_data[val_metric], label=f'Validation {metric.capitalize()}', marker='.')
        axs[plot_index].set_title(metric.capitalize(), fontsize=18, pad=15)
        axs[plot_index].set_xlabel('Epoch', fontsize=14, labelpad=15)
        axs[plot_index].set_ylabel(metric.capitalize(), fontsize=14, labelpad=10)
        axs[plot_index].grid(which='both', color='lightgrey', linewidth=0.5)
        axs[plot_index].legend()
        plot_index += 1

    # Hide any unused axes in case of an odd number of total plots
    for idx in range(plot_index, rows * cols):
        axs[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

