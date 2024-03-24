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
    - :func:`~datawaza.model.create_pipeline` - Create a custom pipeline for data preprocessing and modeling.
    - :func:`~datawaza.model.create_results_df` - Initialize the results_df DataFrame with the columns required for `iterate_model`.
    - :func:`~datawaza.model.eval_model` - Produce a detailed evaluation report for a classification model.
    - :func:`~datawaza.model.iterate_model` - Iterate and evaluate a model pipeline with specified parameters.
    - :func:`~datawaza.model.plot_acf_residuals` - Plot residuals, histogram, ACF, and PACF of a time series ARIMA model.
    - :func:`~datawaza.model.plot_results` - Plot the results of model iterations and select the best metric.
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
                             ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, precision_recall_curve,
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
from datawaza.tools import calc_pfi, calc_vif, extract_coef, log_transform, thousands

# Typing imports
from typing import Optional, Union, Tuple, List, Dict, Any

# TensorFlow and Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warning on import
import tensorflow as tf
from scikeras.wrappers import KerasClassifier


# Functions
def compare_models(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.25,
        model_list: Optional[List[Any]] = None,
        search_type: str = 'grid',
        grid_params: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5,
        ohe_columns: Optional[List[str]] = None,
        drop: str = 'if_binary',
        plot_perf: bool = False,
        scorer: str = 'accuracy',
        neg_display: str = 'Class 0',
        pos_display: str = 'Class 1',
        pos_label: Any = 1,
        random_state: int = 42,
        decimal: int = 4,
        verbose: int = 4,
        title: Optional[str] = None,
        fig_size: Tuple[int, int] = (12, 6),
        figmulti: float = 1.5,
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
        config: Optional[Dict[str, Any]] = None,
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
        show_time: bool = True,
        debug: bool = False
) -> pd.DataFrame:
    """
    Find the best classification model and hyper-parameters for a dataset by
    automating the workflow for multiple models and comparing results.

    This function integrates a number of steps in a typical classification model
    workflow, and it does this for multiple models, all with one command line:
    (1) Auto-detecting single vs. multi-class classification problems, (2) Option
    to Under-sample or Over-smple imbalanced data, (3) Option to use a sub-sample
    of data for SVC or KNN, which can be computation intense, (3) Ability to split
    the Train/Test data at a specified ratio, (4) Creation of a multiple-step
    Pipeline, including Imputation, multiple Column Transformer/Encoding steps,
    Scaling, Feature selection, and the Model, (5) Grid Search of hyper-parameters,
    either full or random, (6) Calculating performance metrics from the standard
    Classification Report (Accuracy, Precision, Recall, F1) but also with ROC AUC,
    and if binary, True Positive Rate, True Negative Rate, False Positive Rate,
    False Negative Rate, (7) Evaluating this performance based on a customizable
    Threshold, (8) Visually showing performance by plotting (a) a Confusion
    Matrix, and if binary, (b) a Histogram of Predicted Probabilities, (c) an ROC
    Curve, and (d) a Precision-Recall Curve. (9) Save all the results in a
    DataFrame for reference and comparison, and (10) Option to plot the results to
    visually compare performance of the specified metric across multiple model
    pipelines with their best parameters.

    To use this function, a configuration should be created that defines the
    desired model configurations and parameters you want to search. In the
    current version, `compare_models` supports the following models:

    - 'logreg' : LogisticRegression
    - 'tree_class' : DecisionTreeClassifier
    - 'knn_class' : KNeighborsClassifier
    - 'svm_class' : SVC
    - 'svm_proba' : SVC (with probability=True)
    - 'forest_class' : RandomForestClassifier
    - 'vot_class' : VotingClassifier
    - 'bag_class' : BaggingClassifier
    - 'boost_class' : GradientBoostingClassifier
    - 'ada_class' : AdaBoostClassifier
    - 'xgb_class' : XGBClassifier
    - 'keras_class' : KerasClassifier

    Specify the models you want to run and compare by adding their class names to a
    list, ex: `models = [LogisticRegression, DecisionTreeClassifier]`. Then pass
    this list as the value of the `model_list` parameter: `model_list = models`.
    You will need to use the text strings  in your configuration file, so it's
    important to note the above pairings.

    When `compare_models` is run, for each model in the `model_list`, the
    `create_pipeline` function will be called to create a pipeline from the
    specified parameters. Each model iteration will have the same pipeline
    construction, except for the final model:

    * `imputer_key` (str) is selected from `config['imputers']`
    * `transformer_keys` (list or str) are selected from `config['transformers']`
    * `scaler_key` (str) is selected from `config['scalers']`
    * `selector_key` (str) is selected from `config['selectors']`
    * `config['no_scale']` lists model keys that should not be scaled.
    * `config['no_poly']` lists models that should not be polynomial transformed.

    Note that unlike the `iterate_model` or `create_pipeline` functions, you do
    not specify the `model_key` (str), which is selected from `config['models']`.
    Instead, the `model_list` is how you specify multiple models at once.

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

    

    Use this function when you want to find the best classification model and
    hyper-parameters for a dataset, after doing any required pre-processing or
    cleaning. It is a significant time saver, replacing numerous manual coding
    steps with one command.

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The target vector.
    test_size : float, optional (default=0.25)
        The proportion of the dataset to include in the test split.
    model_list : List[Any], optional (default=None)
        A list of models to compare. If None, a ValueError is raised.
    search_type : str, optional (default='grid')
        The type of hyperparameter search to perform. Can be either 'grid'
        for GridSearchCV or 'random' for RandomizedSearchCV.
    grid_params : Dict[str, Any], optional (default=None)
        A dictionary of hyperparameter grids for each model. If None, a
        ValueError is raised.
    cv_folds : int, optional (default=5)
        The number of folds for cross-validation.
    ohe_columns : List[str], optional (default=None)
        A list of categorical columns to one-hot encode.
    drop : str, optional (default='if_binary')
        The drop strategy for one-hot encoding.
    plot_perf : bool, optional (default=False)
        Whether to plot the model performance.
    scorer : str, optional (default='accuracy')
        The scorer to use for model evaluation.
    neg_label : Any, optional (default=None)
        The negative class label.
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
    show_time : bool, optional
        Show the timestamp, disable for test cases (default True).
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
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> X = pd.DataFrame(X, columns=['sepal_length', 'sepal_width',
    ...                               'petal_length', 'petal_width'])
    >>> y = pd.Series(y)
    >>> num_columns = list(X.columns)
    >>> cat_columns = []

    Example 1: Compare models with default parameters:

    >>> model_list = [LogisticRegression, DecisionTreeClassifier]
    >>> grid_params = {
    ...     'logreg': {
    ...         'logreg__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    ...         'logreg__solver': ['newton-cg', 'lbfgs', 'saga']
    ...     },
    ...     'tree_class': {
    ...         'tree_class__max_depth': [3, 5, 7],
    ...         'tree_class__min_samples_split': [5, 10, 15],
    ...         'tree_class__criterion': ['gini', 'entropy'],
    ...         'tree_class__min_samples_leaf': [2, 4, 6]
    ...         },
    ... }
    >>> results_df = compare_models(X, y, model_list=model_list, show_time=False,
    ...                  grid_params=grid_params, verbose=1,
    ...                  num_columns=num_columns,
    ...                  cat_columns=cat_columns)  #doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    -----------------------------------------------------------------------------------------
    Starting Data Processing -
    -----------------------------------------------------------------------------------------
    <BLANKLINE>
    Classification type detected: multi
    Unique values in y: [0 1 2]
    y data type after conversion: int64
    <BLANKLINE>
    Train/Test split, test_size:  0.25
    X_train, X_test, y_train, y_test shapes:  (112, 4) (38, 4) (112,) (38,)
    <BLANKLINE>
    -----------------------------------------------------------------------------------------
    1/2: Starting LogisticRegression Grid Search -
    -----------------------------------------------------------------------------------------
    <BLANKLINE>
    Fitting 5 folds for each of 21 candidates, totalling 105 fits
    <BLANKLINE>
    Total Time: 0.0000 seconds
    Average Fit Time: 0.0000 seconds
    Inference Time: 0.0000
    Best CV Accuracy Score: 0.9640
    Train Accuracy Score: 0.9732
    Test Accuracy Score: 1.0000
    Overfit: No
    Overfit Difference: -0.0268
    Best Parameters: {'logreg__C': 10, 'logreg__solver': 'saga'}
    <BLANKLINE>
    -----------------------------------------------------------------------------------------
    2/2: Starting DecisionTreeClassifier Grid Search -
    -----------------------------------------------------------------------------------------
    <BLANKLINE>
    Fitting 5 folds for each of 54 candidates, totalling 270 fits
    <BLANKLINE>
    Total Time: 0.0000 seconds
    Average Fit Time: 0.0000 seconds
    Inference Time: 0.0000
    Best CV Accuracy Score: 0.9545
    Train Accuracy Score: 0.9643
    Test Accuracy Score: 1.0000
    Overfit: No
    Overfit Difference: -0.0357
    Best Parameters: {'tree_class__criterion': 'entropy', 'tree_class__max_depth': 5, 'tree_class__min_samples_leaf': 4, 'tree_class__min_samples_split': 5}
    >>> results_df.head()  #doctest: +NORMALIZE_WHITESPACE
                        Model  Test Size Over Sample Under Sample Resample  Total Fit Time  Fit Count  Average Fit Time  Inference Time Grid Scorer                                        Best Params  Best CV Score  Train Score  Test Score Overfit  Overfit Difference  Train Accuracy Score  Test Accuracy Score  Train Precision Score  Test Precision Score  Train Recall Score  Test Recall Score  Train F1 Score  Test F1 Score  Train ROC AUC Score  Test ROC AUC Score  Threshold  True Positives  False Positives  True Negatives  False Negatives  TPR  FPR  TNR  FNR  False Rate      Pipeline Notes Timestamp
    0      LogisticRegression       0.25        None         None     None               0        105                 0               0    Accuracy        {'logreg__C': 10, 'logreg__solver': 'saga'}       0.964032     0.973214         1.0      No           -0.026786              0.973214                  1.0               0.973437                   1.0            0.973214                1.0        0.973214            1.0             0.998586                 1.0        0.5             NaN              NaN             NaN              NaN  NaN  NaN  NaN  NaN         NaN      [logreg]  None
    1  DecisionTreeClassifier       0.25        None         None     None               0        270                 0               0    Accuracy  {'tree_class__criterion': 'entropy', 'tree_cla...       0.954545     0.964286         1.0      No           -0.035714              0.964286                  1.0               0.965096                   1.0            0.964286                1.0        0.964250            1.0             0.996583                 1.0        0.5             NaN              NaN             NaN              NaN  NaN  NaN  NaN  NaN         NaN  [tree_class]  None

    Example 2: Compare models with custom parameters and evaluation:

    >>> model_list = [SVC, XGBClassifier]
    >>> random_state = 42
    >>> class_weight = None
    >>> my_config = {
    ...     'models': {
    ...         'svm_proba': SVC(random_state=random_state, probability=True, class_weight=class_weight),
    ...         'xgb_class': XGBClassifier(random_state=random_state)
    ...     }
    ... }
    >>> grid_params = {
    ...     'svm_proba': {
    ...         'svm_proba__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    ...         'svm_proba__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    ...     },
    ...     'xgb_class': {
    ...         'xgb_class__learning_rate': [0.01, 0.1, 0.5],
    ...         'xgb_class__max_depth': [3, 5, 7],
    ...         'xgb_class__subsample': [0.7, 0.8, 0.9],
    ...         'xgb_class__colsample_bytree': [0.7, 0.8, 0.9],
    ...         'xgb_class__n_estimators': [50, 100, 200],
    ...         'xgb_class__objective': ['binary:logistic'],
    ...         'xgb_class__gamma': [0, 1, 5, 10]
    ...     }
    ... }
    >>> results_df = compare_models(X, y, model_list=model_list, show_time=False, config=my_config,
    ...                         grid_params=grid_params, cv_folds=3, search_type='random',
    ...                         scorer='f1_micro', model_eval=True, verbose=1,
    ...                         svm_proba=True, random_state=42, pos_label=1,
    ...                         num_columns=num_columns, cat_columns=cat_columns,
    ...                         fig_size=(10, 5))  #doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    -----------------------------------------------------------------------------------------
    Starting Data Processing -
    -----------------------------------------------------------------------------------------
    <BLANKLINE>
    Classification type detected: multi
    Unique values in y: [0 1 2]
    y data type after conversion: int64
    <BLANKLINE>
    Train/Test split, test_size:  0.25
    X_train, X_test, y_train, y_test shapes:  (112, 4) (38, 4) (112,) (38,)
    <BLANKLINE>
    -----------------------------------------------------------------------------------------
    1/2: Starting SVC Random Search -
    -----------------------------------------------------------------------------------------
    <BLANKLINE>
    Fitting 3 folds for each of 10 candidates, totalling 30 fits
    <BLANKLINE>
    Total Time: 0.0000 seconds
    Average Fit Time: 0.0000 seconds
    Inference Time: 0.0000
    Best CV F1 (micro) Score: 0.9552
    Train F1 (micro) Score: 0.9821
    Test F1 (micro) Score: 1.0000
    Overfit: No
    Overfit Difference: -0.0179
    Best Parameters: {'svm_proba__kernel': 'poly', 'svm_proba__C': 100}
    <BLANKLINE>
    SVC Multi-Class Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
               0       1.00      1.00      1.00        15
               1       1.00      1.00      1.00        11
               2       1.00      1.00      1.00        12
    <BLANKLINE>
        accuracy                           1.00        38
       macro avg       1.00      1.00      1.00        38
    weighted avg       1.00      1.00      1.00        38
    <BLANKLINE>
    ROC AUC: 1.0
    <BLANKLINE>
    -----------------------------------------------------------------------------------------
    2/2: Starting XGBClassifier Random Search -
    -----------------------------------------------------------------------------------------
    <BLANKLINE>
    Fitting 3 folds for each of 10 candidates, totalling 30 fits
    <BLANKLINE>
    Total Time: 0.0000 seconds
    Average Fit Time: 0.0000 seconds
    Inference Time: 0.0000
    Best CV F1 (micro) Score: 0.9462
    Train F1 (micro) Score: 0.9554
    Test F1 (micro) Score: 1.0000
    Overfit: No
    Overfit Difference: -0.0446
    Best Parameters: {'xgb_class__subsample': 0.9, 'xgb_class__objective': 'binary:logistic', 'xgb_class__n_estimators': 50, 'xgb_class__max_depth': 5, 'xgb_class__learning_rate': 0.1, 'xgb_class__gamma': 1, 'xgb_class__colsample_bytree': 0.9}
    <BLANKLINE>
    XGBClassifier Multi-Class Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
               0       1.00      1.00      1.00        15
               1       1.00      1.00      1.00        11
               2       1.00      1.00      1.00        12
    <BLANKLINE>
        accuracy                           1.00        38
       macro avg       1.00      1.00      1.00        38
    weighted avg       1.00      1.00      1.00        38
    <BLANKLINE>
    ROC AUC: 1.0
    >>> results_df.head()  #doctest: +NORMALIZE_WHITESPACE
               Model  Test Size Over Sample Under Sample Resample  Total Fit Time  Fit Count  Average Fit Time  Inference Time Grid Scorer                                        Best Params  Best CV Score  Train Score  Test Score Overfit  Overfit Difference  Train Accuracy Score  Test Accuracy Score  Train Precision Score  Test Precision Score  Train Recall Score  Test Recall Score  Train F1 Score  Test F1 Score  Train ROC AUC Score  Test ROC AUC Score  Threshold  True Positives  False Positives  True Negatives  False Negatives  TPR  FPR  TNR  FNR  False Rate     Pipeline Notes Timestamp
    0            SVC       0.25        None         None     None               0         30                 0               0  F1 (micro)  {'svm_proba__kernel': 'poly', 'svm_proba__C': ...       0.955192     0.982143         1.0      No           -0.017857              0.982143                  1.0               0.982143                   1.0            0.982143                1.0        0.982143            1.0             0.998115                 1.0        0.5             NaN              NaN             NaN              NaN  NaN  NaN  NaN  NaN         NaN  [svm_proba]  None
    1  XGBClassifier       0.25        None         None     None               0         30                 0               0  F1 (micro)  {'xgb_class__subsample': 0.9, 'xgb_class__obje...       0.946183     0.955357         1.0      No           -0.044643              0.955357                  1.0               0.955574                   1.0            0.955357                1.0        0.955357            1.0             0.997407                 1.0        0.5             NaN              NaN             NaN              NaN  NaN  NaN  NaN  NaN         NaN  [xgb_class]  None
    """
    # Check for required parameters
    if (model_list is None) or (grid_params is None):
        raise ValueError("Please specify a model_list and grid_params.")

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
        def build_scoring_function(score_type, pos_label=None, average='macro'):
            if pos_label is not None:
                # For binary classification tasks requiring a pos_label
                return (make_scorer(eval(f'{score_type}_score'), pos_label=pos_label),
                        f'{score_type.capitalize()} (pos_label={pos_label})')
            elif average in average_types:
                # For multi-class/multi-label tasks specifying an average type
                return make_scorer(eval(f'{score_type}_score'), average=average), f'{score_type.capitalize()} ({average})'
            else:
                raise ValueError(f"Invalid average type: {average}. Valid options are: {', '.join(average_types)}")

        # Determine the scorer and display name based on input
        if scorer in valid_scorers:
            if scorer in ['precision', 'recall', 'f1'] and pos_label is None:
                # Default to 'macro' average for multi-class tasks if pos_label is not specified
                scoring_function, display_name = build_scoring_function(scorer, average='macro')
            elif scorer.startswith(('precision_', 'recall_', 'f1_')):
                # Extract score type and average type from scorer string
                score_type, avg_type = scorer.split('_')
                scoring_function, display_name = build_scoring_function(score_type, average=avg_type)
            elif scorer == 'accuracy':
                scoring_function, display_name = 'accuracy', 'Accuracy'
            else:
                # Use predefined scikit-learn scorer strings for other cases
                scoring_function, display_name = scorer, scorer.capitalize()
        else:
            # Show an error message if the scorer is invalid
            raise ValueError(f"Unsupported scorer: {scorer}. Valid options are: {', '.join(valid_scorers)}")

        return scoring_function, display_name

    # Define the scorer and display name
    scorer, scorer_name = get_scorer_and_name(scorer=scorer, pos_label=pos_label)

    # Empty timestamp by default for test cases where we don't want time differences to trigger a failure
    timestamp = ''

    # Set initial timestamp for data processing
    if show_time:
        current_time = datetime.now(pytz.timezone(timezone))
        timestamp = current_time.strftime(f'%b %d, %Y %I:%M %p {timezone}')

    if output:
        print(f"\n-----------------------------------------------------------------------------------------")
        print(f"Starting Data Processing - {timestamp}")
        print(f"-----------------------------------------------------------------------------------------\n")

    # Detect the type of classification problem
    unique_y = np.unique(y)
    if len(unique_y) > 2:
        class_type = 'multi'
        average = 'weighted'
    else:
        class_type = 'binary'
        average = 'binary'

    if output:
        print(f"Classification type detected: {class_type}")
        print("Unique values in y:", unique_y)

    # Change data type of y if necessary
    if y.dtype.kind in 'biufc':  # If y is numeric
        y = y.astype(int)  # Convert to int for numeric labels
    else:
        y = y.astype(str)  # Convert to str for categorical labels

    if output:
        print(f"y data type after conversion: {y.dtype}")

    # Perform the train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify,
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
            grid = GridSearchCV(pipe, param_grid=combined_params, scoring=scorer, verbose=verbose, cv=cv_folds, n_jobs=n_jobs)
        elif search_type == 'random':
            grid = RandomizedSearchCV(pipe, param_distributions=combined_params, scoring=scorer, verbose=verbose,
                                      cv=cv_folds, random_state=random_state, n_jobs=n_jobs)
        else:
            raise ValueError("search_type should be either 'grid' for GridSearchCV, or 'random' for RandomizedSearchCV")

        return grid

    # Clean up the grid search type for display
    search_string = search_type.capitalize()

    # Main Loop: Iterate through each model in the list and run the workflow for each
    for i in range(len(model_list)):

        # Grab the model name and total count
        model = model_list[i]
        total = len(model_list)

        # Create the timestamp for this model's iteration
        if show_time:
            current_time = datetime.now(pytz.timezone(timezone))
            timestamp = current_time.strftime(f'%b %d, %Y %I:%M %p {timezone}')
        timestamp_list.append(timestamp)

        if model == LogisticRegression:
            model_name = 'LogisticRegression'
            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting LogisticRegression {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")

            # Create a pipeline from transformer and model parameters
            pipe = create_pipeline(imputer_key=imputer, transformer_keys=transformers, scaler_key=scaler,
                                   selector_key=selector, model_key='logreg', config=config,
                                   cat_columns=cat_columns, num_columns=num_columns, class_weight=class_weight,
                                   random_state=random_state, max_iter=max_iter, impute_first=impute_first)

            grid = create_grid(model_type='logreg')

            resample_list.append("None")

        elif model == DecisionTreeClassifier:
            model_name = 'DecisionTreeClassifier'
            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting DecisionTreeClassifier {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")

            # Create a pipeline from transformer and model parameters
            pipe = create_pipeline(imputer_key=imputer, transformer_keys=transformers, scaler_key=scaler,
                                   selector_key=selector, model_key='tree_class', config=config, cat_columns=cat_columns,
                                   num_columns=num_columns, class_weight=class_weight, random_state=random_state,
                                   max_iter=max_iter, impute_first=impute_first)

            grid = create_grid(model_type='tree_class')

            resample_list.append("None")

        elif model == KNeighborsClassifier:
            model_name = 'KNeighborsClassifier'
            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting KNeighborsClassifier {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")

            # Check if svm_knn_resample is set and model is SVC
            if svm_knn_resample is not None and resample_completed is False:
                X_train, y_train = resample_for_knn_svm(X_train, y_train)
                resample_completed = True
            if resample_completed:
                resample_list.append(svm_knn_resample)
            else:
                resample_list.append("None")

            # Create a pipeline from transformer and model parameters
            pipe = create_pipeline(imputer_key=imputer, transformer_keys=transformers, scaler_key=scaler,
                                   selector_key=selector, model_key='knn_class', config=config, cat_columns=cat_columns,
                                   num_columns=num_columns, class_weight=class_weight, random_state=random_state,
                                   max_iter=max_iter, impute_first=impute_first)

            grid = create_grid(model_type='knn_class')


        elif model == SVC:
            model_name = 'SVC'

            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting SVC {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")

            # Check if svm_knn_resample is set and model is SVC
            if svm_knn_resample is not None and resample_completed is False:
                X_train, y_train = resample_for_knn_svm(X_train, y_train)
                resample_completed = True
            elif resample_completed:
                resample_list.append(svm_knn_resample)
                if output:
                    print(f"Training data already resampled to {svm_knn_resample*100}% of original for KNN and SVM speed improvement")
                    print("X_train, y_train shapes: ", X_train.shape, y_train.shape)
                    print("y_train value counts: ", y_train.value_counts(), "\n")
            else:
                resample_list.append("None")

            if svm_proba:
                # Create a pipeline from transformer and model parameters
                pipe = create_pipeline(imputer_key=imputer, transformer_keys=transformers, scaler_key=scaler,
                                       selector_key=selector, model_key='svm_proba', config=config, cat_columns=cat_columns,
                                       num_columns=num_columns, class_weight=class_weight, random_state=random_state,
                                       max_iter=max_iter, impute_first=impute_first)
                grid = create_grid(model_type='svm_proba')
            else:
                # Create a pipeline from transformer and model parameters
                pipe = create_pipeline(imputer_key=imputer, transformer_keys=transformers, scaler_key=scaler,
                                       selector_key=selector, model_key='svm_class', config=config, cat_columns=cat_columns,
                                       num_columns=num_columns, class_weight=class_weight, random_state=random_state,
                                       max_iter=max_iter, impute_first=impute_first)
                grid = create_grid(model_type='svm_class')

        elif model == RandomForestClassifier:
            model_name = 'RandomForestClassifier'
            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting RandomForestClassifier {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")

            # Create a pipeline from transformer and model parameters
            pipe = create_pipeline(imputer_key=imputer, transformer_keys=transformers, scaler_key=scaler,
                                   selector_key=selector, model_key='forest_class', config=config,
                                   cat_columns=cat_columns, num_columns=num_columns, class_weight=class_weight,
                                   random_state=random_state, max_iter=max_iter, impute_first=impute_first)

            grid = create_grid(model_type='forest_class')

            resample_list.append("None")

        elif model == GradientBoostingClassifier:
            model_name = 'GradientBoostingClassifier'

            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting GradientBoostingClassifier {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")

            # Create a pipeline from transformer and model parameters
            pipe = create_pipeline(imputer_key=imputer, transformer_keys=transformers, scaler_key=scaler,
                                   selector_key=selector, model_key='boost_class', config=config, cat_columns=cat_columns,
                                   num_columns=num_columns, class_weight=class_weight, random_state=random_state,
                                   max_iter=max_iter, impute_first=impute_first)

            grid = create_grid(model_type='boost_class')

            resample_list.append("None")

        elif model == XGBClassifier:
            model_name = 'XGBClassifier'

            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting XGBClassifier {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")

            # Create a pipeline from transformer and model parameters
            pipe = create_pipeline(imputer_key=imputer, transformer_keys=transformers, scaler_key=scaler,
                                   selector_key=selector, model_key='xgb_class', config=config, cat_columns=cat_columns,
                                   num_columns=num_columns, class_weight=class_weight, random_state=random_state,
                                   max_iter=max_iter, impute_first=impute_first)

            grid = create_grid(model_type='xgb_class')

            resample_list.append("None")

        elif model == AdaBoostClassifier:
            model_name = 'AdaBoostClassifier'

            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting AdaBoostClassifier {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")

            # Create a pipeline from transformer and model parameters
            pipe = create_pipeline(imputer_key=imputer, transformer_keys=transformers, scaler_key=scaler,
                                   selector_key=selector, model_key='ada_class', config=config, cat_columns=cat_columns,
                                   num_columns=num_columns, class_weight=class_weight, random_state=random_state,
                                   max_iter=max_iter, impute_first=impute_first)

            grid = create_grid(model_type='ada_class')

            resample_list.append("None")

        elif model == KerasClassifier:
            model_name = 'KerasClassifier'
            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting KerasClassifier {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")

            tf.random.set_seed(random_state)

            # Create a pipeline from transformer and model parameters
            pipe = create_pipeline(imputer_key=imputer, transformer_keys=transformers, scaler_key=scaler,
                                   selector_key=selector, model_key='keras_class', config=config,
                                   cat_columns=cat_columns, num_columns=num_columns, class_weight=class_weight,
                                   random_state=random_state, max_iter=max_iter, impute_first=impute_first)

            grid = create_grid(model_type='keras_class')

            resample_list.append("None")

        elif model == BaggingClassifier:
            model_name = 'BaggingClassifier'
            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting BaggingClassifier {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")

            # Create a pipeline from transformer and model parameters
            pipe = create_pipeline(imputer_key=imputer, transformer_keys=transformers, scaler_key=scaler,
                                   selector_key=selector, model_key='bag_class', config=config, cat_columns=cat_columns,
                                   num_columns=num_columns, class_weight=class_weight, random_state=random_state,
                                   max_iter=max_iter, impute_first=impute_first)

            grid = create_grid(model_type='bag_class')

            resample_list.append("None")

        elif model == VotingClassifier:
            model_name = 'VotingClassifier'
            if output:
                print(f"\n-----------------------------------------------------------------------------------------")
                print(f"{i+1}/{total}: Starting VotingClassifier {search_string} Search - {timestamp}")
                print(f"-----------------------------------------------------------------------------------------\n")

            # Create a pipeline from transformer and model parameters
            pipe = create_pipeline(imputer_key=imputer, transformer_keys=transformers, scaler_key=scaler,
                                   selector_key=selector, model_key='vot_class', config=config, cat_columns=cat_columns,
                                   num_columns=num_columns, class_weight=class_weight, random_state=random_state,
                                   max_iter=max_iter, impute_first=impute_first)

            grid = create_grid(model_type='vot_class')

            resample_list.append("None")

        # Append to each list the value from this iteration, starting with model name, pipeline, etc.
        model_name_list.append(model_name)
        pipeline_list.append(list(pipe.named_steps.keys()))

        # Fit the model and measure total fit time, append to list
        start_time = time.time()
        grid.fit(X_train, y_train)
        if show_time:
            fit_time = time.time() - start_time
        else:
            fit_time = 0  # Don't show changing times for test cases
        fit_time_list.append(fit_time)
        if output:
            print(f"\nTotal Time: {fit_time:.{decimal}f} seconds")

        # Calculate average fit time (for each fold in the CV search) and append to list
        fit_count = len(grid.cv_results_['params']) * cv_folds
        fit_count_list.append(fit_count)
        if show_time:
            avg_fit_time = fit_time / fit_count
        else:
            avg_fit_time = 0  # Don't show changing times for test cases
        avg_fit_time_list.append(avg_fit_time)
        if output:
            print(f"Average Fit Time: {avg_fit_time:.{decimal}f} seconds")

        # Function to apply different thresholds for binary classification
        def apply_threshold(probs, threshold):
            return np.where(probs >= threshold, 1, 0)

        # Debugging data for detecting support of predict_proba
        if debug:
            print("grid.best_estimator_:", grid.best_estimator_)
            print("hasattr(grid.best_estimator_, 'predict_proba'):", hasattr(grid.best_estimator_, 'predict_proba'))
            print("hasattr(grid.best_estimator_, 'decision_function'):", hasattr(grid.best_estimator_, 'decision_function'))

        # Generate train predictions based on class type and threshold
        if class_type == 'binary':
            if hasattr(grid.best_estimator_, 'predict_proba'):
                # Model supports probability estimates
                if threshold != 0.5:
                    if debug:
                        print(f'Class: {class_type}, Method: predict_proba, Threshold: {threshold}, Data: Train')
                    # Get probabilities for the positive class
                    probabilities_train = grid.predict_proba(X_train)[:, 1]
                    # Apply the custom threshold to get binary predictions
                    y_train_pred = apply_threshold(probabilities_train, threshold)
                else:
                    if debug:
                        print(f'Class: {class_type}, Method: predict, Threshold: {threshold}, Data: Train')
                    # Use default predictions for binary classification
                    y_train_pred = grid.predict(X_train)
            elif hasattr(grid.best_estimator_, 'decision_function'):
                if debug:
                    print(f'Class: {class_type}, Method: decision_function, Threshold: {threshold}, Data: Train')
                # Model does not support probability estimates but has a decision function (ex: SVC without probability)
                decision_values_train = grid.decision_function(X_train)
                # Apply the custom threshold to the decision function values
                y_train_pred = apply_threshold(decision_values_train, threshold)
            else:
                if debug:
                    print(f'Class: {class_type}, Method: predict, Threshold: {threshold}, Data: Train')
                # Use default predictions if neither predict_proba nor decision_function are available
                y_train_pred = grid.predict(X_train)
        elif class_type == 'multi':
            if debug:
                print(f'Class: {class_type}, Method: predict, Threshold: {threshold}, Data: Train')
            # Use default predictions for multi-class classification
            y_train_pred = grid.predict(X_train)

        # Start tracking the inference time, or test predictions time
        start_time = time.time()

        # Generate test predictions based on class type and threshold
        if class_type == 'binary':
            if hasattr(grid.best_estimator_, 'predict_proba'):
                if threshold != 0.5:
                    if debug:
                        print(f'Class: {class_type}, Method: predict_proba, Threshold: {threshold}, Data: Test')
                    probabilities_test = grid.predict_proba(X_test)[:, 1]
                    y_test_pred = apply_threshold(probabilities_test, threshold)
                else:
                    if debug:
                        print(f'Class: {class_type}, Method: predict, Threshold: {threshold}, Data: Test')
                    y_test_pred = grid.predict(X_test)
            elif hasattr(grid.best_estimator_, 'decision_function'):
                if debug:
                    print(f'Class: {class_type}, Method: decision_function, Threshold: {threshold}, Data: Test')
                decision_values_test = grid.decision_function(X_test)
                y_test_pred = apply_threshold(decision_values_test, threshold)
            else:
                if debug:
                    print(f'Class: {class_type}, Method: predict, Threshold: {threshold}, Data: Test')
                y_test_pred = grid.predict(X_test)
        elif class_type == 'multi':
            if debug:
                print(f'Class: {class_type}, Method: predict, Threshold: {threshold}, Data: Test')
            y_test_pred = grid.predict(X_test)

        # Capture the inference time, or test predictions time
        if show_time:
            inference_time = time.time() - start_time
        else:
            inference_time = 0  # Don't show changing times for test cases
        inference_time_list.append(inference_time)
        if output:
            print(f"Inference Time: {inference_time:.{decimal}f}")

        # Calculate ROC AUC, based on class type and predict_proba support
        def calculate_roc_auc(grid, X, y, class_type, note):
            try:
                if class_type == 'multi':
                    # For multi-class classification, use predict_proba with multi_class='ovr'
                    if debug:
                        print(f'Class: {class_type}, Method: predict_proba(X), Threshold: {threshold}, Data: {note}, Score: ROC AUC')
                    return roc_auc_score(y, grid.predict_proba(X), multi_class='ovr')
                else:
                    # For binary classification, use predict_proba for positive class
                    if debug:
                        print(f'Class: {class_type}, Method: predict_proba(X)[:, 1], Threshold: {threshold}, Data: {note}, Score: ROC AUC')
                    return roc_auc_score(y, grid.predict_proba(X)[:, 1])
            except AttributeError:
                # If predict_proba is not available, attempt to use decision_function for binary classification
                if debug:
                    print('ATTRIBUTE ERROR')
                if class_type != 'multi' and hasattr(grid, 'decision_function'):
                    if debug:
                        print(f'Class: {class_type}, Method: decision_function(X), Threshold: {threshold}, Data: {note}, Score: ROC AUC')
                    decision_values = grid.decision_function(X)
                    return roc_auc_score(y, decision_values)
                # If neither predict_proba nor decision_function are suitable, return None
                if debug:
                    print(f'Class: {class_type}, Method: NONE, Threshold: {threshold}, Data: {note}, Score: ROC AUC')
                return None

        # Calculate the train and test ROC AUC
        train_roc_auc = calculate_roc_auc(grid, X_train, y_train, class_type=class_type, note='Train')
        test_roc_auc = calculate_roc_auc(grid, X_test, y_test, class_type=class_type, note='Test')

        # Calculate train metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average=average, zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average=average)
        train_f1 = f1_score(y_train, y_train_pred, average=average)

        # Calculate test metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average=average, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average=average)
        test_f1 = f1_score(y_test, y_test_pred, average=average)

        # Append train metrics to lists
        train_accuracy_list.append(train_accuracy)
        train_precision_list.append(train_precision)
        train_recall_list.append(train_recall)
        train_f1_list.append(train_f1)
        train_roc_auc_list.append(train_roc_auc)

        # Append test metrics to lists
        test_accuracy_list.append(test_accuracy)
        test_precision_list.append(test_precision)
        test_recall_list.append(test_recall)
        test_f1_list.append(test_f1)
        test_roc_auc_list.append(test_roc_auc)

        # Get the best Grid Search CV score and append to list
        best_cv_score = grid.best_score_
        best_cv_score_list.append(best_cv_score)
        if output:
            print(f"Best CV {scorer_name} Score: {best_cv_score:.{decimal}f}")

        # Get the best Grid Search Train score adn append to list
        train_score = grid.score(X_train, y_train)
        train_score_list.append(train_score)
        if output:
            print(f"Train {scorer_name} Score: {train_score:.{decimal}f}")

        # Get the best Grid Search Test score adn append to list
        test_score = grid.score(X_test, y_test)
        test_score_list.append(test_score)
        if output:
            print(f"Test {scorer_name} Score: {test_score:.{decimal}f}")

        # Assess the degree of overfit (train score higher than test score)
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
        best_estimator = grid.best_estimator_
        best_estimator_list.append(best_estimator)
        best_params = grid.best_params_
        best_param_list.append(best_params)
        if output:
            print(f"Best Parameters: {best_params}")

        # Output the neural network layers for KerasClassifier
        if model == KerasClassifier:
            if output:
                # Access the Keras model from the best estimator in the grid search
                keras_classifier = grid.best_estimator_.named_steps['keras_class']
                keras_model = keras_classifier.model_
                keras_model.summary()

        # Display model evaluation metrics and plots by calling 'eval_model' function
        # Note: Some of this duplicates what we just calculated, room for future optimization
        if model_eval:

            # Handle binary vs. multi-class, and special case for SVC that requires svm_proba=True
            if model_name != 'SVC' or (model_name == 'SVC' and svm_proba == True):
                # Note: decimal is set to 2 and doesn't respect the decimal param, may change this
                if class_type == 'binary':
                    # Capture binary metrics for processing later, only in the binary case
                    binary_metrics = eval_model(y_test=y_test, y_pred=y_test_pred, x_test=X_test, estimator=grid, debug=debug,
                                                neg_display=neg_display, pos_display=pos_display, pos_label=pos_label,
                                                class_type=class_type, model_name=model_name, threshold=threshold,
                                                decimal=2, plot=True, figsize=(12,11), class_weight=class_weight,
                                                return_metrics=True, output=output)
                elif class_type == 'multi':
                    eval_model(y_test=y_test, y_pred=y_test_pred, x_test=X_test, estimator=grid, debug=debug,
                               neg_display=neg_display, pos_display=pos_display, pos_label=pos_label, class_type=class_type,
                               model_name=model_name, output=output,
                               decimal=2, plot=True, figmulti=figmulti, class_weight=class_weight)

            # For neural network, if plot_curves=True, re-train the model and plot training history
            if model == KerasClassifier and plot_curve:
                # First, apply the imputer
                if imputer:
                    imputed_data = best_estimator.named_steps[imputer].transform(X_train)
                else:
                    imputed_data = X_train

                # Then apply the scaler
                if scaler:
                    scaled_data = best_estimator.named_steps[scaler].transform(imputed_data)
                else:
                    scaled_data = imputed_data
                # Note: Do we need selector here?

                # Fit the best model on the processed data so we can access history
                # Note: Can we do this earlier to avoid fitting the model a second time?
                keras_classifier.fit(scaled_data, y_train)

                # Debug: Print attributes of keras_classifier
                if debug:
                    print(dir(keras_classifier))

                # Access the training history
                history = keras_classifier.model_.history.history
                if debug:
                    print("Training History: ", history)

                # Create the matplotlib figure and axes
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

                # Plot training and validation Loss values
                ax1.grid(which='both', color='lightgrey', linewidth=0.5)
                ax1.plot(history['loss'], label='Training Loss', marker='.')
                ax1.plot(history['val_loss'], label='Validation Loss', marker='.')
                ax1.set_title('Training vs. Validation Loss', fontsize=18, pad=15)
                ax1.set_xlabel('Epoch', fontsize=14, labelpad=15)
                ax1.set_ylabel('Loss', fontsize=14, labelpad=10)
                ax1.legend()

                # Plot training and validation Accuracy values
                ax2.grid(which='both', color='lightgrey', linewidth=0.5)
                ax2.plot(history['accuracy'], label='Training Accuracy', marker='.')
                ax2.plot(history['val_accuracy'], label='Validation Accuracy', marker='.')
                ax2.set_title('Training vs. Validation Accuracy', fontsize=18, pad=15)
                ax2.set_xlabel('Epoch', fontsize=14, labelpad=15)
                ax2.set_ylabel('Accuracy', fontsize=14, labelpad=10)
                ax2.legend(loc='lower right')

                # Show the plot
                plt.tight_layout()
                plt.show()

        # Set the binary metric values based on the list of binary metrics, if it was produced by 'eval_model'
        if binary_metrics is not None:
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
        if debug:
            print('Model', len(model_name_list))
            print('Test Size', len([test_size] * len(model_name_list)))
            print('Over Sample', len([over_sample] * len(model_name_list)))
            print('Under Sample', len([under_sample] * len(model_name_list)))
            print('Resample', len(resample_list))
            print('Total Fit Time', len(fit_time_list))
            print('Fit Count', len(fit_count_list))
            print('Average Fit Time', len(avg_fit_time_list))
            print('Inference Time', len(inference_time_list))
            print('Grid Scorer', len([scorer_name] * len(model_name_list)))
            print('Best Params', len(best_param_list))
            print('Best CV Score', len(best_cv_score_list))
            print('Train Score', len(train_score_list))
            print('Test Score', len(test_score_list))
            print('Overfit', len(overfit_list))
            print('Overfit Difference', len(overfit_diff_list))
            print('Train Accuracy Score', len(train_accuracy_list))
            print('Test Accuracy Score', len(test_accuracy_list))
            print('Train Precision Score', len(train_precision_list))
            print('Test Precision Score', len(test_precision_list))
            print('Train Recall Score', len(train_recall_list))
            print('Test Recall Score', len(test_recall_list))
            print('Train F1 Score', len(train_f1_list))
            print('Test F1 Score', len(test_f1_list))
            print('Train ROC AUC Score', len(train_roc_auc_list))
            print('Test ROC AUC Score', len(test_roc_auc_list))
            print('Threshold', len([threshold] * len(model_name_list)))
            print('True Positives', len(tp_list))
            print('False Positives', len(fp_list))
            print('True Negatives', len(tn_list))
            print('False Negatives', len(fn_list))
            print('TPR', len(tpr_list))
            print('TNR', len(tnr_list))
            print('FNR', len(fnr_list))
            print('False Rate', len(fr_list))
            print('Pipeline', len(pipeline_list))
            print('Notes', len([notes] * len(model_name_list)))
            print('Timestamp', len(timestamp_list))

        # Create the results DataFrame with each list as a column, with a row for model iteration in this run
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
        # Melt the results_df so we can plot the scores for each model
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
    [('ohe', ColumnTransformer(remainder='passthrough',
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
    [('knn_imputer', KNNImputer()), ('ohe_poly2_log', ColumnTransformer(remainder='passthrough',
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
        col_trans = ColumnTransformer(transformer_steps, remainder='passthrough')
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
        pos_label: Union[int, str, bool] = 1,
        pos_display: str = 'Class 1',
        neg_display: str = 'Class 0',
        class_type: str = 'binary',
        estimator: Optional[Any] = None,
        x_test: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        title: Optional[str] = None,
        model_name: str = 'Model',
        class_weight: Optional[str] = None,
        decimal: int = 4,
        plot: bool = False,
        figsize: Tuple[int, int] = (12, 11),
        figmulti: float = 1.5,
        return_metrics: bool = False,
        output: bool = True,
        debug: bool = False
) -> Optional[Dict[str, Union[int, float]]]:
    """
    Produce a detailed evaluation report for a classification model.

    This function provides a comprehensive evaluation of a binary or multi-class
    classification model based on `y_test` (the actual target values) and `y_pred`
    (the predicted target values). It displays a text-based classification report
    enhanced with True/False Positives/Negatives (if binary), and 4 charts if
    `plot` is True: Confusion Matrix, Histogram of Predicted Probabilities, ROC
    Curve, and Precision-Recall Curve.

    If `class_type` is 'binary' (default), it will treat this as a binary
    classification. If `class_type` is 'multi', it will treat this as a multi-class
    problem. To plot the curves or adjust the `threshold` (default 0.5), both
    `X_test` and `estimator` must be provided. These are required for two Scikit
    functions called from within: `roc_auc_score` and `predict_proba`.

    A number of classification metrics are shown in the report: Accuracy,
    Precision, Recall, F1, and ROC AUC. In addition, for binary classification,
    True Positive Rate, False Positive Rate, True Negative Rate, and False
    Negative Rate are shown.

    The `pos_label` (default is 1) can be adjusted to whatever value should be
    treated as the positive class in the binary classification scenario. This
    should match one of the values in `y_test` and `preds`. For readability,
    class display names can be set with `pos_display` and `neg_display`.

    You can customize the `title` of the report completely, or pass the
    `model_name` and it will be displayed in a dynamically generated title. You
    can also specify the number of `decimal` places to show, and size of the
    figure (`fig_size`). For multi-class, you can set a `figmulti` scaling factor
    for the plot.

    You can set the `class_weight` as a display only string that is not used in
    any functions within `eval_model`. This is useful if you trained the model
    with a 'balanced' class_weight, and now want to pass that to this report to
    see the effects.

    A dictionary of the metrics can be returned if `return_metrics` is True, and
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
        True labels for the test set.
    y_pred : np.ndarray
        Predicted labels for the test set.
    pos_label : Union[int, str, bool], optional
        Data value corresponding to the positive class label. Default is 1.
    pos_display : str, optional
        Name of the positive class label. Default is 'Class 1'.
    neg_display : str, optional
        Name of the negative class label. Default is 'Class 0'.
    class_type : str, optional
        Type of classification. Default is 'binary'. Other option is 'multi'.
    estimator : estimator object, optional
        The trained model estimator. Required for plotting curves or using a
        custom threshold. Default is None.
    x_test : np.ndarray, optional
        Feature data for the test set. Required for plotting curves or using a
        custom threshold. Default is None.
    threshold : float, optional
        Threshold for classification. Default is 0.5.
    title : str, optional
        Title for the classification report. Default is None.
    model_name : str
        Included in a dynamic title and some plots. Default is 'Model'.
    class_weight : str, optional
        Class weight parameter if passed in from a parent function. For display
        purposes only on the report. For this to have an effect, it has to be
        used in model training. Default is None.
    decimal : int, optional
        Number of decimal places for metrics in the report. Default is 4.
    plot : bool, optional
        Whether to plot the evaluation graphs. Default is False.
    figsize : Tuple[int, int], optional
        Size of the plots. Default is (12, 11).
    figmulti : float, optional
        Multiplier for adjusting the size of the plots. Default is 1.5.
    return_metrics : bool, optional
        Whether to return the evaluation metrics as a dictionary. Default is False.
    output : bool, optional
        Whether to display the evaluation report and plots. Default is True.
    debug : bool, optional
        Flag to show debugging information.

    Returns
    -------
    metrics : Dict[str, Union[int, float]], optional
        A dictionary containing the evaluation metrics. Returned only if
        `return_metrics` is True.

    Examples
    --------
    Prepare data and model for the examples:

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import SVC
    >>> X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    ...                                                     random_state=42)
    >>> model = SVC(kernel='linear', probability=True, random_state=42)
    >>> model.fit(X_train, y_train)
    SVC(kernel='linear', probability=True, random_state=42)
    >>> y_pred = model.predict(X_test)

    Example 1: Basic evaluation with default parameters, no plots:

    >>> eval_model(y_test=y_test, y_pred=y_pred, model_name='SVC',
    ...            pos_label=1, pos_display='Class 1',
    ...            neg_display='Class 0')  #doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    SVC Binary Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
         Class 0     0.8252    0.9140    0.8673        93
         Class 1     0.9175    0.8318    0.8725       107
    <BLANKLINE>
        accuracy                         0.8700       200
       macro avg     0.8714    0.8729    0.8699       200
    weighted avg     0.8746    0.8700    0.8701       200
    <BLANKLINE>
                   Predicted:Class 0   Class 1
    Actual: Class 0          85        8
    Actual: Class 1          18        89
    <BLANKLINE>
    True Positive Rate / Sensitivity: 0.8318
    True Negative Rate / Specificity: 0.914
    False Positive Rate / Fall-out: 0.086
    False Negative Rate / Miss Rate: 0.1682
    <BLANKLINE>
    Positive Class: Class 1 (1)
    Threshold: 0.5

    Example 2: Evaluation with custom threshold, plotting, and return the metrics:

    >>> svc_metrics = eval_model(y_test=y_test, y_pred=y_pred, model_name='SVC',
    ...            pos_label=1, pos_display='Class 1', neg_display='Class 0',
    ...            estimator=model, x_test=X_test, threshold=0.7, plot=True,
    ...            return_metrics=True)  #doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    SVC Binary Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
         Class 0     0.7154    0.9462    0.8148        93
         Class 1     0.9351    0.6729    0.7826       107
    <BLANKLINE>
        accuracy                         0.8000       200
       macro avg     0.8253    0.8096    0.7987       200
    weighted avg     0.8329    0.8000    0.7976       200
    <BLANKLINE>
    ROC AUC: 0.9231
    <BLANKLINE>
                   Predicted:Class 0   Class 1
    Actual: Class 0          88        5
    Actual: Class 1          35        72
    <BLANKLINE>
    True Positive Rate / Sensitivity: 0.6729
    True Negative Rate / Specificity: 0.9462
    False Positive Rate / Fall-out: 0.0538
    False Negative Rate / Miss Rate: 0.3271
    <BLANKLINE>
    Positive Class: Class 1 (1)
    Threshold: 0.7
    >>> metrics_df = pd.DataFrame.from_dict(svc_metrics, orient='index', columns=['Value'])
    >>> print(metrics_df)
                       Value
    True Positives   72.0000
    False Positives   5.0000
    True Negatives   88.0000
    False Negatives  35.0000
    TPR               0.6729
    TNR               0.9462
    FPR               0.0538
    FNR               0.3271
    """
    # Check if plotting or custom threshold is requested
    if plot or threshold != 0.5:
        # Raise an error if estimator or x_test is not provided for custom threshold or plotting
        if estimator is None or x_test is None:
            raise ValueError("Both estimator and x_test must be provided for custom threshold or plotting curves.")

        # Get predicted probabilities based on the class type
        if class_type == 'binary':
            probabilities = estimator.predict_proba(x_test)[:, 1]  # Get probabilities for the positive class
        elif class_type == 'multi':
            probabilities = estimator.predict_proba(x_test)  # Get probabilities for all classes

        # Apply custom threshold for binary classification
        if class_type == 'binary' and threshold != 0.5:
            # Handle different shapes of the probabilities array
            if len(probabilities.shape) == 2 and probabilities.shape[1] > 1:
                y_pred = np.array([1 if prob >= threshold else 0 for prob in probabilities[:, 1]])
            elif len(probabilities.shape) == 1 or probabilities.shape[1] == 1:
                y_pred = np.array([1 if prob >= threshold else 0 for prob in probabilities])
            else:
                raise ValueError("Unexpected shape for probabilities array")

    # Evaluation for multi-class classification
    if class_type == 'multi':
        unique_labels = np.unique(y_test)
        num_classes = len(unique_labels)
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, estimator.predict_proba(x_test), multi_class='ovr')
        if output:
            # Display the best title we can create
            if title is not None:
                print(f"\n{title}\n")
            elif model_name != 'Model':
                print(f"\n{model_name} Multi-Class Classification Report\n")
            else:
                print(f"\nMulti-Class Classification Report\n")
            print(classification_report(y_test, y_pred, digits=decimal, target_names=[str(label) for label in unique_labels]))
            print("ROC AUC:", round(roc_auc, decimal))
            if class_weight is not None:
                print("Class Weight:", class_weight)

    # Evaluation for binary classification
    elif class_type == 'binary':
        cm = confusion_matrix(y_test, y_pred)
        if x_test is not None:
            roc_auc = roc_auc_score(y_test, estimator.predict_proba(x_test)[:, 1])

        # Calculate evaluation metrics
        TN, FP, FN, TP = cm.ravel()
        FPR = FP / (FP + TN)
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        FNR = FN / (FN + TP)

        if output:
            # Display the best title we can create
            if title is not None:
                print(f"\n{title}\n")
            elif model_name != 'Model':
                print(f"\n{model_name} Binary Classification Report\n")
            else:
                print(f"\nBinary Classification Report\n")

            # Run the Classification report
            print(classification_report(y_test, y_pred, digits=decimal, target_names=[str(neg_display), str(pos_display)]))
            if x_test is not None:
                print("ROC AUC:", round(roc_auc, decimal), "\n")

            # Print confusion matrix
            print(f"{'':<15}{'Predicted:':<10}{neg_display:<10}{pos_display:<10}")
            print(f"{'Actual: ' + str(neg_display):<25}{cm[0][0]:<10}{cm[0][1]:<10}")
            print(f"{'Actual: ' + str(pos_display):<25}{cm[1][0]:<10}{cm[1][1]:<10}")

            # Print evaluation metrics
            print("\nTrue Positive Rate / Sensitivity:", round(TPR, decimal))
            print("True Negative Rate / Specificity:", round(TNR, decimal))
            print("False Positive Rate / Fall-out:", round(FPR, decimal))
            print("False Negative Rate / Miss Rate:", round(FNR, decimal))
            print(f"\nPositive Class: {pos_display} ({pos_label})")
            if class_weight is not None:
                print("Class Weight:", class_weight)
            print("Threshold:", threshold)


        # Store evaluation metrics in a dictionary
        metrics = {
            "True Positives": TP,
            "False Positives": FP,
            "True Negatives": TN,
            "False Negatives": FN,
            "TPR": round(TPR, decimal),
            "TNR": round(TNR, decimal),
            "FPR": round(FPR, decimal),
            "FNR": round(FNR, decimal),
        }

    # Plot the charts if requested
    if plot and output:
        blue = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)  # Define a blue color for plots

        if class_type == 'multi':
            # Calculate the figure size for multi-class plots
            multiplier = figmulti
            max_size = 20
            size = min(num_classes * multiplier, max_size)
            figsize = (size, size)

            # Create a figure and axis for multi-class confusion matrix
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

            # Plot the confusion matrix
            cm_display = ConfusionMatrixDisplay(cm, display_labels=unique_labels)
            cm_display.plot(cmap='Blues', ax=ax1, colorbar=False)
            for text in cm_display.text_:
                for t in text:
                    t.set_fontsize(10)
            ax1.set_title(f'Confusion Matrix', fontsize=18, pad=15)
            ax1.set_xlabel('Predicted Label', fontsize=14, labelpad=15)
            ax1.set_ylabel('True Label', fontsize=14, labelpad=10)
            ax1.tick_params(axis='both', which='major', labelsize=10)

            plt.show()

        elif class_type == 'binary':
            # Create a figure and subplots for binary classification plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

            # Plot the confusion matrix
            cm_matrix = ConfusionMatrixDisplay(cm, display_labels=[neg_display, pos_display])
            cm_matrix.plot(cmap='Blues', ax=ax1, colorbar=False)
            for text in cm_matrix.text_:
                for t in text:
                    t.set_fontsize(14)
                    t.set_text(f"{int(float(t.get_text()))}")
            ax1.set_title(f'Confusion Matrix', fontsize=18, pad=15)
            ax1.set_xlabel('Predicted Label', fontsize=14, labelpad=15)
            ax1.set_ylabel('True Label', fontsize=14, labelpad=10)
            ax1.tick_params(axis='both', which='major', labelsize=11)

            # Plot the histogram of predicted probabilities
            ax2.hist(probabilities, color=blue, edgecolor='black', alpha=0.7, label=f'{model_name} Probabilities')
            ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold: {threshold:.{decimal}f}')
            ax2.set_title('Histogram of Predicted Probabilities', fontsize=18, pad=15)
            ax2.set_xlabel('Probability', fontsize=14, labelpad=15)
            ax2.set_ylabel('Frequency', fontsize=14, labelpad=10)
            ax2.set_xticks(np.arange(0, 1.1, 0.1))
            ax2.legend()

            # Calculate false positive rate, true positive rate, and thresholds for ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, probabilities, pos_label=pos_label)

            # Plot the ROC curve
            ax3.plot([0, 1], [0, 1], color='grey', linestyle=':', label='Chance Baseline')
            RocCurveDisplay.from_estimator(estimator, x_test, y_test, pos_label=pos_label,
                                           marker='.', ax=ax3, linewidth=2, label=f'{model_name} Curve',
                                           color=blue, response_method='predict_proba')

            # Plot the threshold point on the ROC curve
            if threshold is not None:
                closest_idx = np.argmin(np.abs(thresholds - threshold))
                fpr_point = fpr[closest_idx]
                tpr_point = tpr[closest_idx]

                ax3.scatter(fpr_point, tpr_point, color='red', s=80, zorder=5, label=f'Threshold {threshold:.{decimal}f}')

                ax3.axvline(x=fpr_point, ymax=tpr_point-0.025, color='red', linestyle='--', lw=1,
                            label=f'TPR: {tpr_point:.{decimal}f}, FPR: {fpr_point:.{decimal}f}')
                ax3.axhline(y=tpr_point, xmax=fpr_point+0.04, color='red', linestyle='--', lw=1)

            ax3.set_xticks(np.arange(0, 1.1, 0.1))
            ax3.set_yticks(np.arange(0, 1.1, 0.1))
            ax3.set_ylim(0,1.05)
            ax3.set_xlim(-0.05,1.0)
            ax3.grid(which='both', color='lightgrey', linewidth=0.5)
            ax3.set_title('ROC Curve', fontsize=18, pad=15)
            ax3.set_xlabel('False Positive Rate', fontsize=14, labelpad=15)
            ax3.set_ylabel('True Positive Rate', fontsize=14, labelpad=10)
            ax3.legend(loc='lower right')

            # Calculate precision, recall, and thresholds for Precision-Recall curve
            precision, recall, thresholds = precision_recall_curve(y_test, probabilities, pos_label=pos_label)

            # Plot the Precision-Recall curve
            ax4.plot(recall, precision, marker='.', label=f'{model_name} Curve', color=blue)
            if threshold is not None:
                chosen_threshold = threshold
                closest_point = np.argmin(np.abs(thresholds - chosen_threshold))
                ax4.scatter(recall[closest_point], precision[closest_point], color='red', s=80, zorder=5,
                            label=f'Threshold: {chosen_threshold:.{decimal}f}')
                ax4.axvline(x=recall[closest_point], ymax=precision[closest_point]-0.025, color='red',
                            linestyle='--', lw=1, label=f'Precision: {precision[closest_point]:.{decimal}f},'
                                                        f'Recall: {recall[closest_point]:.{decimal}f}')
                ax4.axhline(y=precision[closest_point], xmax=recall[closest_point]-0.025, color='red',
                            linestyle='--', lw=1)
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

    # Return a dictionary of metrics if requested
    if return_metrics:
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
        show_time: bool = True,
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
    show_time : bool, optional
        Show the timestamp, disable for test cases (default True).
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
    ...                       model='linreg', show_time=False)
    <BLANKLINE>
    ITERATION 1 RESULTS
    <BLANKLINE>
    Pipeline: linreg
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
    ...     save=True, save_df=results_df, show_time=False)
    <BLANKLINE>
    ITERATION 2 RESULTS
    <BLANKLINE>
    Pipeline: poly2 -> stand -> ridge
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
    if show_time:
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
        decimal: int = 2,
        return_df: bool = False,
        x_column: str = 'Iteration',
        y_label: str = None,
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
    sns.lineplot(data=df_long, x=x_column, y='Value', hue='Metric')

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

        # Plot the vertical dotted line and dot
        plt.axvline(x=best_iter, color='green', linestyle='--', zorder=2,
                    label=f"{x_column} {best_iter}: {select_metric}: {best_val_formatted}")
        plt.scatter(best_iter, y_coord, color='green', s=60, zorder=3)

    # Continue the plot
    plt.grid(linestyle='--', linewidth=0.5, color='#DDDDDD')
    plt.legend(loc='best')

    # Format the X axis
    plt.xticks(df[x_column].unique())
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