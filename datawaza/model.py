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
    - :func:`~datawaza.model.create_pipeline` - Create a custom pipeline for data preprocessing and modeling.
    - :func:`~datawaza.model.create_results_df` - Initialize the results_df DataFrame with the columns required for `iterate_model`.
    - :func:`~datawaza.model.iterate_model` - Iterate and evaluate a model pipeline with specified parameters.
    - :func:`~datawaza.model.plot_results` - Plot the results of model iterations and select the best metric.
"""

# Metadata
__author__ = "Jim Beno"
__email__ = "jim@jimbeno.net"
__version__ = "0.1.0"
__license__ = "GNU GPLv3"

# Standard library imports
import os
from datetime import datetime

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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   PolynomialFeatures, RobustScaler, StandardScaler)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Other third-party libraries for machine learning
from category_encoders import JamesSteinEncoder

# Local Datawaza helper function imports
from datawaza.tools import calc_pfi, calc_vif, extract_coef, log_transform, thousands

# Miscellaneous imports
from joblib import dump


# Functions
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
            raise ValueError("If no config is provided, X_cat_columns and "
                             "X_num_columns must be passed.")
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
                'js': (JamesSteinEncoder(), cat_columns),
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
        decimal: int = 2,
        lowess: bool = False,
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

    `create_pipeline` is called to create a pipeline from the specified parameters:

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
    decimal : int, optional
        Number of decimal places for displaying metrics (default 2).
    show_time : bool, optional
        Show the timestamp, disable for test cases (default True).
    lowess : bool, optional
        Flag to display lowess curve in residual plots (default False).
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
    Cross Validation:
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
            print('Xn_train.columns', x_train.columns)
            print('Xn_test.columns', x_test.columns)

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
    current_time = datetime.now(pytz.timezone('US/Pacific'))
    timestamp = current_time.strftime('%b %d, %Y %I:%M %p PST')
    if show_time:
        print(f'{timestamp}\n')

    if cross or grid:
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

        grid = GridSearchCV(pipe, param_grid=config['params'][grid_params], cv=config['cv'][grid_cv], scoring=grid_score, verbose=grid_verbose)
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
        best_grid_params = None
        best_grid_score = None
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

    if save:
        if save_df is not None:
            results_df = save_df
        else:
            # Create results_df if it doesn't exist
            results_df = pd.DataFrame(columns=['Iteration', 'Train MSE', 'Test MSE', 'Train RMSE', 'Test RMSE',
                                               'Train MAE', 'Test MAE', 'Train R^2 Score', 'Test R^2 Score', 'Best Grid Mean Score',
                                               'Best Grid Params', 'Pipeline', 'Note', 'Date'])

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

        # Convert the dictionary to a dataframe
        df_iteration = pd.DataFrame([results])

        # Append the results dataframe to the existing results dataframe
        results_df = pd.concat([results_df if not results_df.empty else None,
                                df_iteration], ignore_index=True)

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
            print(f"\FAILED to save the model as {filename}")

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