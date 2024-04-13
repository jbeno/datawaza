# tools.py – Tools module of Datawaza
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
This module provides helper tools used in data analysis, cleaning, and modeling.
It contains functions to check for duplicates in lists, split a dataframe into two
by numeric vs. categorical variables, format numbers on the axis of a chart,
perform log transformations, calculate VIF and Feature Permutation Importance,
and extract the coefficients from models that support them.

Classes:
    - :class:`~datawaza.tools.DebugPrinter` - Conditionally print debugging information during the execution of a script.
        - :meth:`~datawaza.tools.DebugPrinter.print` - Print a message if debugging is enabled.
        - :meth:`~datawaza.tools.DebugPrinter.set_debug` - Enable or disable debugging mode.
    - :class:`~datawaza.tools.LogTransformer` - Apply logarithmic transformation to numerical features.
        - :meth:`~datawaza.tools.LogTransformer.fit` - Fit the transformer to the input data.
        - :meth:`~datawaza.tools.LogTransformer.transform` - Apply the logarithmic transformation to the input data.
        - :meth:`~datawaza.tools.LogTransformer.get_feature_names_out` - Get the feature names after applying the transformation.

Functions:
    - :func:`~datawaza.tools.calc_pfi` - Calculate Permutation Feature Importance for a trained model.
    - :func:`~datawaza.tools.calc_vif` - Calculate the Variance Inflation Factor (VIF) for each feature.
    - :func:`~datawaza.tools.check_for_duplicates` - Check for duplicate items (ex: column names) across multiple lists.
    - :func:`~datawaza.tools.extract_coef` - Extract feature names and coefficients from a trained model.
    - :func:`~datawaza.tools.format_df` - Format columns of a DataFrame as either large or small numbers.
    - :func:`~datawaza.tools.log_transform` - Apply a log transformation to specified columns in a DataFrame.
    - :func:`~datawaza.tools.model_summary` - Create a DataFrame summary of a Keras model's architecture and parameters.
    - :func:`~datawaza.tools.split_dataframe` - Split a DataFrame into categorical and numerical columns.
    - :func:`~datawaza.tools.thousand_dollars` - Format a number as currency with thousands separators on a matplotlib chart axis.
    - :func:`~datawaza.tools.thousands` - Format a number with thousands separators on a matplotlib chart axis.
"""

# Metadata
__author__ = "Jim Beno"
__email__ = "jim@jimbeno.net"
__version__ = "0.1.3"
__license__ = "GNU GPLv3"

# Standard library imports
import os
import inspect

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning: Model selection and evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

# Machine Learning: Pipeline and transformations
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Machine Learning: Models
from sklearn.linear_model import (
    LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier,
    Perceptron, RidgeClassifier, RidgeClassifierCV, SGDClassifier, SGDOneClassSVM,
    LinearRegression, Ridge, RidgeCV, SGDRegressor, ElasticNet, ElasticNetCV,
    Lars, LarsCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC,
    OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, ARDRegression,
    BayesianRidge, HuberRegressor, QuantileRegressor, RANSACRegressor,
    TheilSenRegressor
)

# Typing imports
from typing import Optional, Union, Tuple, List, Dict, Any

# TensorFlow and Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warning on import
import tensorflow as tf
import keras as keras


# Functions
def calc_pfi(model,
             X: pd.DataFrame,
             y: pd.Series,
             scoring: Any = None,
             n_repeats: int = 10,
             random_state: int = 42,
             decimal: int = 2
             ) -> pd.DataFrame:
    """
    Calculate Permutation Feature Importance for a trained model.

    This function calculates the Permutation Feature Importance (PFI) for
    each feature in the input dataset using a trained model. PFI measures
    the importance of each feature by permuting its values and observing
    the impact on the model's performance. Features with higher
    permutation importance scores are considered more important for the
    model's predictions.

    The function returns a DataFrame with the feature names, mean
    permutation importance scores, and standard deviations of the scores.
    The DataFrame is sorted in descending order based on the mean scores.
    It's just a wrapper around the Scikit-learn `permutation_importance`
    function to display the results in a convenient format.

    Use this function to identify the most important features for a
    trained model and gain insights into the model's behavior.

    Parameters
    ----------
    model :
        The trained model object. It should have a `predict` method.
    X : pd.DataFrame
        The input DataFrame containing the features used for prediction.
    y : pd.Series
        The target variable or labels corresponding to the input features.
    scoring : Any, optional
        Scorer to use. It can be a single string (see sklearn 'scoring_parameter') or
        a callable that returns a single value. Default is None, which uses the
        estimator's default scorer.
    n_repeats : int, optional
        The number of times to permute each feature. Higher values provide
        more stable importance scores but increase computation time.
        Default is 10.
    random_state : int, optional
        The random seed for reproducibility. Default is 42.
    decimal : int, optional
        The number of decimals to round to when displaying output.
        Default is 2.

    Returns
    -------
    pd.DataFrame
        A DataFrame with three columns: 'Feature' (feature names),
        'Importance Mean' (mean permutation importance scores), and
        'Importance Std' (standard deviations of the scores). The DataFrame is
        sorted in descending order based on the 'Importance Mean' column.

    Examples
    --------
    Prepare a sample dataset and train a model:

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> iris = load_iris()
    >>> X = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> y = pd.Series(iris.target)
    >>> model = RandomForestClassifier(random_state=42)
    >>> model.fit(X, y)
    RandomForestClassifier(random_state=42)

    Calculate Permutation Feature Importance:

    >>> pfi_df = calc_pfi(model, X, y, decimal=4)
    >>> pfi_df
                 Feature Importance Mean Importance Std
    2  petal length (cm)          0.2227         0.0243
    3   petal width (cm)          0.1807         0.0212
    0  sepal length (cm)          0.0147         0.0065
    1   sepal width (cm)          0.0127         0.0047
    """
    # Calculate Permutation Feature Importance
    r = permutation_importance(model, X, y, n_repeats=n_repeats, scoring=scoring,
                               random_state=random_state)

    # Create a DataFrame with feature names, mean scores, and std scores
    pfi_df = pd.DataFrame({"Feature": X.columns,
                           "Importance Mean": r.importances_mean,
                           "Importance Std": r.importances_std})

    # Sort the DataFrame by mean scores in descending order
    pfi_sorted = pfi_df.sort_values(by="Importance Mean", ascending=False)

    # Format the PFI values for better readability
    pfi_formatted = format_df(pfi_sorted, small_num_cols=['Importance Mean', 'Importance Std'], decimal=decimal)

    return pfi_formatted


def calc_vif(X: pd.DataFrame,
             num_columns: Optional[List[str]] = None,
             decimal: int = 2
             ) -> pd.DataFrame:
    """
    Calculate the Variance Inflation Factor (VIF) for each feature.

    This function calculates the VIF for each feature in the input
    dataset. VIF is a measure of multicollinearity, which indicates the
    degree to which a feature can be explained by other features in the
    dataset. A higher VIF value suggests higher multicollinearity, and a
    VIF value exceeding 5 or 10 is often regarded as indicating severe
    multicollinearity.

    By default, VIF will be calculated for all numeric columns in the `X`
    DataFrame. You can optionally specify columns with `num_columns`. You
    can also control how many decimal places are shown with `decimal`.

    The function also interprets the level of multicollinearity based on
    the VIF values and assigns a corresponding category: "Extreme" (VIF
    >= 100), "High" (10 <= VIF < 100), "Moderate" (5 <= VIF < 10), or
    "Low" (VIF < 5).

    Use this function to identify features with high multicollinearity in
    your dataset before performing further analysis or modeling.

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame containing the features to calculate VIF for.
    num_columns : List[str], optional
        List of column names to consider for VIF calculation. If
        provided, only the specified numeric columns will be used. If
        None (default), all numeric columns in the DataFrame will be
        used.
    decimal : int, optional
        The number of decimals to round to when displaying output.
        Default is 2.

    Returns
    -------
    pd.DataFrame
        A DataFrame with three columns: 'Features' (feature names), 'VIF'
        (VIF values), and 'Multicollinearity' (interpreted level of
        multicollinearity). The DataFrame is sorted in descending order
        based on the VIF values.

    Examples
    --------
    Prepare a sample dataset for the examples:

    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> X = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> num_columns = list(X.columns)

    Example 1: Calculate VIF for all numeric features in the iris dataset:

    >>> vif_df = calc_vif(X)
    >>> vif_df
                Features    VIF Multicollinearity
    2  petal length (cm)  31.26              High
    3   petal width (cm)  16.09              High
    0  sepal length (cm)   7.07          Moderate
    1   sepal width (cm)   2.10               Low

    Example 2: Calculate VIF for specific numeric features, 4 decimals:

    >>> vif_df = calc_vif(X, num_columns=num_columns, decimal=4)
    >>> vif_df
                Features      VIF Multicollinearity
    2  petal length (cm)  31.2615              High
    3   petal width (cm)  16.0902              High
    0  sepal length (cm)   7.0727          Moderate
    1   sepal width (cm)   2.1009               Low
    """
    from sklearn.linear_model import LinearRegression

    def interpret_vif(vif):
        if vif >= 100:
            return "Extreme"
        elif vif >= 10:
            return "High"
        elif vif >= 5:
            return "Moderate"
        else:
            return "Low"

    # Set a high threshold for very large VIFs
    MAX_VIF = 1000

    # If num_columns is not provided, select all numeric columns
    if num_columns is None:
        num_columns = X.select_dtypes(include=[np.number]).columns

    vif_dict = {}

    for feature in num_columns:
        other_features = [col for col in num_columns if col != feature]

        # Split the dataset, one independent variable against all others
        X_other, y = X[other_features], X[feature]

        # Fit the model and obtain R^2
        r_squared = LinearRegression().fit(X_other, y).score(X_other, y)

        # Compute the VIF, with a check for r_squared close to 1
        if 1 - r_squared < 1e-5:
            vif = MAX_VIF
        else:
            vif = 1 / (1 - r_squared)

        vif_dict[feature] = vif

    # Create a DataFrame with VIF values
    vif_df = pd.DataFrame({"Features": vif_dict.keys(), "VIF": vif_dict.values()})

    # Flag severe multicollinearity
    vif_df["Multicollinearity"] = vif_df["VIF"].apply(interpret_vif)

    # Sort the DataFrame by VIF values in descending order
    vif_sorted = vif_df.sort_values(by='VIF', ascending=False)

    # Format the VIF values for better readability
    vif_formatted = format_df(vif_sorted, small_num_cols=['VIF'], decimal=decimal)

    return vif_formatted

def check_for_duplicates(*lists: List[str],
                         df: Optional[pd.DataFrame] = None) -> None:
    """
    Check for duplicate items (ex: column names) across multiple lists.

    This function takes an arbitrary number of lists and checks for duplicate items
    across the lists, as well as items appearing more than once within each list.
    It prints a summary of the items and the lists they appear in. Additionally, if
    a DataFrame is provided, it checks for any columns in the DataFrame that are
    missing from the lists and prints them.

    Use this function when you are organizing columns in a large DataFrame into
    lists that represent their variable type (ex: num_columns, cat_columns). This
    helps to ensure you haven't duplicated a column accidentally. And the optional
    DataFrame check helps you identify columns that haven't been assigned to a list
    yet. This is really useful when you're dealing with a large dataset.

    Parameters
    ----------
    *lists : List[str]
        An arbitrary number of lists containing items (ex: column names) to check
        for duplicates.
    df : pd.DataFrame, optional
        A DataFrame to check for missing columns that are not present in the lists.
        Default is None.

    Returns
    -------
    None
        The function prints the duplicate items, the lists they appear in, and any
        missing columns in the DataFrame (if provided).

    Examples
    --------
    Prepare data for examples, with intentional duplicates:

    >>> df = pd.DataFrame({'age': [], 'height': [], 'weight': [], 'gender': [],
    ... 'city': [], 'country': []})
    >>> num_cols = ['age', 'height', 'weight']
    >>> cat_cols = ['gender', 'age', 'country', 'country']

    Example 1: Check for duplicate column names in two lists:

    >>> check_for_duplicates(num_cols, cat_cols)
    Items appearing in more than one list, or more than once per list:
    age (2): num_cols, cat_cols
    country (2): cat_cols, cat_cols

    Fix the duplicate column:

    >>> cat_cols = ['gender', 'country']

    Example 2: Check for duplicates, and look for missing columns in a DataFrame:

    >>> check_for_duplicates(num_cols, cat_cols, df=df)
    Items appearing in more than one list, or more than once per list:
    None.
    <BLANKLINE>
    Columns in the dataframe missing from the lists:
    city

    Fix the missing column:

    >>> cat_cols = ['gender', 'city', 'country']

    Final check:

    >>> check_for_duplicates(num_cols, cat_cols, df=df)
    Items appearing in more than one list, or more than once per list:
    None.
    <BLANKLINE>
    Columns in the dataframe missing from the lists:
    None.
    """
    # Get the frame and local variables of the caller
    caller_frame = inspect.currentframe().f_back
    caller_locals = caller_frame.f_locals

    # Create a dictionary to store the mapping of columns to the lists they appear in
    column_lists_map = {}

    # Iterate over each list passed as an argument
    for lst in lists:
        # Get the name of the list variable from the caller's local variables
        list_name = [name for name, value in caller_locals.items() if value is lst][0]
        # Iterate over each column in the current list
        for column in lst:
            if column not in column_lists_map:
                # If the column is not in the map, add it with the current list name
                column_lists_map[column] = [list_name]
            else:
                # Append the current list name, even if it exists, to check
                # for duplicated items or column names within the same list
                column_lists_map[column].append(list_name)

    # Create a dictionary of duplicate columns and the lists they appear in
    duplicates = {column: lists for column, lists in column_lists_map.items() if len(lists) > 1}

    # Print the summary of duplicate columns
    print("Items appearing in more than one list, or more than once per list:")
    if duplicates:
        for column, lists in duplicates.items():
            print(f"{column} ({len(lists)}): {', '.join(lists)}")
    else:
        print("None.")

    # If a DataFrame is passed, check for column names that are missing from the lists
    if df is not None:
        all_columns = column_lists_map.keys()
        missing_columns = set(df.columns) - set(all_columns)
        print("\nColumns in the dataframe missing from the lists:")
        if missing_columns:
            for column in missing_columns:
                print(column)
        else:
            print("None.")


def extract_coef(
        grid_or_pipe: Union[GridSearchCV, Pipeline],
        X: pd.DataFrame,
        format: bool = True,
        decimal: int = 2,
        debug: bool = False
) -> pd.DataFrame:
    """
    Extract feature names and coefficients from a trained model.

    This function traverses through the steps of a GridSearchCV or
    Pipeline object and extracts the feature names and coefficients from
    the final trained model. It attempts to handle transformations such as
    ColumnTransformer and feature scaling steps. However, due to the complexity
    of some transformations, and inconsistent support on tracking feature
    names, the final output feature names may be different than the input.

    Note: This function currently supports only single target regression
    problems. It also checks against a list of known classes that support
    coefficient extraction. This list may not be comprehensive.

    Parameters
    ----------
    grid_or_pipe : Union[GridSearchCV, Pipeline]
        A trained GridSearchCV or Pipeline object containing the model.
    X : pd.DataFrame
        The input DataFrame used during training, to get the original
        feature names.
    format : bool, optional
        Applies formatting to the results to make it easier to read, but
        converts numbers to strings. Default is True.
    decimal : int, optional
        The number of decimals to round to when displaying output.
        Default is 2.
    debug : bool, optional
        If True, print debugging information during the extraction
        process. Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns: 'Feature' (the names of the
        selected features) and 'Coefficient' (the corresponding
        coefficients of the features).

    Example
    -------
    Prepare sample data for the example:

    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.preprocessing import StandardScaler
    >>> X, y = fetch_california_housing(return_X_y=True)
    >>> X = pd.DataFrame(X, columns=['MedInc', 'HouseAge', 'AveRooms',
    ...                               'AveBedrms', 'Population', 'AveOccup',
    ...                               'Latitude', 'Longitude'])

    Create and fit a model pipeline:

    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('model', Ridge())
    ... ])
    >>> pipe.fit(X, y)
    Pipeline(steps=[('scaler', StandardScaler()), ('model', Ridge())])

    Example 1: Extract feature names and coefficients from the fitted model:

    >>> extract_coef(pipe, X, decimal=4)
          Feature Coefficient
    0      MedInc      0.8296
    1    HouseAge      0.1188
    2    AveRooms     -0.2654
    3   AveBedrms      0.3055
    4  Population     -0.0045
    5    AveOccup     -0.0393
    6    Latitude     -0.8993
    7   Longitude     -0.8699

    Example 2: Extract feature names adn coefficients without formatting:

    >>> extract_coef(pipe, X, format=False)
          Feature Coefficient
    0      MedInc    0.829593
    1    HouseAge    0.118817
    2    AveRooms   -0.265397
    3   AveBedrms    0.305525
    4  Population    -0.00448
    5    AveOccup    -0.03933
    6    Latitude   -0.899266
    7   Longitude   -0.869916

    Example 3: Extract coefficients from a grid search object:

    >>> parameters = {'model__alpha': [1.0, 0.5]}
    >>> grid = GridSearchCV(pipe, parameters)
    >>> grid.fit(X, y)  #doctest: +NORMALIZE_WHITESPACE
    GridSearchCV(estimator=Pipeline(steps=[('scaler', StandardScaler()),
                                           ('model', Ridge())]),
                 param_grid={'model__alpha': [1.0, 0.5]})
    >>> extract_coef(grid, X)
          Feature Coefficient
    0      MedInc        0.83
    1    HouseAge        0.12
    2    AveRooms       -0.27
    3   AveBedrms        0.31
    4  Population       -0.00
    5    AveOccup       -0.04
    6    Latitude       -0.90
    7   Longitude       -0.87
    """
    # List of classes that support the .coef_ attribute
    SUPPORTED_COEF_CLASSES = (
        LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier,
        Perceptron, RidgeClassifier, RidgeClassifierCV, SGDClassifier, SGDOneClassSVM,
        LinearRegression, Ridge, RidgeCV, SGDRegressor, ElasticNet, ElasticNetCV,
        Lars, LarsCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC,
        OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, ARDRegression,
        BayesianRidge, HuberRegressor, QuantileRegressor, RANSACRegressor,
        TheilSenRegressor
    )

    def supports_coef(estimator):
        """Check if estimator supports .coef_"""
        return isinstance(estimator, SUPPORTED_COEF_CLASSES)

    # Determine the type of the passed object and set flags
    if hasattr(grid_or_pipe, 'best_estimator_'):
        estimator = grid_or_pipe.best_estimator_
        is_grid = True
        is_pipe = False
        if debug:
            print('Grid: ', is_grid)
    else:
        estimator = grid_or_pipe
        is_pipe = True
        is_grid = False
        if debug:
            print('Pipe: ', is_pipe)

    # Initial setup
    current_features = list(X.columns)
    if debug:
        print('current_features: ', current_features)
    mapping = pd.DataFrame({
        'feature_name': current_features,
        'intermediate_name1': current_features,
        'selected': [True] * len(current_features),
        'coefficients': [None] * len(current_features)
    })

    for step_name, step_transformer in estimator.named_steps.items():
        if debug:
            print(f"Processing step: {step_name} in {step_transformer}")

        # If transformer is a ColumnTransformer
        if isinstance(step_transformer, ColumnTransformer):
            new_features = []
            for name, trans, columns in step_transformer.transformers_:
                if hasattr(trans, 'get_feature_names_out'):
                    try:
                        if hasattr(trans, 'feature_names_in_'):
                            out_features = trans.get_feature_names_out(trans.feature_names_in_)
                        else:
                            out_features = trans.get_feature_names_out(columns)
                    except ValueError:
                        out_features = [f"{name}_{i}" for i in range(trans.transform(X.iloc[:, columns].values).shape[1])]
                else:
                    out_features = columns
                new_features.extend(out_features)

            current_features = new_features
            mapping = pd.DataFrame({
                'feature_name': current_features,
                'intermediate_name1': current_features,
                'selected': [True] * len(current_features),
                'coefficients': [None] * len(current_features)
            })
            if debug:
                print("Mapping: ", mapping)

        # Reduction
        elif hasattr(step_transformer, 'get_support'):
            mask = step_transformer.get_support()
            mapping.loc[mapping['feature_name'].isin(current_features), 'selected'] = mask
            current_features = mapping[mapping['selected']]['feature_name'].tolist()


    # If there's a model with coefficients in this step, update coefficients
    if supports_coef(step_transformer):
        coefficients = step_transformer.coef_.ravel()
        selected_rows = mapping[mapping['selected']].index
        if debug:
            print("Coefficients: ", coefficients)
            print(f"Number of coefficients: {len(coefficients)}")  # Debugging
            print(f"Number of selected rows: {len(selected_rows)}")  # Debugging

        if len(coefficients) == len(selected_rows):
            mapping.loc[selected_rows, 'coefficients'] = coefficients.tolist()
        else:
            print(f"Mismatch in coefficients and selected rows for step: {step_name}")

    # For transformers inside ColumnTransformer
    if isinstance(step_transformer, ColumnTransformer):
        if debug:
            print("ColumnTransformer:", step_transformer)
        transformers = step_transformer.transformers_
        if debug:
            print("Transformers: ", transformers)
        new_features = []  # Collect new features from this step
        for name, trans, columns in transformers:
            # OneHotEncoder or similar expanding transformers
            if hasattr(trans, 'get_feature_names_out'):
                out_features = list(trans.get_feature_names_out(columns))
                new_features.extend(out_features)
                if debug:
                    print("Out features: ", out_features)
                    print("New features: ", new_features)
            else:
                new_features.extend(columns)

        current_features = new_features

        # Update mapping based on current_features
        mapping = pd.DataFrame({
            'feature_name': current_features,
            'intermediate_name1': current_features,
            'selected': [True] * len(current_features),
            'coefficients': [None] * len(current_features)
        })
        if debug:
            print("Mapping: ", mapping)
    # Filtering the final selected features and their coefficients
    final_data = mapping[mapping['selected']]

    # Rename the columns to "Feature" and "Coefficient"
    final_data = final_data[['feature_name', 'coefficients']].rename(columns={'feature_name': 'Feature', 'coefficients': 'Coefficient'})

    # Format the coefficient values for better readability
    if format:
        final_data = format_df(final_data, small_num_cols=['Coefficient'], decimal=decimal)

    return final_data


def format_df(
        df: pd.DataFrame,
        large_num_cols: Optional[List[str]] = None,
        small_num_cols: Optional[List[str]] = None,
        decimal: int = 2
) -> pd.DataFrame:
    """
    Format columns of a DataFrame as either large or small numbers.

    This function formats the specified columns in the input DataFrame.
    Large numbers are formatted with commas as thousands separators and
    no decimal places. Small numbers are formatted with a specified
    number of decimal places (they will have commas as well). Use
    `decimal` to define how many decimal places to display.

    Use this function when you need to format specific columns in a
    DataFrame for better readability or presentation purposes.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the columns to be formatted.
    large_num_cols : List[str], optional
        List of column names containing large numbers to be formatted with
        commas as thousands separators, no decimals. Default is None.
    small_num_cols : List[str], optional
        List of column names containing small numbers to be formatted with
        a specified number of decimal places. Default is None.
    decimal : int, optional
        The number of decimal places to display for small numbers. Default
        is 2.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the specified columns formatted.

    Examples
    --------
    Prepare the data for the examples:

    >>> df = pd.DataFrame({
    ...     'A': [112345697, 28799522, 391039492, 10959013409, 3522343059],
    ...     'B': [0.123401, 0.234501, 0.345601, 0.456701, 0.567801],
    ...     'C': ['X', 'Y', 'Z', 'X', 'Y']
    ... })
    >>> df
                 A         B  C
    0    112345697  0.123401  X
    1     28799522  0.234501  Y
    2    391039492  0.345601  Z
    3  10959013409  0.456701  X
    4   3522343059  0.567801  Y

    Example 1: Format large numbers and small numbers with default decimal places:

    >>> formatted_df = format_df(df, large_num_cols=['A'], small_num_cols=['B'])
    >>> formatted_df
                    A     B  C
    0     112,345,697  0.12  X
    1      28,799,522  0.23  Y
    2     391,039,492  0.35  Z
    3  10,959,013,409  0.46  X
    4   3,522,343,059  0.57  Y

    Example 2: Format small numbers with a specified number of decimal places:

    >>> formatted_df = format_df(df, large_num_cols=['A'], small_num_cols=['B'],
    ...                          decimal=4)
    >>> formatted_df
                    A       B  C
    0     112,345,697  0.1234  X
    1      28,799,522  0.2345  Y
    2     391,039,492  0.3456  Z
    3  10,959,013,409  0.4567  X
    4   3,522,343,059  0.5678  Y
    """
    # Function to format a column
    def format_columns(val, col_type):
        # Check if value is NaN or not a numeric type; return as is if true
        if pd.isna(val) or not isinstance(val, (int, float)):
            return val
        if col_type == "large":
            return '{:,.0f}'.format(val)
        elif col_type == "small":
            return f'{{:,.{decimal}f}}'.format(val)

    # Create a copy of the input DataFrame to avoid modifying the original
    formatted_df = df.copy()

    # Format columns with large numbers
    if large_num_cols:
        for col in large_num_cols:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: format_columns(x, "large")
            )

    # Format columns with small numbers
    if small_num_cols:
        for col in small_num_cols:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: format_columns(x, "small")
            )

    return formatted_df


def log_transform(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply a log transformation to specified columns in a DataFrame.

    This function applies a log transformation (base e) to the specified
    columns of the input DataFrame. The log-transformed columns are
    appended to the DataFrame with the suffix '_log'. If a column
    contains a negative value, a log transformation is not possible. In this
    case, a warning message will be printed, and the function will continue
    and try to transform additional columns.

    Use this function when you need to log-transform skewed columns in
    a DataFrame to approximate a more normal distribution for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the columns to be log-transformed.
    columns : List[str], optional
        List of column names to be log-transformed. If None, all columns
        in the DataFrame will be considered. Default is None.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the log-transformed columns appended. The
        log-transformed columns have the suffix '_log'.

    Examples
    --------
    Prepare data for the examples:

    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [10, 20, 30, 40, 50],
    ...     'C': [100, 200, 300, 400, 500]
    ... })

    Example 1: Log-transform all columns:

    >>> df_log = log_transform(df)
    >>> df_log
       A   B    C     A_log     B_log     C_log
    0  1  10  100  0.693147  2.397895  4.615121
    1  2  20  200  1.098612  3.044522  5.303305
    2  3  30  300  1.386294  3.433987  5.707110
    3  4  40  400  1.609438  3.713572  5.993961
    4  5  50  500  1.791759  3.931826  6.216606

    Example 2: Log-transform specific columns:

    >>> df_log = log_transform(df, columns=['A', 'C'])
    >>> df_log
       A   B    C     A_log     C_log
    0  1  10  100  0.693147  4.615121
    1  2  20  200  1.098612  5.303305
    2  3  30  300  1.386294  5.707110
    3  4  40  400  1.609438  5.993961
    4  5  50  500  1.791759  6.216606

    Example 3: Encounter an error with a negative value:

    >>> df['D'] = [-1, 2, 3, 4, 5]
    >>> df['E'] = [5, 4, 3, 2, 1]
    >>> df_log = log_transform(df)
    WARNING: Column 'D' has negative values and cannot be log-transformed.
    >>> df_log
       A   B    C  D  E     A_log     B_log     C_log     E_log
    0  1  10  100 -1  5  0.693147  2.397895  4.615121  1.791759
    1  2  20  200  2  4  1.098612  3.044522  5.303305  1.609438
    2  3  30  300  3  3  1.386294  3.433987  5.707110  1.386294
    3  4  40  400  4  2  1.609438  3.713572  5.993961  1.098612
    4  5  50  500  5  1  1.791759  3.931826  6.216606  0.693147
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    df_log = df.copy(deep=True)

    # If columns parameter is not provided, use all columns in the DataFrame
    if columns is None:
        columns = df.columns

    # Initialize an empty list to store the names of log-transformed columns
    log_columns = []

    # Iterate over the specified columns and apply log transformation
    for col in columns:
        # Check if the column has negative values
        if df[col].min() < 0:
            print(f"WARNING: Column '{col}' has negative values and cannot be log-transformed.")
            # Skip this iteration and go to the next column
            continue

        # Apply log transformation and append the transformed column
        df_log[col + '_log'] = np.log1p(df[col])
        log_columns.append(col + '_log')

    return df_log


def model_summary(
        model: keras.Model
) -> pd.DataFrame:
    """
    Create a DataFrame summary of a Keras model's architecture and parameters.

    This function takes a Keras model as input and returns a pandas DataFrame
    containing a summary of the model's architecture, including the model name,
    type, total parameters, trainable parameters, non-trainable parameters, layer
    names, types, activations, output shapes, the number of parameters, and the
    parameter sizes in bytes for each layer.

    Use this function when you need to obtain a structured summary of a Keras
    model's architecture and parameters for analysis, reporting, or
    visualization purposes. This is also used to test some other functions
    where the model.summary() output varies enough to fail the test cases.

    Parameters
    ----------
    model : keras.Model
        The Keras model for which to generate the summary.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the model summary, with columns for layer
        name, type, activation, output shape, number of parameters, and parameter
        size in bytes. Additional rows are included to show the total, trainable,
        and non-trainable parameters along with their byte sizes.

    Examples
    --------
    >>> pd.set_option('display.max_columns', None)  # For test consistency
    >>> pd.set_option('display.width', None)  # For test consistency
    >>> model = keras.Sequential([
    ...     keras.layers.Input(shape=(10,), name='Input'),
    ...     keras.layers.Dense(64, activation='relu', name='Dense_1'),
    ...     keras.layers.Dense(32, activation='relu', name='Dense_2'),
    ...     keras.layers.Dense(1, activation='sigmoid', name='Dense_3'),
    ... ], name='Sequential_Model')
    >>> model.build()
    >>> model_summary(model)  #doctest: +NORMALIZE_WHITESPACE
            Item                  Name         Type Activation Output Shape  Parameters    Bytes
    0      Model      Sequential_Model   Sequential       None         None         NaN      NaN
    1      Input                 Input  KerasTensor       None   (None, 10)         0.0      0.0
    2      Layer               Dense_1        Dense       relu   (None, 64)       704.0   2816.0
    3      Layer               Dense_2        Dense       relu   (None, 32)      2080.0   8320.0
    4      Layer               Dense_3        Dense    sigmoid    (None, 1)        33.0    132.0
    5  Statistic          Total Params         None       None         None      2817.0  11268.0
    6  Statistic      Trainable Params         None       None         None      2817.0  11268.0
    7  Statistic  Non-Trainable Params         None       None         None         0.0      0.0
    """
    if not model.built:
        print("Model is not built. Please build the model by calling `model.build(input_shape)` or by running `model.fit()` with some data.")
        return pd.DataFrame()  # Return an empty DataFrame if the model is not built

    def format_size(num_params):
        return num_params * 4  # Assuming parameters are float32, each taking 4 bytes

    layers_summary = []

    # Model row
    layers_summary.append(["Model", model.name, model.__class__.__name__, None, None, None, None])

    # Input layer(s)
    for input_tensor in model.inputs:
        layers_summary.append([
            "Input", input_tensor.name.split(':')[0], input_tensor.__class__.__name__,
            None, str(input_tensor.shape), 0, 0
        ])

    # Layers
    for layer in model.layers:
        activation = getattr(layer, 'activation', None)
        activation_name = activation.__name__ if activation else None
        try:
            output_shape = str(layer.output.shape)
        except AttributeError:
            output_shape = 'Unavailable'
        layers_summary.append([
            "Layer", layer.name, layer.__class__.__name__, activation_name,
            output_shape, layer.count_params(), format_size(layer.count_params())
        ])

    # Statistics
    total_params = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_variables)
    non_trainable_params = total_params - trainable_params

    layers_summary.append(["Statistic", "Total Params", None, None, None, total_params, format_size(total_params)])
    layers_summary.append(["Statistic", "Trainable Params", None, None, None, trainable_params, format_size(trainable_params)])
    layers_summary.append(["Statistic", "Non-Trainable Params", None, None, None, non_trainable_params, format_size(non_trainable_params)])

    summary_df = pd.DataFrame(layers_summary, columns=["Item", "Name", "Type", "Activation", "Output Shape", "Parameters", "Bytes"])

    return summary_df


def split_dataframe(
        df: pd.DataFrame,
        n: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into categorical and numerical columns.

    This function splits the input DataFrame into two separate DataFrames based on
    the number of unique values in each column. Columns with `n` or fewer unique
    values are considered categorical and are placed in `df_cat`, while columns
    with more than `n` unique values are considered numerical and are placed in
    `df_num`.

    Use this function when you need to separate categorical and numerical columns
    in a DataFrame for further analysis or processing.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to split.
    n : int
        The maximum number of unique values for a column to be considered
        categorical.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames:
        - df_cat: Contains the categorical columns of `df`.
        - df_num: Contains the numerical columns of `df`.

    Examples
    --------
    Prepare the data for the examples:

    >>> data = {
    ...     'A': [5.1, 2.0, 3.2, 1.4, 7.2],
    ...     'B': ['Yes', 'No', 'No', 'Yes', 'No'],
    ...     'C': [10, 20, 30, 40, 50],
    ...     'D': ['High', 'Low', 'High', 'Low', 'Low']
    ... }
    >>> df = pd.DataFrame(data)

    Example 1: Split the DataFrame based on 2 unique values:

    >>> df_cat, df_num = split_dataframe(df, n=2)
    >>> df_cat
         B     D
    0  Yes  High
    1   No   Low
    2   No  High
    3  Yes   Low
    4   No   Low
    >>> df_num
         A   C
    0  5.1  10
    1  2.0  20
    2  3.2  30
    3  1.4  40
    4  7.2  50
    """
    # Initialize the 2 dataframes
    df_cat = pd.DataFrame()
    df_num = pd.DataFrame()

    # Check unique values of each column
    for col in df.columns:
        # If Less than or equal to n, add it to the categorical df
        if df[col].nunique() <= n:
            df_cat[col] = df[col]
        # Otherwise add it to the numerical df
        else:
            df_num[col] = df[col]

    # Return the 2 dataframes
    return df_cat, df_num


def dollars(
        x: float,
        pos: int = 0
) -> str:
    """
    Format a number as currency with thousands separators on a matplotlib chart
    axis.

    This function takes a numeric value `x` and formats it as a string with
    thousands separators and a dollar sign prefix. The `pos` parameter is required
    by the matplotlib library for tick formatting but is not used in this function.

    Use this function when you need to display currency values in a more readable
    format, particularly in the context of matplotlib or seaborn plots.

    Parameters
    ----------
    x : float
        The number to format.
    pos : int, optional
        The position of the number. This parameter is not used in the function
        but is required by matplotlib for tick formatting. Default is 0.

    Returns
    -------
    str
        The formatted number as a string with thousands separators and dollar sign.

    Examples
    --------
    Example 1: Format a large currency value with default parameters:

    >>> x = 1234567.89
    >>> formatted_num = dollars(x)
    >>> print(formatted_num)
    $1,234,567

    Example 2: Use the function for tick formatting in a seaborn scatterplot:

    >>> import pandas as pd
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.ticker import FuncFormatter

    >>> # Create a sample DataFrame for plotting
    >>> data = {
    ...     'housing_median_age': [41.0, 21.0, 52.0, 52.0, 52.0, 52.0, 52.0],
    ...     'total_rooms': [880.0, 7099.0, 1467.0, 1274.0, 1627.0, 919.0, 2535.0],
    ...     'median_house_value': [452600.0, 358500.0, 352100.0, 341300.0,
    ...     342200.0, 269700.0, 299200.0]
    ... }
    >>> df = pd.DataFrame(data)

    >>> plt.figure(figsize=(10, 6))  # doctest: +SKIP
    >>> plt.title('Total Rooms vs. Median House Value', fontsize=18, pad=15)  # doctest: +SKIP
    >>> sns.scatterplot(data=df, x='total_rooms', y='median_house_value')  # doctest: +SKIP
    >>> plt.xlabel('Total Rooms', fontsize=14, labelpad=10)  # doctest: +SKIP
    >>> plt.ylabel('Median House Value', fontsize=14)  # doctest: +SKIP
    >>> plt.gca().yaxis.set_major_formatter(FuncFormatter(dollars))
    >>> plt.show()  # Displays the plot (visual output not shown)  # doctest: +SKIP
    """
    s = '${:0,d}'.format(int(x))
    return s


def thousands(
        x: float,
        pos: int = 0
) -> str:
    """
    Format a number with thousands separators on a matplotlib chart axis.

    This function takes a numeric value `x` and formats it as a string with
    thousands separators. The `pos` parameter is required by the matplotlib
    library for tick formatting but is not used in this function.

    Use this function when you need to display large numbers in a more readable
    format, particularly in the context of matplotlib or seaborn plots.

    Parameters
    ----------
    x : float
        The number to format.
    pos : int, optional
        The position of the number. This parameter is not used in the function
        but is required by matplotlib for tick formatting. Default is 0.

    Returns
    -------
    str
        The formatted number as a string with thousands separators.

    Examples
    --------
    Example 1: Format a large number with default parameters:

    >>> x = 1234567.89
    >>> formatted_num = thousands(x)
    >>> print(formatted_num)
    1,234,567

    Example 2: Use the function for tick formatting in a seaborn histogram plot:

    >>> import pandas as pd
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.ticker import FuncFormatter

    >>> # Create a sample DataFrame for plotting
    >>> data = {
    ...     'housing_median_age': [41.0, 21.0, 52.0, 52.0, 52.0, 52.0, 52.0],
    ...     'total_rooms': [880.0, 7099.0, 1467.0, 1274.0, 1627.0, 919.0, 2535.0],
    ...     'median_house_value': [452600.0, 358500.0, 352100.0, 341300.0,
    ...     342200.0, 269700.0, 299200.0]
    ... }
    >>> df = pd.DataFrame(data)

    >>> plt.figure(figsize=(10, 6))  # doctest: +SKIP
    >>> plt.title('Total Rooms vs. Median House Value', fontsize=18, pad=15)  # doctest: +SKIP
    >>> sns.scatterplot(data=df, x='total_rooms', y='median_house_value')  # doctest: +SKIP
    >>> plt.xlabel('Total Rooms', fontsize=14, labelpad=10)  # doctest: +SKIP
    >>> plt.ylabel('Median House Value', fontsize=14)  # doctest: +SKIP
    >>> plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands))
    >>> plt.show()  # Displays the plot (visual output not shown)  # doctest: +SKIP
    """
    s = '{:0,d}'.format(int(x))
    return s


# Classes
class DebugPrinter:
    """
    Conditionally print debugging information during the execution of a script.

    This class provides a simple way to print debugging information during the
    execution of a script. By setting the `debug` attribute to True, you can
    enable or disable debugging output throughout the script. The `print()`
    method works like the built-in `print()` function but only prints output
    when debugging is enabled.

    Use this class when you need to easily control and print debugging messages
    in your script, allowing you to enable or disable debugging output as
    needed. It allows you to avoid nesting a bunch of print statements
    underneath an "if debug:" statement, and it's lighter weight than a full
    logging setup.

    Parameters
    ----------
    debug : bool, optional
        Whether to enable debugging output. Default is False.

    Examples
    --------
    Set some test variables for the examples:

    >>> name = 'Setting'
    >>> value = 10

    Example 1: Create a DebugPrinter object and print a debug message:

    >>> db = DebugPrinter(debug=True)
    >>> db.print('This is a debug message.')
    This is a debug message.

    Example 2: Disable debugging and print a message that doesn't display:

    >>> db.set_debug(False)
    >>> db.print("This is a debug message that won't show.")

    Example 3: Re-enable debug, and print a formatted message with variables:

    >>> db.set_debug(True)
    >>> db.print(f'This is a debug message. ({name}: {value})')
    This is a debug message. (Setting: 10)
    """

    def __init__(
            self,
            debug: bool = False
    ):
        """
        Initialize the DebugPrinter object with the specified debugging setting.
        """
        self.debug = debug

    def print(self, *args, **kwargs):
        """
        Print debugging information if debugging is enabled.

        Parameters
        ----------
        *args
            Any number of positional arguments to print.
        **kwargs
            Any keyword arguments to pass to the built-in `print()` function.
        """
        if self.debug:
            print(*args, **kwargs)

    def set_debug(self, debug: bool):
        """
        Set the debugging setting to enable or disable debugging output.

        Parameters
        ----------
        debug : bool
            Whether to enable or disable debugging output.
        """
        self.debug = debug


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Apply logarithmic transformation to numerical features.

    This transformer applies a logarithmic transformation to the input
    features using `np.log1p()`, which calculates the natural logarithm
    of 1 plus the input values. It is useful for transforming skewed
    distributions to be more approximately normal.

    The transformer inherits from BaseEstimator and TransformerMixin
    to ensure compatibility with scikit-learn pipelines and model
    selection tools.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> X, _ = load_iris(return_X_y=True)
    >>> transformer = LogTransformer()
    >>> X_transformed = transformer.fit_transform(X)
    >>> transformer.get_feature_names_out(['sepal_length', 'sepal_width',
    ...                                     'petal_length', 'petal_width'])
    ['sepal_length_log', 'sepal_width_log', 'petal_length_log', 'petal_width_log']
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : None
            Ignored. This parameter exists only for compatibility with
            scikit-learn pipelines.

        Returns
        -------
        self : LogTransformer
            The fitted transformer.
        """
        return self

    def transform(self, X):
        """
        Apply the logarithmic transformation to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features)
            The transformed data.
        """
        return np.log1p(X)

    def get_feature_names_out(self, input_features=None):
        """
        Get the feature names after applying the transformation.

        Parameters
        ----------
        input_features : list of str, default=None
            The input feature names. If None, the feature names will be
            generated as 'x0', 'x1', etc.

        Returns
        -------
        feature_names_out : list of str
            The feature names after applying the transformation, suffixed
            with '_log'.
        """
        return [f"{col}_log" for col in input_features]

