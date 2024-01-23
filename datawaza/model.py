"""
This module provides tools to streamline data modeling workflows.
It contains functions to set up pipelines, iterate over models, and evaluate results.
"""

# Metadata
__author__ = "Jim Beno"
__email__ = "jim@jimbeno.net"
__version__ = "0.1"
__license__ = "GNU GPLv3"

# Imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, FunctionTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from category_encoders import JamesSteinEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from datetime import datetime
from joblib import dump, load
import pytz
import os


# Constants and Global Variables

# List of classes that support the .coef_ attribute
SUPPORTED_COEF_CLASSES = (
    linear_model.LogisticRegression,
    linear_model.LogisticRegressionCV,
    linear_model.PassiveAggressiveClassifier,
    linear_model.Perceptron,
    linear_model.RidgeClassifier,
    linear_model.RidgeClassifierCV,
    linear_model.SGDClassifier,
    linear_model.SGDOneClassSVM,
    linear_model.LinearRegression,
    linear_model.Ridge,
    linear_model.RidgeCV,
    linear_model.SGDRegressor,
    linear_model.ElasticNet,
    linear_model.ElasticNetCV,
    linear_model.Lars,
    linear_model.LarsCV,
    linear_model.Lasso,
    linear_model.LassoCV,
    linear_model.LassoLars,
    linear_model.LassoLarsCV,
    linear_model.LassoLarsIC,
    linear_model.OrthogonalMatchingPursuit,
    linear_model.OrthogonalMatchingPursuitCV,
    linear_model.ARDRegression,
    linear_model.BayesianRidge,
    linear_model.HuberRegressor,
    linear_model.QuantileRegressor,
    linear_model.RANSACRegressor,
    linear_model.TheilSenRegressor
)

# Define the results_df global variable that will store the results of iterate_model if Save=True:
results_df = None


# This needs to be initialized in notebook with the following lines of code:
# import datawaza as dw
# dw.results_df = pd.DataFrame(columns=['Iteration', 'Train MSE', 'Test MSE', 'Train RMSE', 'Test RMSE', 'Train MAE',
#                'Test MAE', 'Train R^2 Score', 'Test R^2 Score', 'Pipeline', 'Best Grid Params', 'Note', 'Date'])


# Functions
def supports_coef(estimator):
    """Check if estimator supports .coef_"""
    return isinstance(estimator, SUPPORTED_COEF_CLASSES)


def extract_features_and_coefficients(grid_or_pipe, X, debug=False):
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
            print(f"Processing step: {step_name} in {step_transformer}")  # Debugging

        n_features_in = len(current_features)  # Number of features at the start of this step

        # If transformer is a ColumnTransformer
        if isinstance(step_transformer, ColumnTransformer):
            new_features = []  # Collect new features from this step
            step_transformer_list = step_transformer.transformers_
            for name, trans, columns in step_transformer_list:
                # OneHotEncoder or similar expanding transformers
                if hasattr(trans, 'get_feature_names_out'):
                    out_features = list(trans.get_feature_names_out(columns))
                    new_features.extend(out_features)
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

        # Reduction
        elif hasattr(step_transformer, 'get_support'):
            mask = step_transformer.get_support()
            # Update selected column in mapping
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

    return final_data[['feature_name', 'coefficients']]


def calc_fpi(model, X, y, n_repeats=10, random_state=42):
    # Calculate Feature Permutation Importance to find out which features have the most effect
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)

    return pd.DataFrame({"Variables": X.columns,
                         "Score Mean": r.importances_mean,
                         "Score Std": r.importances_std}).sort_values(by="Score Mean", ascending=False)


def calc_vif(exogs, data):
    # Set a high threshold, e.g., 1e10, for very large VIFs
    MAX_VIF = 1e10

    vif_dict = {}

    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        # split the dataset, one independent variable against all others
        X, y = data[not_exog], data[exog]

        # fit the model and obtain R^2
        r_squared = LinearRegression().fit(X, y).score(X, y)

        # compute the VIF, with a check for r_squared close to 1
        if 1 - r_squared < 1e-5:  # or some other small threshold that makes sense for your application
            vif = MAX_VIF
        else:
            vif = 1 / (1 - r_squared)

        vif_dict[exog] = vif

    return pd.DataFrame({"VIF": vif_dict})


# MODEL ITERATION: default_config, create_pipeline, iterate_model

# default_config: Version 0.1
# Default configuration of parameters used by iterate_model and create_pipeline
# New configurations can be passed in by the user when function is called
#

# create_pipeline: Version 0.1
#
def create_pipeline(transformer_keys=None, scaler_key=None, selector_key=None, model_key=None, config=None,
                    X_cat_columns=None, X_num_columns=None):
    """
    Creates a pipeline for data preprocessing and modeling.

    This function allows for flexibility in defining the preprocessing and
    modeling steps of the pipeline. You can specify which transformers to apply
    to the data, whether to scale the data, and which model to use for predictions.
    If a step is not specified, it will be skipped.

    Parameters:
    - model_key (str): The key corresponding to the model in the config['models'] dictionary.
    - transformer_keys (list of str, str, or None): The keys corresponding to the transformers
        to apply to the data. This can be a list of string keys or a single string key corresponding
        to transformers in the config['transformers'] dictionary. If not provided, no transformers will be applied.
    - scaler_key (str or None): The key corresponding to the scaler to use to scale the data.
        This can be a string key corresponding to a scaler in the config['scalers'] dictionary.
        If not provided, the data will not be scaled.
    - selector_key (str or None): The key corresponding to the feature selector.
        This can be a string key corresponding to a scaler in the config['selectors'] dictionary.
        If not provided, no feature selection will be performed.
    - X_num_columns (list-like, optional): List of numeric columns from the input dataframe. This is used
        in the default_config for the relevant transformers.
    - X_cat_columns (list-like, optional): List of categorical columns from the input dataframe. This is used
        in the default_config for the elevant encoders.

    Returns:
    pipeline (sklearn.pipeline.Pipeline): A scikit-learn pipeline consisting of the specified steps.

    Example:
    pipeline = create_pipeline('linreg', transformer_keys=['ohe', 'poly2'], scaler_key='stand', config=my_config)
    """

    # Check for configuration file parameter, if none, use default in library
    if config is None:
        # If no column lists are provided, raise an error
        if not X_cat_columns and not X_num_columns:
            raise ValueError("If no config is provided, X_cat_columns and X_num_columns must be passed.")
        config = {
            'transformers': {
                'ohe': (OneHotEncoder(drop='if_binary', handle_unknown='ignore'), X_cat_columns),
                'ord': (OrdinalEncoder(), X_cat_columns),
                'js': (JamesSteinEncoder(), X_cat_columns),
                'poly2': (PolynomialFeatures(degree=2, include_bias=False), X_num_columns),
                'poly2_bias': (PolynomialFeatures(degree=2, include_bias=True), X_num_columns),
                'poly3': (PolynomialFeatures(degree=3, include_bias=False), X_num_columns),
                'poly3_bias': (PolynomialFeatures(degree=3, include_bias=True), X_num_columns),
                'log': (FunctionTransformer(np.log1p, validate=True), X_num_columns)
            },
            'scalers': {
                'stand': StandardScaler(),
                'robust': RobustScaler(),
                'minmax': MinMaxScaler()
            },
            'selectors': {
                'sfs': SequentialFeatureSelector(LinearRegression()),
                'sfs_7': SequentialFeatureSelector(LinearRegression(), n_features_to_select=7),
                'sfs_6': SequentialFeatureSelector(LinearRegression(), n_features_to_select=6),
                'sfs_5': SequentialFeatureSelector(LinearRegression(), n_features_to_select=5),
                'sfs_4': SequentialFeatureSelector(LinearRegression(), n_features_to_select=4),
                'sfs_3': SequentialFeatureSelector(LinearRegression(), n_features_to_select=3),
                'sfs_bw': SequentialFeatureSelector(LinearRegression(), direction='backward')
            },
            'models': {
                'linreg': LinearRegression(),
                'ridge': Ridge(),
                'lasso': Lasso(random_state=42),
                'random_forest': RandomForestRegressor(),
                'gradient_boost': GradientBoostingRegressor(),
            }
        }

    # Initialize an empty list for the transformation steps
    steps = []

    # If transformers are provided, add them to the steps
    if transformer_keys is not None:
        transformer_steps = []

        for key in (transformer_keys if isinstance(transformer_keys, list) else [transformer_keys]):
            transformer, cols = config['transformers'][key]

            transformer_steps.append((key, transformer, cols))

        # Create column transformer
        col_trans = ColumnTransformer(transformer_steps, remainder='passthrough')
        transformer_name = 'Transformers: ' + '_'.join(transformer_keys) if isinstance(transformer_keys,
                                                                                       list) else 'Transformers: ' + transformer_keys
        steps.append((transformer_name, col_trans))

    # If a scaler is provided, add it to the steps
    if scaler_key is not None:
        scaler_obj = config['scalers'][scaler_key]
        steps.append(('Scaler: ' + scaler_key, scaler_obj))

    # If a selector is provided, add it to the steps
    if selector_key is not None:
        selector_obj = config['selectors'][selector_key]
        steps.append(('Selector: ' + selector_key, selector_obj))

    # If a model is provided, add it to the steps
    if model_key is not None:
        model_obj = config['models'][model_key]
        steps.append(('Model: ' + model_key, model_obj))

    # Create and return pipeline
    pipeline = Pipeline(steps)
    return pipeline


def iterate_model(x_train, x_test, y_train, y_test, model=None, transformers=None, scaler=None, selector=None,
                  drop=None, iteration='1', note='', save=False, export=False, plot=False, coef=False, perm=False,
                  vif=False, cross=False, cv_folds=5, config=None, debug=False, grid=False, grid_params=None,
                  grid_cv=None, grid_score='r2', grid_verbose=1, decimal=2):
    """
    Creates a pipeline from specified parameters for transformers, scalers, and models. Parameters must be
    defined in configuration dictionary containing 3 dictionaries: transformer_dict, scaler_dict, model_dict.
    See 'default_config' in this library file for reference, customize at will. Then fits the pipeline to the passed
    training data, and evaluates its performance with both test and training data. Options to see plots of residuals
    and actuals vs. predicted, save results to results_df with user-defined note, display coefficients, calculate
    permutation feature importance, variance inflation factor (VIF), and cross-validation.

    Parameters:
    - X_train, X_test: Training and test feature sets.
    - y_train, y_test: Training and test target sets.
    - config: Configuration dictionary of parameters for pipeline construction (see default_config)
    - model: Key for the model to be used (ex: 'linreg', 'lasso', 'ridge').
    - transformers: List of transformation keys to apply (ex: ['ohe', 'poly2']).
    - scaler: Key for the scaler to be applied (ex: 'stand')
    - selector: Key for the selector to be applied (ex: 'sfs')
    - drop: List of columns to be dropped from the training and test sets.
    - iteration: A string identifier for the iteration.
    - note: Any note or comment to be added for the iteration.
    - save: Boolean flag to decide if the results should be saved to the global results dataframe (results_df).
    - plot: Flag to plot residual plots and actuals vs. predicted for training and test data.
    - coef: Flag to print and plot model coefficients.
    - perm: Flag to compute and display permutation feature importance.
    - vif: Flag to calculate and display Variance Inflation Factor for features.
    - cross: Flag to perform cross-validation and print results.
    - cv_folds: Number of folds to be used for cross-validation if cross=True.
    - debug: Flag to show debugging information like the details of the pipeline.

    Prerequisites:
    - Dictionaries of parameters for transformers, scalers, and models: transformer_dict, scaler_dict, model_dict.
    - Lists identifying columns for various transformations and encodings, e.g., ohe_columns, ord_columns, etc.

    Outputs:
    - Prints results, performance metrics, and other specified outputs.
    - Updates the global results dataframe if save=True.
    - Displays plots based on flags like plot, coef.

    Example:
    iterate_model(x_train, x_test, y_train, y_test, transformers=['ohe','poly2'], scaler='stand', model='linreg',
        drop=['col1'], iteration="1", save=True, plot=True)
    """
    # Drop specified columns from x_train and x_test
    if drop is not None:
        x_train = x_train.drop(columns=drop)
        x_test = x_test.drop(columns=drop)
        if debug:
            print('Drop:', drop)
            print('x_train.columns', x_train.columns)
            print('x_test.columns', x_test.columns)

    # Check for configuration file parameter, if none, use default in library
    if config is None:
        X_num_columns = x_train.select_dtypes(include=[np.number]).columns.tolist()
        X_cat_columns = x_train.select_dtypes(exclude=[np.number]).columns.tolist()
        if debug:
            print('Config:', config)
            print('X_num_columns:', X_num_columns)
            print('X_cat_columns:', X_cat_columns)
    else:
        X_num_columns = None
        X_cat_columns = None

    # Create a pipeline from transformer and model parameters
    pipe = create_pipeline(transformer_keys=transformers, scaler_key=scaler, selector_key=selector, model_key=model,
                           config=config, X_cat_columns=X_cat_columns, X_num_columns=X_num_columns)
    if debug:
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
    print(f'{timestamp}\n')

    if cross or grid:
        print('Cross Validation:\n')
    # Before fitting the pipeline, check if cross-validation is desired:
    if cross:
        # Flatten y_train for compatibility
        yn_train_flat = y_train.values.flatten() if isinstance(y_train, pd.Series) else np.array(y_train).flatten()
        cv_scores = cross_val_score(pipe, x_train, yn_train_flat, cv=cv_folds, scoring='r2')

        print(f'Cross-Validation (R^2) Scores for {cv_folds} Folds:')
        for i, score in enumerate(cv_scores, 1):
            print(f'Fold {i}: {score:{format_str}}')
        print(f'Average: {np.mean(cv_scores):{format_str}}')
        print(f'Standard Deviation: {np.std(cv_scores):{format_str}}\n')

    if grid:

        grid = GridSearchCV(pipe, param_grid=config['params'][grid_params], cv=config['cv'][grid_cv],
                            scoring=grid_score, verbose=grid_verbose)
        if debug:
            print('Grid: ', grid)
            print('Grid Parameters: ', grid.get_params())
        # Fit the grid and predict
        grid.fit(x_train, y_train)
        # best_model = grid.best_estimator_
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
        # print(f'Best Grid parameters: {best_grid_params}\n')
        param_str = ', '.join(f"{key}: {value}" for key, value in best_grid_params.items())
        print(f"Best Grid parameters: {param_str}\n")
        # print(f'Best Grid estimator: {best_grid_estimator}')
        # print(f'Best Grid index: {best_grid_index}')
        # print(f'Grid results: {grid_results}')

    # Print the results
    print('Predictions:')
    print(f'{"":<15} {"Train":>15} {"Test":>15}')
    # print('-'*55)
    print(f'{"MSE:":<15} {yn_train_mse:>15{format_str}} {yn_test_mse:>15{format_str}}')
    print(f'{"RMSE:":<15} {yn_train_rmse:>15{format_str}} {yn_test_rmse:>15{format_str}}')
    print(f'{"MAE:":<15} {yn_train_mae:>15{format_str}} {yn_test_mae:>15{format_str}}')
    print(f'{"R^2 Score:":<15} {train_score:>15{format_str}} {test_score:>15{format_str}}')

    if save:
        # Access to the dataframe for storing results
        global results_df
        # Check if results_df exists in the global scope
        if 'results_df' not in globals():
            # Create results_df if it doesn't exist
            results_df = pd.DataFrame(columns=['Iteration', 'Train MSE', 'Test MSE', 'Train RMSE', 'Test RMSE',
                                               'Train MAE', 'Test MAE', 'Train R^2 Score', 'Test R^2 Score',
                                               'Best Grid Mean Score',
                                               'Best Grid Params', 'Pipeline', 'Note', 'Date'])
            print("\n'results_df' not found in global scope. A new one has been created.")

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
        results_df = pd.concat([results_df, df_iteration], ignore_index=True)

        # Permutation Feature Importance
    if perm:
        print("\nPermutation Feature Importance:")
        if grid:
            perm_imp_res = calc_fpi(grid, x_train, y_train)
        else:
            perm_imp_res = calc_fpi(pipe, x_train, y_train)

        # Create a Score column
        perm_imp_res['Score'] = perm_imp_res['Score Mean'].apply(lambda x: f"{x:{format_str}}") + " ± " + perm_imp_res[
            'Score Std'].apply(lambda x: f"{x:{format_str}}")

        # Adjust the variable names for better alignment in printout
        perm_imp_res['Variables'] = perm_imp_res['Variables'].str.ljust(25)

        # Create a copy for printing and rename the 'Variables' column header to empty
        print_df = perm_imp_res.copy()
        print_df = print_df.rename(columns={"Variables": ""})

        # Print the DataFrame with only the Variables and the Score column
        print(print_df[['', 'Score']].to_string(index=False))

    if vif:
        all_numeric = not bool(x_train.select_dtypes(exclude=[np.number]).shape[1])

        if all_numeric:
            suitable = True
        else:
            # Check if transformers is not empty
            if transformers:
                transformer_list = [transformers] if isinstance(transformers, str) else transformers
                suitable_for_vif = {'ohe', 'ord', 'ohe_drop'}
                if any(t in suitable_for_vif for t in transformer_list):
                    suitable = True
                else:
                    suitable = False
            elif drop:
                suitable = True
            else:
                suitable = False

        if suitable:
            print("\nVariance Inflation Factor:")
            if all_numeric:
                vif_df = x_train
            else:
                if transformers is not None:
                    # Create a pipeline with the transformers only
                    # vif_pipe = create_pipeline(transformer_keys=transformers, config=config, X_cat_columns=X_cat_columns, X_num_columns=X_num_columns)
                    if grid:
                        vif_pipe = grid
                        feature_names = grid.best
                    elif pipe:
                        vif_pipe = pipe
                    if debug:
                        print('VIF Pipeline:', vif_pipe)
                        print('VIF Pipeline Parameters:', vif_pipe.get_params())
                    # vif_pipe.fit(x_train, y_train)
                    # feature_names = vif_pipe.get_feature_names_out()
                    #
                    transformed_data = vif_pipe.transform(x_train)
                    vif_df = pd.DataFrame(transformed_data, columns=feature_names)
            vif_results = calc_vif(vif_df.columns, vif_df).sort_values(by='VIF', ascending=False)
            vif_results['VIF'] = vif_results['VIF'].apply(lambda x: f'{{:,.{decimal}f}}'.format(x))
            print(vif_results)
        else:
            print("\nVIF calculation skipped. The transformations applied are not suitable for VIF calculation.")

    if plot:
        print('')
        y_train = y_train.values.flatten() if isinstance(y_train, pd.Series) else np.array(y_train).flatten()
        y_test = y_test.values.flatten() if isinstance(y_test, pd.Series) else np.array(y_test).flatten()

        yn_train_pred = yn_train_pred.flatten()
        yn_test_pred = yn_test_pred.flatten()

        # Generate residual plots
        plt.figure(figsize=(12, 3))

        plt.subplot(1, 2, 1)
        sns.residplot(x=y_train, y=yn_train_pred, lowess=True, scatter_kws={'s': 30, 'edgecolor': 'white'},
                      line_kws={'color': 'red', 'lw': '1'})
        plt.title(f'Training Residuals - Iteration {iteration}')

        plt.subplot(1, 2, 2)
        sns.residplot(x=y_test, y=yn_test_pred, lowess=True, scatter_kws={'s': 30, 'edgecolor': 'white'},
                      line_kws={'color': 'red', 'lw': '1'})
        plt.title(f'Test Residuals - Iteration {iteration}')

        plt.show()

        # Generate predicted vs actual plots
        plt.figure(figsize=(12, 3))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=y_train, y=yn_train_pred, s=30, edgecolor='white')
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linewidth=1)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Training Predicted vs. Actual - Iteration {iteration}')

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=y_test, y=yn_test_pred, s=30, edgecolor='white')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=1)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Test Predicted vs. Actual - Iteration {iteration}')

        plt.show()

    # Calculate coefficients if model supports
    if coef:
        # Extract features and coefficients using the function
        coefficients_df = extract_features_and_coefficients(
            grid.best_estimator_ if grid else pipe, x_train, debug=debug
        )

        # Check if there are any non-NaN coefficients
        if coefficients_df['coefficients'].notna().any():
            # Ensure the coefficients are shaped as a 2D numpy array
            coefficients = coefficients_df[['coefficients']].values
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
                coefficients = coefficients.ravel()
                plt.figure(figsize=(12, 3))
                x_values = range(1, len(coefficients) + 1)
                plt.bar(x_values, coefficients)
                plt.xticks(x_values)
                plt.xlabel('Feature')
                plt.ylabel('Value')
                plt.title('Coefficients')
                plt.axhline(y=0, color='black', linestyle='dotted', lw=1)
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

    if grid:
        return best_model, grid_results
    else:
        return best_model
