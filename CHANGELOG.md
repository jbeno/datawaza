# Changelog

All notable changes to the Datawaza project will be documented in this file.

## [0.1.3] - 2024-04-08

### Added
- New functions added to Model module (model.py):
  - compare_models() - Find the best classification model and hyper-parameters for a dataset.
  - create_nn_binary() - Create a binary classification neural network model.
  - create_nn_multi() - Create a multi-class classification neural network model.
    - Note: Known issue with KerasClassifier calling OneHotEncoder with the 'sparse' parameter, which was renamed to 'sparse_output' in scikit-learn 1.2. This will cause a TypeError until the issue is resolved. See [SciKeras issue 316](https://github.com/adriangb/scikeras/issues/316). This is solved in SciKeras 0.13.0 once it's released.
  - eval_model() - Produce a detailed evaluation report for a classification model.
  - plot_acf_residuals() - Plot residuals, histogram, ACF, and PACF of a time series ARIMA model.
  - plot_train_history() - Plot the training and validation history of a fitted Keras model.
- New functions added to Explore module (explore.py):
  - plot_scatt() - Create a scatter plot using Seaborn's scatterplot function.
- New functions added to Tools module (tools.py):
  - DebugPrinter - Conditionally print debugging information during the execution of a script.
  - model_summary() - Create a DataFrame summary of a Keras model's architecture and parameters.

### Changed
- Package configuration
  - setup.py - Support for Python 3.9 - 3.11, modified requirements, separate [doc] and [test] tags.
    - Because Cartopy does not support Python 3.8, and that's a dependency for `plot_map_ca`, 3.8 is not supported. Because SciKeras does not support Python 3.12, and that's a dependency for `compare_models`, 3.12 is not supported.
  - Additional dependencies: importlib_resources, scikeras, xgboost, imbalanced-learn, tensorflow, keras. See [requirements.txt](requirements.txt) for the full list
- Explore module (explore.py):
  - plot_map_ca - Detect Python version. To get path to package 'data' directory that stores map files, use importlib.resources for >= 3.10, otherwise importlib_resources
- Model module (model.py):
  - eval_model() - Changed logic for handling class labels/display names to now use class_map dictionary. Bug fixes.
  - iterate_model() - Added ability to do Random Grid Search.
  - plot_results() - Added ability to switch from line chart to bar chart.
- Minor bug fixes

## [0.1.2] - 2024-03-19

First pre-release to test package installation.

### Added
- Explore module (explore.py) for data exploration and visualization:
  - get_corr() - Display the top n positive and negative correlations with a target variable in a DataFrame. 
  - get_outliers() - Detects and summarizes outliers for the specified numeric columns in a DataFrame, based on an IQR ratio. 
  - get_unique() - Print the unique values of all variables below a threshold n, including counts and percentages. 
  - plot_3d() - Create a 3D scatter plot using Plotly Express. 
  - plot_charts() - Display multiple bar plots and histograms for categorical and/or continuous variables in a DataFrame, with an option to dimension by the specified hue.
  - plot_corr() - Plot the top n correlations of one variable against others in a DataFrame.
  - plot_map_ca() - Plot longitude and latitude data on a geographic map of California.
- Clean module (clean.py) for data cleaning:
  - convert_data_values() - Convert mixed data values (ex: GB, MB, KB) to a common unit of measurement.
  - convert_dtypes() - Convert specified columns in a DataFrame to the desired data type.
  - convert_time_values() - Convert time values in specified columns of a DataFrame to a target format.
  - reduce_multicollinearity() - Reduce multicollinearity in a DataFrame by removing highly correlated features.
  - split_outliers() - Split a DataFrame into two based on the presence of outliers.
- Model module (model.py) for model iteration and evaluation:
  - create_pipeline() - Create a custom pipeline for data preprocessing and modeling. 
  - create_results_df() - Initialize the results_df DataFrame with the columns required for iterate_model. 
  - iterate_model() - Iterate and evaluate a model pipeline with specified parameters. 
  - plot_results() - Plot the results of model iterations and select the best metric.
- Tools module (tools.py) with helper functions:
  - LogTransformer - Apply logarithmic transformation to numerical features.
  - calc_pfi() - Calculate Permutation Feature Importance for a trained model.
  - calc_vif() - Calculate the Variance Inflation Factor (VIF) for each feature. 
  - check_for_duplicates() - Check for duplicate items (ex: column names) across multiple lists.
  - extract_coef() - Extract feature names and coefficients from a trained model.
  - format_df() - Format columns of a DataFrame as either large or small numbers.
  - log_transform() - Apply a log transformation to specified columns in a DataFrame.
  - split_dataframe() - Split a DataFrame into categorical and numerical columns.
  - thousand_dollars() - Format a number as currency with thousands separators on a matplotlib chart axis.
  - thousands() - Format a number with thousands separators on a matplotlib chart axis.
- Documentation site (datawaza.com)
  - API module documentation based on Sphinx and Readthedocs.io
  - User Guide in the form of a Jupyter notebook with examples of every function
- Test cases via Doctest examples in each function, minimal coverage

## [0.0.1] - 2023-08-20

- 2023-08-20: Initial setup of the `datawaza` package structure
