# Changelog

All notable changes to the Datawaza project will be documented in this file.

## [0.1.3] - UNPUBLISHED

### Added
- Model module (model.py):
  - compare_models() - Find the best classification model and hyper-parameters for a dataset.
  - eval_model() - Produce a detailed evaluation report for a classification model.
  - plot_acf_residuals() - Plot residuals, histogram, ACF, and PACF of a time series ARIMA model.

### Changed
- Package configuration
  - setup.py - Support for Python 3.9 - 3.12, modified requirements, separate [doc] and [test] tags.
  - Additional dependencies: importlib_resources, scikeras, xgboost, imbalanced-learn
- Explore module (explore.py):
  - plot_map_ca - Detect Python version. To get path to package 'data' directory that stores map files, use importlib.resources for >= 3.10, otherwise importlib_resources
- Model module (model.py):
  - iterate_model() - Added ability to do Random Grid Search.
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
