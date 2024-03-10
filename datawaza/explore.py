# explore.py – Explore module of Datawaza
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
This module provides tools to streamline exploratory data analysis.
It contains functions to find unique values, plot distributions, detect outliers,
extract the top correlations, and plot correlations.

Functions:
    - :func:`~datawaza.explore.get_corr` - Display the top `n` positive and negative correlations with a target variable in a DataFrame.
    - :func:`~datawaza.explore.get_outliers` - Detects and summarizes outliers for the specified numeric columns in a DataFrame, based on an IQR ratio.
    - :func:`~datawaza.explore.get_unique` - Print the unique values of all variables below a threshold `n`, including counts and percentages.
    - :func:`~datawaza.explore.plot_charts` - Display multiple bar plots and histograms for categorical and/or continuous variables in a DataFrame, with an option to dimension by the specified `hue`.
    - :func:`~datawaza.explore.plot_corr` - Plot the top `n` correlations of one variable against others in a DataFrame.
"""

# Metadata
__author__ = "Jim Beno"
__email__ = "jim@jimbeno.net"
__version__ = "0.1"
__license__ = "GNU GPLv3"

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from typing import Optional, Union, Tuple, List, Dict
from scipy.stats import iqr


# Functions
def get_outliers(
        df: pd.DataFrame,
        num_columns: List[str],
        ratio: float = 1.5,
        exclude_zeros: bool = False,
        plot: bool = False,
        width: int = 15,
        height: int = 2
) -> pd.DataFrame:
    """
    Detects and summarizes outliers for the specified numeric columns in a
    DataFrame, based on an IQR ratio.

    This function identifies outliers using Tukey's method, where outliers are
    considered to be those data points that fall below Q1 - `ratio` * IQR or above
    Q3 + `ratio` * IQR. You can exclude zeros from the calculations, as they can
    appear as outliers and skew your results. You can also change the default IQR
    `ratio` of  1.5. If outliers are found, they will be summarized in the returned
    DataFrame. In addition, the distributions of the variables with outliers can be
    plotted as boxplots.

    Use this function to identify outliers during the early stages of exploratory
    data analysis. With one line, you can see: total non-null, total zero values,
    zero percent, outlier count, outlier percent, skewness, and kurtosis. You can
    also visually spot outliers outside of the whiskers in the boxplots. Then you
    can decide how you want to handle the outliers (ex: log transform, drop, etc.)

    Parameters
    ----------
    df : DataFrame
        The DataFrame to analyze for outliers.
    num_columns : List[str]
        List of column names in `df` to check for outliers. These should be names
        of columns with numerical data.
    ratio : float, optional
        The multiplier for IQR to determine the threshold for outliers. Default
        is 1.5.
    exclude_zeros : bool, optional
        If set to True, zeros are excluded from the outlier calculation. Default
        is False.
    plot : bool, optional
        If set to True, box plots of outlier distributions are displayed. Default
        is False.
    width : int, optional
        The width of the plot figure. This parameter only has an effect if `plot`
        is True. Default is 15.
    height : int, optional
        The height of each subplot row. This parameter only has an effect if `plot`
        is True. Default is 2.

    Returns
    -------
    pd.DataFrame
        A DataFrame summarizing the outliers found in each specified column,
        including the number of non-null and zero values, percentage of zero
        values, count of outliers, percentage of outliers, and measures of skewness
        and kurtosis.

    Examples
    --------
    Prepare the data for the examples:

    >>> np.random.seed(0)  # For reproducibility
    >>> df = pd.DataFrame({
    ...     'A': np.random.randn(100),
    ...     'B': np.random.exponential(scale=2.0, size=100),
    ...     'C': np.random.randn(100)
    ... })
    >>> df.at[2, 'A'] = 0; df.at[5, 'A'] = 0  # Assign some zeros
    >>> df.at[3, 'B'] = np.nan; df.at[7, 'B'] = np.nan  # Assign some NaNs
    >>> num_columns = ['A', 'B', 'C']  # Store numeric columns

    Example 1: Create a dataframe that lists outlier statistics:

    >>> outlier_summary = get_outliers(df, num_columns)
    >>> print(outlier_summary)
      Column  Total Non-Null  Total Zero  ...  Outlier Percent  Skewness  Kurtosis
    1      B              98           0  ...             4.08      2.62     10.48
    0      A             100           2  ...             1.00      0.01     -0.25
    2      C             100           0  ...             1.00     -0.03      0.19
    <BLANKLINE>
    [3 rows x 8 columns]

    Example 2: Create a dataframe that lists outlier statistics, excluding zeros:

    >>> outlier_summary = get_outliers(df, num_columns, exclude_zeros=True)
    >>> print(outlier_summary)
      Column  Total Non-Null  Total Zero  ...  Outlier Percent  Skewness  Kurtosis
    0      B              98           0  ...             4.08      2.62     10.48
    1      C             100           0  ...             1.00     -0.03      0.19
    <BLANKLINE>
    [2 rows x 8 columns]
    """
    outlier_data = []

    for col in num_columns:
        non_null_data = df[col].dropna()
        if exclude_zeros:
            non_null_data = non_null_data[non_null_data != 0]
        else:
            non_null_data = non_null_data

        if non_null_data.empty:
            continue

        Q1 = np.percentile(non_null_data, 25)
        Q3 = np.percentile(non_null_data, 75)
        IQR = iqr(non_null_data)

        lower_bound = Q1 - ratio * IQR
        upper_bound = Q3 + ratio * IQR

        outliers = non_null_data[(non_null_data < lower_bound) | (non_null_data > upper_bound)]
        outlier_count = outliers.count()
        total_non_null = non_null_data.count()
        total_zero = non_null_data[non_null_data == 0].count()
        zero_percent = round((total_zero / total_non_null) * 100, 2)

        if outlier_count > 0:
            percentage = round((outlier_count / total_non_null) * 100, 2)
            skewness = round(non_null_data.skew(), 2)
            kurtosis = round(non_null_data.kurt(), 2)
            outlier_data.append([col, total_non_null, total_zero, zero_percent, outlier_count, percentage, skewness, kurtosis])

    outlier_df = pd.DataFrame(outlier_data, columns=['Column', 'Total Non-Null', 'Total Zero', 'Zero Percent', 'Outlier Count',
                                                     'Outlier Percent', 'Skewness', 'Kurtosis'])
    outlier_df = outlier_df.sort_values(by='Outlier Percent', ascending=False)

    if plot:
        plt.figure(figsize=(width, len(outlier_df) * height))
        plot_index = 1

        for index, row in outlier_df.iterrows():
            plt.subplot(len(outlier_df) // 2 + len(outlier_df) % 2, 2, plot_index)
            sns.boxplot(x=df[row['Column']], orient='h')
            plt.title(f"{row['Column']}, Outliers: {row['Outlier Count']} ({row['Outlier Percent']}%)")

            plot_index += 1

        plt.tight_layout()
        plt.show()

    return outlier_df


def get_corr(
        df: DataFrame,
        n: int = 5,
        var: Optional[str] = None,
        show_results: bool = True,
        return_arrays: bool = False
) -> Union[None, Tuple[np.ndarray, np.ndarray]]:
    """
    Display the top `n` positive and negative correlations with a target variable
    in a DataFrame.

    This function computes the correlation matrix for the provided DataFrame, and
    identifies the top `n` positively and negatively correlated pairs of variables.
    By default, it prints a summary of these correlations. Optionally, it can
    return arrays of the variable names involved in these top correlations,
    avoiding duplicates.

    Use this to quickly identify the strongest correlations with a target variable.
    You can also use this to reduce a DataFrame with a large number of features
    down to just the top `n` correlated features. Extract the names of the top
    correlated features into 2 separate arrays (one for positive, one for
    negative). Concatenate those variable lists and append the target variable. Use
    this concatenated array to create a new DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze for correlations.
    n : int, optional
        The number of top positive and negative correlations to list. Default is 5.
    var : str, optional
        A specific variable of interest. If provided, the function will only
        consider correlations involving this variable. Default is None.
    show_results : bool, optional
        Flag to indicate if the results should be printed. Default is True.
    return_arrays : bool, optional
        Flag to indicate if the function should return arrays of variable names
        involved in the top correlations. Default is False.

    Returns
    -------
    tuple, optional
        If `return_arrays` is set to True, the function returns a tuple containing
        two arrays: (1) `positive_variables`: An array of variable names involved
        in the top n positive correlations. (2) `negative_variables`: An array of
        variable names involved in the top n negative correlations. If
        `return_arrays` is False, the function returns nothing.

    Examples
    --------
    Prepare the data for the examples:

    >>> np.random.seed(0)  # For reproducibility
    >>> n_samples = 100
    >>> # Create variables
    >>> temp = np.linspace(10, 30, n_samples) + np.random.normal(0, 2, n_samples)
    >>> sales = temp * 3 + np.random.normal(0, 10, n_samples)
    >>> fuel = 100 - temp * 2 + np.random.normal(0, 5, n_samples)
    >>> humidity = 70 - temp * 1.5 + np.random.normal(0, 4, n_samples)
    >>> ac_units_sold = temp * 2 + np.random.normal(0, 15, n_samples)
    >>> # Create DataFrame
    >>> df = pd.DataFrame({'Temp': temp, 'Sales': sales, 'Fuel': fuel,
    ...                    'Humidity': humidity, 'AC_Units_Sold': ac_units_sold})

    Example 1: Print the top 'n' correlations, both positive and negative:

    >>> get_corr(df, n=2, var='Temp')
    Top 2 positive correlations:
          Variable 1 Variable 2  Correlation
    0          Sales       Temp         0.85
    1  AC_Units_Sold       Temp         0.62
    <BLANKLINE>
    Top 2 negative correlations:
      Variable 1 Variable 2  Correlation
    0       Fuel       Temp        -0.92
    1   Humidity       Temp        -0.92

    Example 2: Create arrays with the top correlated feature names:

    >>> (top_pos, top_neg) = get_corr(df, n=1, var='Temp', show_results=False,
    ...     return_arrays=True)
    >>> print(top_pos)
    ['Sales']
    >>> print(top_neg)
    ['Fuel']

    Example 3: Create a dataframe of top correlated features from those arrays:

    >>> top_features = np.concatenate((top_pos, top_neg, ['Temp']))
    >>> df_top_features = df[top_features]
    >>> print(df_top_features[:2])
           Sales       Fuel       Temp
    0  59.415821  71.097881  13.528105
    1  19.529413  76.798435  11.002335
    """
    pd.set_option('display.expand_frame_repr', False)

    corr = round(df.corr(numeric_only=True), 2)

    # Unstack correlation matrix into a DataFrame
    corr_df = corr.unstack().reset_index()
    corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']

    # If a variable is specified, filter to correlations involving that variable
    if var is not None:
        corr_df = corr_df[(corr_df['Variable 1'] == var) | (corr_df['Variable 2'] == var)]

    # Remove self-correlations and duplicates
    corr_df = corr_df[corr_df['Variable 1'] != corr_df['Variable 2']]
    corr_df[['Variable 1', 'Variable 2']] = np.sort(corr_df[['Variable 1', 'Variable 2']], axis=1)
    corr_df = corr_df.drop_duplicates(subset=['Variable 1', 'Variable 2'])

    # Sort by absolute correlation value from highest to lowest
    corr_df['AbsCorrelation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values(by='AbsCorrelation', ascending=False)

    # Drop the absolute value column
    corr_df = corr_df.drop(columns='AbsCorrelation').reset_index(drop=True)

    # Get the first n positive and negative correlations
    positive_corr = corr_df[corr_df['Correlation'] > 0].head(n).reset_index(drop=True)
    negative_corr = corr_df[corr_df['Correlation'] < 0].head(n).reset_index(drop=True)

    # Print the results
    if show_results:
        print("Top", n, "positive correlations:")
        print(positive_corr)
        print("\nTop", n, "negative correlations:")
        print(negative_corr)

    # Return the arrays
    if return_arrays:
        # Remove target variable from the arrays
        positive_variables = positive_corr[['Variable 1', 'Variable 2']].values.flatten()
        positive_variables = positive_variables[positive_variables != var]

        negative_variables = negative_corr[['Variable 1', 'Variable 2']].values.flatten()
        negative_variables = negative_variables[negative_variables != var]

        return positive_variables, negative_variables


def get_unique(
        df: DataFrame,
        n: int = 20,
        sort: str = 'count',
        show_list: bool = True,
        count: bool = True,
        percent: bool = True,
        plot: bool = False,
        cont: bool = False,
        strip: bool = False,
        dropna: bool = False,
        fig_size: Tuple[int, int] = (6, 4),
        rotation: int = 45
) -> None:
    """
    Print the unique values of all variables below a threshold `n`, including
    counts and percentages.

    This function examines the unique values of all the variables in a DataFrame.
    If the number is below a threshold `n`, it will list their unique values. For
    each value, it prints out the count and percentage of the dataset with that
    value. You can change the sort, and there are options to strip single quotes
    from the variable names, or exclude NaN values. You can optionally show
    descriptive statistics for the continuous variables able the 'n' threshold, or
    display simple plots.

    Use this to quickly examine the features of your dataset at the beginning of
    exploratory data analysis. Use `df.nunique()` to first determine how many
    unique values each variable has, and identify a number that likely separates
    the categorical from continuous numeric variables. Then run get_unique using
    that number as `n` (this avoids iterating over continuous data).

    Parameters
    ----------
    df : DataFrame
        The dataframe that contains the variables you want to analyze.
    n : int, optional
        The maximum number of unique values to consider. This helps to avoid
        iterating over continuous data. Default is 20.
    sort : str, optional
        Determines the sorting of unique values:
        - 'name' - sorts alphabetically/numerically,
        - 'count' - sorts by count of unique values (descending),
        - 'percent' - sorts by percentage of each unique value (descending).
        Default is 'count'.
    show_list : bool, optional
        If True, shows the list of unique values. Default is True.
    count : bool, optional
        If True, shows counts of each unique value. Default is False.
    percent : bool, optional
        If True, shows the percentage of each unique value. Default is False.
    plot : bool, optional
        If True, shows a basic chart for each variable. Default is False.
    cont : bool, optional
        If True, analyzes variables with unique values greater than 'n'
        as continuous data. Default is False.
    strip : bool, optional
        If True, removes single quotes from the variable names. Default is False.
    dropna : bool, optional
        If True, excludes NaN values from unique value lists. Default is False.
    fig_size : tuple, optional
        Size of figure if plotting is enabled. Default is (6, 4).
    rotation : int, optional
        Rotation angle of X axis ticks if plotting is enabled. Default is 45.

    Returns
    -------
    None
        The function prints the analysis directly.

    Examples
    --------
    Prepare the data for the examples:

    >>> df = pd.DataFrame({'Animal': ["'Cat'", "'Dog'", "'Cat'", "'Mountain Lion'",
    ...     "'Dog'", "'Dog'"],
    ...     'Sex': ['Male', 'Female', 'Male', 'Male', 'Female', np.nan],
    ...     'Weight': [6.5, 12.5, 7.7, 84.1, 22.3, 29.2]
    ... })

    Example 1: Print unique values below a threshold of 3:

    >>> get_unique(df, n=3)
    <BLANKLINE>
    CATEGORICAL: Variables with unique values equal to or below: 3
    <BLANKLINE>
    Animal has 3 unique values:
    <BLANKLINE>
        'Dog'               3   50.0%
        'Cat'               2   33.33%
        'Mountain Lion'     1   16.67%
    <BLANKLINE>
    Sex has 3 unique values:
    <BLANKLINE>
        Male         3   50.0%
        Female       2   33.33%
        nan          1   16.67%

    Example 2: Sort values by name, strip single quotes, drop NaN:

    >>> get_unique(df, n=3, sort='name', strip=True, dropna=True)
    <BLANKLINE>
    CATEGORICAL: Variables with unique values equal to or below: 3
    <BLANKLINE>
    Animal has 3 unique values:
    <BLANKLINE>
        Cat                 2   33.33%
        Dog                 3   50.0%
        Mountain Lion       1   16.67%
    <BLANKLINE>
    Sex has 2 unique values:
    <BLANKLINE>
        Female       2   40.0%
        Male         3   60.0%
    """
    # Calculate # of unique values for each variable in the dataframe
    var_list = df.nunique(axis=0, dropna=dropna)

    # Iterate through each categorical variable in the list below n
    print(f"\nCATEGORICAL: Variables with unique values equal to or below: {n}")
    for i in range(len(var_list)):
        var_name = var_list.index[i]
        unique_count = var_list.iloc[i]

        # If unique value count is less than n, get the list of values, counts, percentages
        if unique_count <= n:
            number = df[var_name].value_counts(dropna=dropna)
            if dropna:
                total_count = df[var_name].dropna().shape[0]  # Count of non-NaN entries
            else:
                total_count = df[var_name].shape[0]  # Total entries, including NaN
            perc = round(number / total_count * 100, 2)
            orig = number.index
            # Strip out the single quotes
            name = [str(n) for n in number.index]
            name = [n.strip('\'') for n in name]
            # Store everything in dataframe uv for consistent access and sorting
            uv = pd.DataFrame({'orig': orig, 'name': name, 'number': number, 'perc': perc})

            # Sort the unique values by name or count, if specified
            if sort == 'name':
                uv = uv.sort_values(by='name', ascending=True)
            elif sort == 'count':
                uv = uv.sort_values(by='number', ascending=False)
            elif sort == 'percent':
                uv = uv.sort_values(by='perc', ascending=False)

            # Print out the list of unique values for each variable
            if show_list:
                print(f"\n{var_name} has {unique_count} unique values:\n")
                for w, x, y, z in uv.itertuples(index=False):
                    # Decide on to use stripped name or not
                    if strip:
                        w = x
                    # Put some spacing after the value names for readability
                    w_str = str(w)
                    w_pad_size = uv.name.str.len().max() + 7
                    w_pad = w_str + " " * (w_pad_size - len(w_str))
                    y_str = str(y)
                    y_pad_max = uv.number.max()
                    y_pad_max_str = str(y_pad_max)
                    y_pad_size = len(y_pad_max_str) + 3
                    y_pad = y_str + " " * (y_pad_size - len(y_str))
                    if count and percent:
                        print("    " + str(w_pad) + str(y_pad) + str(z) + "%")
                    elif count:
                        print("    " + str(w_pad) + str(y))
                    elif percent:
                        print("    " + str(w_pad) + str(z) + "%")
                    else:
                        print("    " + str(w))

            # Plot countplot if plot=True
            if plot:
                print("\n")
                plt.figure(figsize=fig_size)
                if strip:
                    if sort == 'count':
                        sns.barplot(data=uv, x='name', y='number', order=uv.sort_values('number', ascending=False).name)
                    else:
                        sns.barplot(data=uv, x=uv.loc[0], y='number', order=uv.sort_values('name', ascending=True).name)
                else:
                    if sort == 'count':
                        sns.barplot(data=uv, x='orig', y='number', order=uv.sort_values('number', ascending=False).orig)
                    else:
                        sns.barplot(data=uv, x='orig', y='number', order=uv.sort_values('orig', ascending=True).orig)
                plt.title(var_name)
                plt.xlabel('')
                plt.ylabel('')
                plt.xticks(rotation=rotation)
                plt.grid(False)
                plt.show()

    if cont:
        # Iterate through each continuous variable in the list above n
        print(f"\nCONTINUOUS: Variables with unique values greater than: {n}")
        for i in range(len(var_list)):
            var_name = var_list.index[i]
            unique_count = var_list.iloc[i]

            if unique_count > n:
                print(f"\n{var_name} has {unique_count} unique values:\n")
                print(var_name)
                print(df[var_name].describe())

                # Plot countplot if plot=True
                if plot:
                    print("\n")
                    plt.figure(figsize=fig_size)
                    sns.histplot(data=df, x=var_name)
                    plt.title(var_name)
                    plt.xlabel('')
                    plt.ylabel('')
                    plt.xticks(rotation=rotation)
                    plt.grid(False)
                    plt.show()


def plot_charts(df: pd.DataFrame,
                plot_type: str = 'both',
                n: int = 10,
                ncols: int = 2,
                fig_width: int = 15,
                subplot_height: int = 4,
                rotation: int = 0,
                cat_cols: Optional[List[str]] = None,
                cont_cols: Optional[List[str]] = None,
                dtype_check: bool = True,
                sample_size: Optional[Union[int, float]] = None,
                random_state: int = 42,
                hue: Optional[str] = None,
                color_discrete_map: Optional[Dict] = None,
                normalize: bool = False,
                kde: bool = False,
                multiple: str = 'layer',
                log_scale: bool = False,
                ignore_zero: bool = False) -> None:
    """
    Display multiple bar plots and histograms for categorical and/or continuous
    variables in a DataFrame, with an option to dimension by the specified `hue`.

    This function allows you to plot a large number of distributions with one line
    of code. You choose which type of plots to create by setting `plot_type` to
    `cat`, `cont`, or `both`. Categorical variables are plotted with
    `sns.countplot` ordered by descending value counts for a clean appearance.
    Continuous variables are plotted with `sns.histplot`. There are two approaches
    to identifying categorical vs. continuous variables: (a) you can specify
    `cat_cols` and `cont_cols` as lists of the respective column names, or (b) you
    can specify `n` as the dividing line, and any variable with `n` or lower unique
    values will be treated as categorical. In addition, you can enable
    `dtype_check` on the continuous columns to only include columns of data type
    `int64` or `float64`.

    For each type of variable, it creates a subplot layout that has `ncols`
    columns, and is `fig_width` wide. It calculates how many rows are required to
    display all the plots, and each row is `subplot_height` high. Specify `hue` if
    you want to dimension the plots by another variable. You can set
    `color_discrete_map` to a color mapping dictionary for the values of the `hue`
    variable. You can also customize some parameters of the plots, such as
    `rotation` of the X axis tick labels. For categorical variables, you can
    normalize the plots to show proportions instead of counts by setting
    `normalize` to True.

    For histograms, you can display KDE lines with `kde`, and change how the `hue`
    variable appears by setting `multiple`. If you have a large amount of data that
    is taking too long to process, you can take a random sample of your data by
    setting `sample_size` to either a count or proportion. To handle skewed data,
    you have two options: (a) you can enable log scale on the X axis with
    `log_scale`, and (b) you can ignore zero values with `ignore_zero` (these can
    sometimes dominate the left end of a chart).

    Use this function to quickly visualize the distributions of your data during
    exploratory data analysis. With one line, you can produce a comprehensive
    series of plots that can help you spot issues that will require handling during
    data cleaning. By setting `hue` to your target y variable, you might be able to
    catch glimpses of potential correlations or relationships.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the variables to be analyzed.
    plot_type : str, optional
        The type of charts to plot: 'cat' for categorical, 'cont' for continuous,
        or 'both'. Default is 'both'.
    n : int, optional
        Threshold for distinguishing between categorical (≤ n unique values) and
        continuous (> n unique values) variables. Default is 10.
    ncols : int, optional
        The number of columns in the subplot grid. Default is 2.
    fig_width : int, optional
        The width of the entire plot figure (not individual subplots). Default 15.
    subplot_height : int, optional
        The height of each subplot. Default is 4.
    rotation : int, optional
        The rotation angle for x-axis labels. Default is 0.
    cat_cols : List[str], optional
        List of column names to treat as categorical variables. Inferred from
        unique count if not provided.
    cont_cols : List[str], optional
        List of column names to treat as continuous variables. Inferred from unique
        count if not provided.
    dtype_check : bool, optional
        If True, considers only numeric types for continuous variables. Default is
        True.
    sample_size : int or float, optional
        If provided, indicates the fraction (if < 1) or number (if ≥ 1) of samples
        to draw from the dataframe for histograms.
    random_state : int, optional. Default is 42
        Set random state for reproducibility if using sample_size to perform a
        random sample for histograms.
    hue : str, optional
        Name of the column for hue-based dimensioning in the plots.
    color_discrete_map : Dict, optional
        A color mapping dictionary for the values in the 'hue' variable.
    normalize : bool, optional
        If True, normalizes categorical plots to show proportions instead of
        counts. Default is False.
    kde : bool, optional
        If True, shows Kernel Density Estimate (KDE) line on continuous histograms.
        Default is False.
    multiple : str, optional
        Method to handle the hue variable in countplots. Options are 'layer',
        'dodge', 'stack', 'fill'. Default is 'layer'.
    log_scale : bool, optional
        If True, uses log scale for continuous histograms. Default is False.
    ignore_zero : bool, optional
        If True, ignores zero values in continuous histograms. Default is False.

    Returns
    -------
    None
        Creates and displays plots without returning any value.

    Examples
    --------
    Prepare the data for the examples:

    >>> df = pd.DataFrame({
    ...     'Category A': np.random.choice(['A', 'B', 'C'], size=100),
    ...     'Category B': np.random.choice(['D', 'E', 'F', 'G', 'H', 'I', 'J'],
    ...                                    size=100),
    ...     'Measure 1': np.random.randn(100),
    ...     'Measure 2': np.random.exponential(scale=2.0, size=100),
    ...     'Target': np.random.choice(['Yes', 'No'], size=100)
    ... })
    >>> cat_cols = ['Category A', 'Category B', 'Target']
    >>> num_cols = ['Measure 1', 'Measure 2']

    Example 1: Plot both categorical and continuous variables based on a boundary
    of `n` unique values:

    >>> plot_charts(df, n=7)

    Example 2: Plot only categorical variables using a column list, dimensioned
    by `hue`:

    >>> plot_charts(df, plot_type='cat', cat_cols=cat_cols, hue='Target')

    Example 3: Customize the subplot width, number of columns, and rotation of
    the X axis tick labels:

    >>> plot_charts(df, plot_type='both', n=7, fig_width=20, ncols=3, rotation=90)

    Example 4: Plot only histograms dimensioned by hue (stacked values), with KDE
    lines, and X axis in log scale:

    >>> plot_charts(df, plot_type='cont', cont_cols=num_cols, hue='Target',
    ...             multiple='stack', kde=True, log_scale=True)
    """
    def plot_categorical(df, cols, ncols, fig_width, subplot_height, rotation, sample_size, hue, color_discrete_map, normalize):
        if sample_size:
            df = df.sample(sample_size, random_state=random_state)
        nplots = len(cols)
        nrows = nplots//ncols
        if nplots % ncols:
            nrows += 1

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, nrows*subplot_height), constrained_layout=True)
        if isinstance(axs, np.ndarray):
            if len(axs.shape) > 1:
                axs = axs.ravel()
        else:
            axs = [axs]

        for i, col in enumerate(cols):
            if normalize:
                # Normalize the counts
                df_copy = df.copy()
                data = df_copy.groupby(col)[hue].value_counts(normalize=True).rename('proportion').reset_index()
                sns.barplot(data=data, x=col, y='proportion', hue=hue, palette=color_discrete_map, ax=axs[i])
                axs[i].set_ylabel('Proportion', fontsize=12)
            else:
                order = df[col].value_counts().index
                if hue == None:
                    sns.countplot(data=df, x=col, hue=col, legend=False, palette=color_discrete_map, ax=axs[i], order=order)
                else:
                    sns.countplot(data=df, x=col, hue=hue, palette=color_discrete_map, ax=axs[i], order=order)
                axs[i].set_ylabel('Count', fontsize=12)
            axs[i].set_xlabel(' ', fontsize=12)
            axs[i].set_title(col, fontsize=16, pad=10)
            axs[i].tick_params(axis='x', rotation=rotation)

        # Remove empty subplots
        for empty_subplot in axs[nplots:]:
            fig.delaxes(empty_subplot)

    def plot_continuous(df, cols, ncols, fig_width, subplot_height, sample_size, hue,
                        color_discrete_map, kde, multiple, log_scale, ignore_zero):
        if sample_size:
            df = df.sample(sample_size)
        nplots = len(cols)
        nrows = nplots//ncols
        if nplots % ncols:
            nrows += 1

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, nrows * subplot_height), constrained_layout=True)
        if isinstance(axs, np.ndarray):
            if len(axs.shape) > 1:
                axs = axs.ravel()
        else:
            axs = [axs]

        for i, col in enumerate(cols):
            if hue is not None:
                if ignore_zero:
                    sns.histplot(data=df[df[col]>0], x=col, hue=hue, palette=color_discrete_map, ax=axs[i], kde=kde, multiple=multiple, log_scale=log_scale)
                else:
                    sns.histplot(data=df, x=col, hue=hue, palette=color_discrete_map, ax=axs[i], kde=kde, multiple=multiple, log_scale=log_scale)
            else:
                if ignore_zero:
                    sns.histplot(data=df[df[col]>0], x=col, ax=axs[i])
                else:
                    sns.histplot(data=df, x=col, ax=axs[i])
            axs[i].set_title(col, fontsize=16, pad=10)
            axs[i].set_ylabel('Count', fontsize=12)
            axs[i].set_xlabel(' ', fontsize=12)
            axs[i].tick_params(axis='x', rotation=rotation)

        # Remove empty subplots
        for empty_subplot in axs[nplots:]:
            fig.delaxes(empty_subplot)

    unique_count = df.nunique()
    if cat_cols is None:
        cat_cols = unique_count[unique_count <= n].index.tolist()
        if hue in cat_cols:
            cat_cols.remove(hue)
    if cont_cols is None:
        cont_cols = unique_count[unique_count > n].index.tolist()

    if dtype_check:
        cont_cols = [col for col in cont_cols if df[col].dtype in ['int64', 'float64']]

    if plot_type == 'cat' or plot_type == 'both':
        plot_categorical(df, cat_cols, ncols, fig_width, subplot_height, rotation, sample_size, hue, color_discrete_map, normalize)
    if plot_type == 'cont' or plot_type == 'both':
        plot_continuous(df, cont_cols, ncols, fig_width, subplot_height, sample_size, hue, color_discrete_map, kde, multiple, log_scale, ignore_zero)

def plot_corr(df: pd.DataFrame,
              column: str,
              n: int,
              method: str = 'pearson',
              size: Tuple[int, int] = (15, 8),
              rotation: int = 45,
              palette: str = 'RdYlGn',
              decimal: int = 2) -> None:
    """
    Plot the top `n` correlations of one variable against others in a DataFrame.

    This function generates a barplot that visually represents the correlations of
    a specified column with other numeric columns in a DataFrame. It displays both
    the strength (height of the bars) and the nature (color) of the correlations
    (positive or negative). The function computes correlations using the specified
    method and presents the strongest positive and negative correlations up to the
    number specified by `n`. Correlations are ordered from strongest to lowest,
    from the outside in.

    Use this to communicate the correlations of one particular variable (ex: target
    y) in relation to others with a very clean design. It's much easier to scan
    this correlation chart vs. trying to find the variable of interest in a
    heatmap. The fixed Y axis scales, and Red-Yellow-Green color palette, ensure
    the actual magnitudes of the positive or negative correlations are clear and
    not misinterpreted.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the variables for correlation analysis.
    column : str
        The name of the column to evaluate correlations against.
    n : int
        The number of correlations to display, split evenly between positive and
        negative correlations.
    method : str, optional
        The method of correlation calculation, as per `df.corr()` method options
        ('pearson', 'kendall', 'spearman'). Default is 'pearson'.
    size : Tuple[int, int], optional
        The size of the resulting plot. Default is (15, 8).
    rotation : int, optional
        The rotation angle for x-axis labels. Default is 45 degrees.
    palette : str, optional
        The colormap for representing correlation values. Default is 'RdYlGn'.
    decimal : int, optional
        The number of decimal places for rounding correlation values. Default is 2.

    Returns
    -------
    None
        Displays the barplot but does not return any value.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'A': np.random.rand(50),
    ...     'B': np.random.rand(50),
    ...     'C': np.random.rand(50)
    ... })
    >>> plot_corr(df, 'A', n=4)

    This will display a barplot of the top 2 positive and top 2 negative
    correlations of column 'A' with columns 'B' and 'C'.
    """
    # Calculate correlations
    corr = round(df.corr(method=method, numeric_only=True)[column].sort_values(), decimal)

    # Drop column from correlations (correlating with itself)
    corr = corr.drop(column)

    # Get the most negative and most positive correlations, sorted by absolute value
    most_negative = corr.sort_values().head(n // 2)
    most_positive = corr.sort_values().tail(n // 2)

    # Concatenate these two series and sort the final series by correlation value
    corr = pd.concat([most_negative, most_positive]).sort_values()
    corr.dropna(inplace=True)

    # Generate colors based on correlation values using a colormap
    cmap = plt.get_cmap(palette)
    colors = cmap((corr.values + 1) / 2)

    # Plot the chart
    plt.figure(figsize=size)
    plt.axhline(y=0, color='lightgrey', alpha=0.8, linestyle='-')
    bars = plt.bar(corr.index, corr.values, color=colors)

    # Add value labels to the end of each bar
    for bar in bars:
        yval = bar.get_height()
        if yval < 0:
            plt.text(bar.get_x() + bar.get_width() / 3.0, yval - 0.05, yval, va='top', fontsize=9)
        else:
            plt.text(bar.get_x() + bar.get_width() / 3.0, yval + 0.05, yval, va='bottom', fontsize=9)

    plt.title('Correlation with ' + column, fontsize=20)
    plt.ylabel('Correlation', fontsize=14)
    plt.xlabel('Other Variables', fontsize=14)
    plt.xticks(rotation=rotation)
    plt.ylim(-1, 1)
    plt.show()
