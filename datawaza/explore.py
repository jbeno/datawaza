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
extract the top correlations, plot correlations, plot 3D charts, and plot data
on a map of California.

Functions:
    - :func:`~datawaza.explore.get_corr` - Display the top `n` positive and negative correlations with a target variable in a DataFrame.
    - :func:`~datawaza.explore.get_outliers` - Detects and summarizes outliers for the specified numeric columns in a DataFrame, based on an IQR ratio.
    - :func:`~datawaza.explore.get_unique` - Print the unique values of all variables below a threshold `n`, including counts and percentages.
    - :func:`~datawaza.explore.plot_3d` - Create a 3D scatter plot using Plotly Express.
    - :func:`~datawaza.explore.plot_charts` - Display multiple bar plots and histograms for categorical and/or continuous variables in a DataFrame, with an option to dimension by the specified `hue`.
    - :func:`~datawaza.explore.plot_corr` - Plot the top `n` correlations of one variable against others in a DataFrame.
    - :func:`~datawaza.explore.plot_map_ca` - Plot longitude and latitude data on a geographic map of California.
    - :func:`~datawaza.explore.plot_scatt` - Create a scatter plot using Seaborn's scatterplot function.
    - :func:`~datawaza.explore.print_ascii_image` - Print ASCII representation of one or two PyTorch images.
"""

# Metadata
__author__ = "Jim Beno"
__email__ = "jim@jimbeno.net"
__version__ = "0.1.3"
__license__ = "GNU GPLv3"

# Standard library imports
import matplotlib.patheffects as pe

# Data manipulation and analysis
import numpy as np
import pandas as pd
from pandas import DataFrame
import geopandas as gpd

# Machine learning libraries
import torch

# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import plotly.express as px

# Statistical and geographic mapping
from scipy.stats import iqr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Package / file handling imports
import sys
if sys.version_info >= (3, 10):
    from importlib.resources import files, as_file
else:
    from importlib_resources import files

# Local Datawaza helper function imports
from datawaza.tools import thousands, dollars

# Typing imports
from typing import Optional, Union, Tuple, List, Dict, Any


# Functions
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
    >>> pd.set_option('display.max_columns', None)  # For test consistency
    >>> pd.set_option('display.width', None)  # For test consistency
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
      Column  Total Non-Null  Total Zero  Zero Percent  Outlier Count  Outlier Percent  Skewness  Kurtosis
    1      B              98           0           0.0              4             4.08      2.62     10.48
    0      A             100           2           2.0              1             1.00      0.01     -0.25
    2      C             100           0           0.0              1             1.00     -0.03      0.19

    Example 2: Create a dataframe that lists outlier statistics, excluding zeros
    and plot the box plots:

    >>> outlier_summary = get_outliers(df, num_columns, exclude_zeros=True,
    ...                                plot=True, width=14, height=3)
    >>> print(outlier_summary)
      Column  Total Non-Null  Total Zero  Zero Percent  Outlier Count  Outlier Percent  Skewness  Kurtosis
    0      B              98           0           0.0              4             4.08      2.62     10.48
    1      C             100           0           0.0              1             1.00     -0.03      0.19
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

    Example 3: Sort values by percent, plot charts, and show the continuous
    statistics for those over the 'n' threshold:

    >>> get_unique(df, n=3, sort='percent', plot=True, cont=True)
    <BLANKLINE>
    CATEGORICAL: Variables with unique values equal to or below: 3
    <BLANKLINE>
    Animal has 3 unique values:
    <BLANKLINE>
        'Dog'               3   50.0%
        'Cat'               2   33.33%
        'Mountain Lion'     1   16.67%
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    Sex has 3 unique values:
    <BLANKLINE>
        Male         3   50.0%
        Female       2   33.33%
        nan          1   16.67%
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    CONTINUOUS: Variables with unique values greater than: 3
    <BLANKLINE>
    Weight has 6 unique values:
    <BLANKLINE>
    Weight
    count     6.000000
    mean     27.050000
    std      29.292712
    min       6.500000
    25%       8.900000
    50%      17.400000
    75%      27.475000
    max      84.100000
    Name: Weight, dtype: float64
    <BLANKLINE>
    <BLANKLINE>
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


from typing import List, Dict, Optional

def plot_3d(
        df: pd.DataFrame,
        x: str,
        y: str,
        z: str,
        color: Optional[str] = None,
        color_discrete_sequence: Optional[List[str]] = None,
        color_discrete_map: Optional[Dict[str, str]] = None,
        color_continuous_scale: Optional[List[str]] = None,
        x_scale: str = 'linear',
        y_scale: str = 'linear',
        z_scale: str = 'linear',
        height: int = 600,
        width: int = 1000,
        font_size: int = 10
) -> None:
    """
    Create a 3D scatter plot using Plotly Express.

    This function generates an interactive 3D scatter plot using the
    Plotly Express library. It allows for customization of the `x`, `y`, and `z`
    axes, as well as color coding of the points based on the column specified
    for `color` (similar to the `hue` parameter in Seaborn). A `color_discrete_map`
    dictionary can be passed to map specific values of the `color` column to
    colors. Alternatively, you can just pass a `color_discrete_map` or
    `color_continuous_scale` depending on the type of values in the `color`
    column. Onlye 1 of these 3 coloring methods should be used at a time. The plot
    can also be displayed with either a linear or logarithmic scale on each axis
    by setting `x_scale`, `y_scale`, or `z_scale` from 'linear' to 'log'.

    Use this function to visualize and explore relationships between three
    variables in a dataset, with the option to color code the points based
    on a fourth variable. It is a great way to visualize the top 3 principal
    components, dimensioned by the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be plotted.
    x : str
        The column name to be used for the x-axis.
    y : str
        The column name to be used for the y-axis.
    z : str
        The column name to be used for the z-axis.
    color : str, optional
        The column name to be used for color coding the points. Default is
        None.
    color_discrete_sequence : List[str], optional
        Strings should define valid CSS-colors. When `color` is set and the values
        in the corresponding column are not numeric, values in that column are
        assigned colors by cycling through `color_discrete_sequence` in the order
        described in `category_orders`. Various color sequences are available in
        `plotly.express.colors.qualitative`. Default is None.
    color_discrete_map : Dict[str, str], optional
        String values should define valid CSS-colors. Used to assign specific
        colors values in the `color` column. Default is None.
    color_continuous_scale : List[str], optional
        Strings should define valid CSS-colors. This list is used to build a
        continuous color scale when the `color` column contains numeric data.
        Various color scales are available in `plotly.express.colors.sequential`,
        `plotly.express.colors.diverging`, and `plotly.express.colors.cyclical`.
        Default is None.
    x_scale : str, optional
        The scale type for the X axis. Use 'log' for logarithmic scale.
        Default is 'linear'.
    y_scale : str, optional
        The scale type for the Y axis. Use 'log' for logarithmic scale.
        Default is 'linear'.
    z_scale : str, optional
        The scale type for the Z axis. Use 'log' for logarithmic scale.
        Default is 'linear'.
    height : int, optional
        The height of the plot in pixels.
        Default is 600.
    width : int, optional
        The width of the plot in pixels.
        Default is 1000.
    font_size : int, optional
        The size of the font used in the plot.
        Default is 10.

    Returns
    -------
    None
        The function displays the interactive 3D scatter plot using Plotly
        Express.

    Examples
    --------
    Prepare the data for the examples:

    >>> df = pd.DataFrame({
    ...     'X': [1, 2, 3, 4, 5],
    ...     'Y': [2, 4, 6, 8, 10],
    ...     'Z': [3, 6, 9, 12, 15],
    ...     'Category': ['A', 'B', 'A', 'B', 'A'],
    ...     'Continuous': [10, 20, 30, 40, 50]
    ... })

    Example 1: Create a basic 3D scatter plot:

    >>> plot_3d(df, x='X', y='Y', z='Z')

    Example 2: Create a 3D scatter plot with default color coding, and log scale
    on the X axis:

    >>> plot_3d(df, x='X', y='Y', z='Z', color='Category', x_scale='log')

    Example 3: Create a 3D scatter plot with a discrete color palette:

    >>> plot_3d(df, x='X', y='Y', z='Z', color='Category',
    ...         color_discrete_sequence=px.colors.qualitative.Prism)

    Example 4: Create a 3D scatter plot with a continuous color palette:

    >>> plot_3d(df, x='X', y='Y', z='Z', color='Continuous',
    ...         color_continuous_scale=px.colors.sequential.Viridis)

    Example 5: Create a 3D scatter plot with a custom discrete color map,
    and adjust the height and width:

    >>> category_color_map = {'A': px.colors.qualitative.D3[0],
    ...                       'B': px.colors.qualitative.D3[1]}
    >>> plot_3d(df, x='X', y='Y', z='Z', color='Category',
    ...         color_discrete_map=category_color_map,
    ...         height=800, width=1200)
    """
    # Create the 3D scatter plot and set the title
    if color is not None:
        # Handle mappings of values in 'color' column to 'color_discrete_map'
        if color_discrete_map is not None:
            fig = px.scatter_3d(df, x=x, y=y, z=z,
                                color=color,
                                color_discrete_map=color_discrete_map,
                                height=height,
                                width=width)
        # Handle a discrete color palette
        elif color_discrete_sequence is not None:
            fig = px.scatter_3d(df, x=x, y=y, z=z,
                                color=color,
                                color_discrete_sequence=color_discrete_sequence,
                                height=height,
                                width=width)
        # Handle a continuous color palette
        elif color_continuous_scale is not None:
            fig = px.scatter_3d(df, x=x, y=y, z=z,
                                color=color,
                                color_continuous_scale=color_continuous_scale,
                                height=height,
                                width=width)
        # Handle no specified palette
        else:
            fig = px.scatter_3d(df, x=x, y=y, z=z,
                                color=color,
                                height=height,
                                width=width)
        title_text = f"{x}, {y}, {z} by {color}"
    # No color specified
    else:
        fig = px.scatter_3d(df, x=x, y=y, z=z,
                            height=height,
                            width=width)
        title_text = f"{x}, {y}, {z}"

    # Adjust the 3D perspective and plot styling
    fig.update_layout(title={'text': title_text, 'y': 0.9, 'x': 0.5,
                             'xanchor': 'center', 'yanchor': 'top'},
                      showlegend=True,
                      scene_camera=dict(up=dict(x=0, y=0, z=1),
                                        center=dict(x=0, y=0, z=-0.1),
                                        eye=dict(x=1.5, y=-1.4, z=0.5)),
                      margin=dict(l=0, r=0, b=0, t=0),
                      scene=dict(xaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title=x,
                                            title_font=dict(size=font_size),
                                            tickfont=dict(size=font_size),
                                            type=x_scale),
                                 yaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title=y,
                                            title_font=dict(size=font_size),
                                            tickfont=dict(size=font_size),
                                            type=y_scale),
                                 zaxis=dict(backgroundcolor='lightgrey',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title=z,
                                            title_font=dict(size=font_size),
                                            tickfont=dict(size=font_size),
                                            type=z_scale)))

    # Update the marker style
    fig.update_traces(marker=dict(size=3, opacity=1,
                                  line=dict(color='black', width=0.1)))

    # Display the plot
    fig.show()


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
                ignore_zero: bool = False
) -> None:
    """
    Display multiple bar plots and histograms for categorical and/or continuous
    variables in a DataFrame, with an option to dimension by the specified `hue`.

    This function allows you to plot a large number of distributions with one line
    of code. You choose which type of plots to create by setting `plot_type` to
    `cat`, `cont`, or `both`. Categorical variables are plotted with
    `sns.barplot` ordered by descending value counts for a clean appearance.
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
    ...     'Category C': np.random.choice(['K', 'L', 'M', 'N', 'O', 'P', 'Q',
    ...                                     'R', 'S', 'T', 'U', 'V', 'W', 'X'],
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
    lines, X axis in log scale, and check data types:

    >>> plot_charts(df, plot_type='cont', cont_cols=num_cols, hue='Target',
    ...             multiple='stack', kde=True, log_scale=True, dtype_check=True)

    Example 5: Take a sample of the data and plot only histograms dimensioned
    by hue (layer values), ignore zero values:

    >>> plot_charts(df, plot_type='cont', cont_cols=num_cols, hue='Target',
    ...             multiple='layer', sample_size=0.5, ignore_zero=True)

    Example 6: Normalize the values and plot categorical values:

    >>> plot_charts(df, plot_type='cat', cat_cols=cat_cols, hue='Target',
    ...             normalize=True)

    Example 7: Plot more than 10 categorical values, specify just one column name,
    and increase the figure size, with just one column:

    >>> plot_charts(df, plot_type='cat', cat_cols=['Category C'],
    ...             fig_width=12, subplot_height=7, ncols=1, rotation=90)

    """
    # Function to sample the data
    def get_sample(df, sample_size):
        # If sample_size is less than 1, treat it as a fraction
        if sample_size < 1:
            df_sample = df.sample(frac=sample_size, random_state=random_state)
        # If sample_size is greater than or equalt to 1, treat it as an exact sample count
        elif sample_size >= 1:
            # Convert floats to int
            sample_size = int(sample_size)
            df_sample = df.sample(n=sample_size, random_state=random_state)
        else:
            print(f'WARNING: Could not sample. get_sample() called, but sample_size = {sample_size}.')
            df_sample = df
        # Return a sampled dataframe
        return df_sample

    # Function to plot categorical variables as bar plots
    def plot_categorical(df, cols, ncols, fig_width, subplot_height, rotation, sample_size, hue, color_discrete_map, normalize):
        # If sample_size is defined, draw a sample
        df = get_sample(df, sample_size) if sample_size else df

        # Warn when we're asked to normalize but don't have hue
        if normalize and not hue:
            print(f"WARNING: Can't normalize without hue. normalize = {normalize}, hue = {hue}.")

        # Calculate the number of charts to plot
        nplots = len(cols)
        nrows = nplots//ncols
        if nplots % ncols:
            nrows += 1

        # Create the figure with subplots
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, nrows*subplot_height), constrained_layout=True)
        if isinstance(axs, np.ndarray):
            if len(axs.shape) > 1:
                axs = axs.ravel()
        else:
            axs = [axs]

        # Iterate through each categorical column to plot
        for i, col in enumerate(cols):
            if normalize and hue:
                # Normalize the counts, but only if hue is defined
                df_copy = df.copy()
                data = df_copy.groupby(col)[hue].value_counts(normalize=True).rename('proportion').reset_index()
                sns.barplot(data=data, x=col, y='proportion', hue=hue, palette=color_discrete_map, ax=axs[i])
                axs[i].set_ylabel('Proportion', fontsize=12)
            else:
                # Sort the DataFrame by the count of occurrences of each category
                if hue:
                    # Handle the case where the column is the same as the hue variable
                    if col == hue:
                        sorted_df = df[[hue]].groupby(hue).value_counts().reset_index()
                    else:
                        sorted_df = df[[col, hue]].groupby(hue).value_counts().reset_index()
                else:
                    sorted_df = df[col].value_counts().reset_index()
                    sorted_df.columns = [col, 'count']

                # Convert the column to string type to prevent numerical sorting
                sorted_df[col] = sorted_df[col].astype(str)

                # Generate a color palette
                if hue:
                    # Generate a color palette to match the number of hue values
                    num_hue_values = len(sorted_df[hue].value_counts())
                    if num_hue_values > 10:
                        colors = sns.color_palette("husl", num_hue_values)
                    elif color_discrete_map:
                        colors = color_discrete_map
                    else:
                        colors = sns.color_palette("tab10", num_hue_values)
                else:
                    # Generate a color palette to match the number of categories
                    num_categories = sorted_df.shape[0]
                    if num_categories > 10:
                        colors = sns.color_palette("husl", num_categories)
                    elif color_discrete_map:
                        colors = color_discrete_map
                    else:
                        colors = sns.color_palette("tab10", num_categories)

                # Plot the appropriate chart
                if hue == None:
                    sns.barplot(data=sorted_df, x=col, y='count', hue=col, legend=False, palette=colors, ax=axs[i])
                else:
                    sns.barplot(data=sorted_df, x=col, y='count', hue=hue, palette=colors, ax=axs[i])
                axs[i].set_ylabel('Count', fontsize=12)
                axs[i].yaxis.set_major_formatter(FuncFormatter(thousands))
            axs[i].set_xlabel(' ', fontsize=12)
            axs[i].set_title(col, fontsize=16, pad=10)
            axs[i].tick_params(axis='x', rotation=rotation)

        # Remove empty subplots
        for empty_subplot in axs[nplots:]:
            fig.delaxes(empty_subplot)

        # Show the plot
        plt.show()

    # Function to plot continuous variables as histograms
    def plot_continuous(df, cols, ncols, fig_width, subplot_height, sample_size, hue,
                        color_discrete_map, kde, multiple, log_scale, ignore_zero):
        # If sample_size is defined, draw a sample
        df = get_sample(df, sample_size) if sample_size else df

        # Calculate the number of charts to plot
        nplots = len(cols)
        nrows = nplots//ncols
        if nplots % ncols:
            nrows += 1

        # Create the figure with subplots
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, nrows * subplot_height), constrained_layout=True)
        if isinstance(axs, np.ndarray):
            if len(axs.shape) > 1:
                axs = axs.ravel()
        else:
            axs = [axs]

        # Iterate through each continuous column to plot
        for i, col in enumerate(cols):
            if hue is not None:
                if ignore_zero:
                    # Optionally ignore zero values in case they dwarf the rest of the chart
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
            axs[i].yaxis.set_major_formatter(FuncFormatter(thousands))
            axs[i].set_xlabel(' ', fontsize=12)
            axs[i].tick_params(axis='x', rotation=rotation)

        # Remove empty subplots
        for empty_subplot in axs[nplots:]:
            fig.delaxes(empty_subplot)

        # Show the plot
        plt.show()

    # Start by getting counts of unique values
    unique_count = df.nunique()

    # Plot the categorical columns, if selected
    if plot_type == 'cat' or plot_type == 'both':
        if cat_cols is None:
            # If columns not specified, find them based on less than or equal to 'n' unique values
            cat_cols = unique_count[unique_count <= n].index.tolist()
            # if hue in cat_cols:
            #     cat_cols.remove(hue)
        plot_categorical(df, cat_cols, ncols, fig_width, subplot_height, rotation, sample_size, hue, color_discrete_map, normalize)

    # Plot the continuous columns, if selected
    if plot_type == 'cont' or plot_type == 'both':
        if cont_cols is None:
            # If columns not specified, find them based on greater than 'n' unique values
            cont_cols = unique_count[unique_count > n].index.tolist()
        # If data type check is requested, filter out anything not int or float
        if dtype_check:
            cont_cols = [col for col in cont_cols if df[col].dtype in ['int64', 'float64']]
        plot_continuous(df, cont_cols, ncols, fig_width, subplot_height, sample_size, hue, color_discrete_map, kde, multiple, log_scale, ignore_zero)


def plot_scatt(
        df: pd.DataFrame,
        x: str,
        y: str,
        hue: Optional[str] = None,
        hue_order: Optional[List[str]] = None,
        size: Optional[Union[str, int]] = None,
        size_range: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        title_fontsize: int = 18,
        title_pad: int = 15,
        x_label: Optional[str] = None,
        x_format: Optional[str] = None,
        x_scale: Optional[str] = None,
        x_lim: Optional[Tuple[float, float]] = None,
        y_label: Optional[str] = None,
        y_format: Optional[str] = None,
        y_scale: Optional[str] = None,
        y_lim: Optional[Tuple[float, float]] = None,
        label_fontsize: int = 14,
        label_pad: int = 10,
        grid: bool = False,
        legend: bool = True,
        legend_title: Optional[str] = None,
        legend_loc: str = 'best',
        fig_size: Tuple[int, int] = (12, 6),
        decimal: int = 2,
        save: bool = False,
        **kwargs
) -> None:
    """
    Create a scatter plot using Seaborn's scatterplot function.

    This function generates a scatter plot using the Seaborn library. It allows
    for customization of the `x` and `y` axes, as well as the `hue` and `size`
    dimensions. The `hue` parameter is used to color the points based on a
    categorical column, while the `size` parameter is used to vary the size of
    the points based on a numerical column or a fixed value. You can also set
    the range of sizes with `size_range`, and the title of the plot with `title`.
    The `alpha` parameter controls the transparency of the points. You can also
    specify a color map with `color_map` to change the color scheme of the plot.
    The `fig_size` parameter allows you to set the size of the figure.

    Use this function to visualize relationships between two variables in a
    dataset, with the option to color and size the points based on additional
    variables. It is a great way to explore correlations between variables and
    identify patterns in the data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be plotted.
    x : str
        The column name to be used for the x-axis.
    y : str
        The column name to be used for the y-axis.
    hue : str, optional
        The column name to be used for color coding the points. Default is None.
    hue_order : List[str], optional
        The order of the hue variable levels. Default is None.
    size : str or int, optional
        The column name to be used for varying the size of the points, or a fixed
        size value for all points. Default is None.
    size_range : Tuple[int, int], optional
        The range of sizes for the points. Default is None.
    title : str, optional
        The title of the plot. Default is None.
    title_fontsize : int, optional
        The font size of the title. Default is 18.
    title_pad : int, optional
        The padding of the title. Default is 15.
    x_label : str, optional
        The label for the x-axis. Default is None.
    x_format : str, optional
        The format of the x-axis labels. Default is None.
    x_scale : str, optional
        The scale of the x-axis. Default is None.
    x_lim : Tuple[float, float], optional
        The limits of the x-axis. Default is None.
    y_label : str, optional
        The label for the y-axis. Default is None.
    y_format : str, optional
        The format of the y-axis labels. Default is None.
    y_scale : str, optional
        The scale of the y-axis. Default is None.
    y_lim : Tuple[float, float], optional
        The limits of the y-axis. Default is None.
    label_fontsize : int, optional
        The font size of the axis labels. Default is 14.
    label_pad : int, optional
        The padding of the axis labels. Default is 10.
    grid : bool, optional
        Whether to display a grid on the plot. Default is False.
    legend : bool, optional
        Whether to display a legend on the plot. Default is True.
    legend_title : str, optional
        The title of the legend. Default is None.
    legend_loc : str, optional
        The location of the legend. Default is 'best'.
    fig_size : Tuple[int, int], optional
        The size of the figure. Default is (12, 6).
    decimal : int, optional
        The number of decimal places to display on the axis labels. Default is 2.
    save : bool, optional
        Whether to save the plot as an image file. Default is False.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        `sns.scatterplot()` function. This allows for more flexibility and
        customization of the scatter plot.

    Returns
    -------
    None
        Displays the scatter plot using Seaborn.

    Examples
    --------

    Prepare some data for the examples:

    >>> np.random.seed(42)  # For reproducibility
    >>> x = np.linspace(0, 10, 50)
    >>> y = 2 * x**2 + 3 * x + 1 + np.random.normal(0, 100, 50)
    >>> categories = np.where(x < 5, 'A', 'B')
    >>> sizes = np.where(x < 5, 30, 60)
    >>> df = pd.DataFrame({
    ...     'X': x,
    ...     'Y': y,
    ...     'Category': categories,
    ...     'Size': sizes
    ... })
    >>> color_palette = {'A': 'red', 'B': 'green'}

    Example 1: Create a basic scatter plot with a fixed size for all points:

    >>> plot_scatt(df, x='X', y='Y', size=50, alpha=0.7)

    Example 2: Create a scatter plot with color coding based on a category and
    varying point sizes based on a numerical column:

    >>> plot_scatt(df, x='X', y='Y', hue='Category', size='Size',
    ...            size_range=(20, 100))

    Example 3: Create a scatter plot with a custom title, color map, axis labels,
    and legend:

    >>> plot_scatt(df, x='X', y='Y', title='Polynomial Trend', hue='Size',
    ...            palette='viridis', x_label='X Axis', y_label='Y Axis',
    ...            legend=True)

    Example 4: Create a scatter plot with custom x and y limits and axis formats:
    >>> plot_scatt(df, x='X', y='Y', x_lim=(0, 8), y_lim=(0, 400),
    ...            x_format='small_number', y_format='{:,.2f}')

    Example 5: Create a scatter plot with varying point sizes based on a numerical
    column and save it to a file:

    >>> plot_scatt(df, x='X', y='Y', size='Size', size_range=(20, 100),
    ...            title='Scatter Plot', save=True)

    Example 6: Create a scatter plot with varying marker styles based on a
    categorical column:

    >>> plot_scatt(df, x='X', y='Y', hue='Category', style='Category')

    Example 7: Create a scatter plot with a custom marker style and color palette:

    >>> plot_scatt(df, x='X', y='Y', marker='D', hue='Category',
    ...            palette=color_palette)
    """
    # Check if required parameters are provided
    if df is None:
        raise ValueError("The 'df' parameter is required.")
    if x is None or x not in df.columns:
        raise ValueError(f"The 'x' parameter is required and must be a valid "
                         f"column name in the DataFrame. Got: {x}")
    if y is None or y not in df.columns:
        raise ValueError(f"The 'y' parameter is required and must be a valid "
                         f"column name in the DataFrame. Got: {y}")

    # Check if optional parameters are valid column names
    if hue is not None and hue not in df.columns:
        raise ValueError(f"The 'hue' parameter must be a valid column name in "
                         f"the DataFrame. Got: {hue}")
    if isinstance(size, str) and size not in df.columns:
        raise ValueError(f"The 'size' parameter must be a valid column name in "
                         f"the DataFrame or an integer. Got: {size}")

    # Check if title is provided when save is True
    if save and title is None:
        raise ValueError("The 'title' parameter is required when 'save' is set "
                         "to True.")

    # Disable legend if there are no `hue` or `size` to differentiate
    if hue is None and size is None:
        legend = False

    # Function to get string formatting for the axis values
    def get_format(kind, decimal):
        if kind in ['thousands', 'large_number', 'large_numbers']:
            return FuncFormatter(lambda x, pos: f"{x:,.0f}")
        elif kind == 'large_dollars':
            return FuncFormatter(lambda x, pos: f"${x:,.0f}")
        elif kind in ['small_number', 'small_numbers']:
            return FuncFormatter(lambda x, pos: f"{x:.{decimal}f}")
        elif kind == 'small_dollars':
            return FuncFormatter(lambda x, pos: f"${x:.{decimal}f}")
        elif kind == 'percent':
            return FuncFormatter(lambda x, pos: f"{x:.0f}%")
        else:
            # Assume 'kind' is a custom format string, create a formatter using it
            return FuncFormatter(lambda x, pos: kind.format(x))

    # Create the figure at the specified size
    plt.figure(figsize=fig_size)

    # Format the grid
    if grid:
        plt.grid(visible=True, color='lightgrey', alpha=0.5, linestyle='-',
                 linewidth=0.5)

    # Set the title with the specified font size and padding
    if title:
        plt.title(title, fontsize=title_fontsize, pad=title_pad)

    # Create the scatter plot with the specified parameters and additional keyword arguments
    sns.scatterplot(data=df, x=x, y=y, hue=hue, hue_order=hue_order,
                    size=size, sizes=size_range, **kwargs)

    # Set the labels for the x axis with the specified font size and padding
    if x_label:
        plt.xlabel(xlabel=x_label, fontsize=label_fontsize, labelpad=label_pad)
    else:
        plt.xlabel(xlabel=x, fontsize=label_fontsize, labelpad=label_pad)

    # Set the labels for the y axis with the specified font size and padding
    if y_label:
        plt.ylabel(ylabel=y_label, fontsize=label_fontsize, labelpad=label_pad)
    else:
        plt.ylabel(ylabel=y, fontsize=label_fontsize, labelpad=label_pad)

    # Format the x axis if specified with named format or custom format
    if x_format:
        plt.gca().xaxis.set_major_formatter(get_format(x_format, decimal))

    # Format the y axis if specified with named format or custom format
    if y_format:
        plt.gca().yaxis.set_major_formatter(get_format(y_format, decimal))

    # Set the scale of the x axis if specified
    if x_scale:
        plt.xscale(x_scale)

    # Set the scale of the y axis if specified
    if y_scale:
        plt.yscale(y_scale)

    # Set the limits of the x axis if specified
    if x_lim:
        plt.xlim(x_lim)

    # Set the limits of the y axis if specified
    if y_lim:
        plt.ylim(y_lim)

    # Legend handling
    if legend:
        plt.legend(title=legend_title, loc=legend_loc)

    # Clean up the layout
    plt.tight_layout()

    # Save the plot if specified
    if save:
        filename = title.lower().replace(' ', '_') + '.png'
        plt.savefig(filename)

    # Show the plot
    plt.show()

def plot_corr(df: pd.DataFrame,
              column: str,
              n: int,
              method: str = 'pearson',
              size: Tuple[int, int] = (15, 8),
              rotation: int = 45,
              palette: str = 'RdYlGn',
              decimal: int = 2
) -> None:
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


def plot_map_ca(
        df: pd.DataFrame,
        lon: str = 'Longitude',
        lat: str = 'Latitude',
        hue: Optional[str] = None,
        size: Optional[str] = None,
        size_range: Tuple[int, int] = (50, 200),
        title: str = 'Geographic Chart',
        dot_size: Optional[int] = None,
        alpha: float = 0.8,
        color_map: Optional[str] = None,
        fig_size: Tuple[int, int] = (12, 12)
) -> None:
    """
    Plot longitude and latitude data on a geographic map of California.

    This function creates a geographic map of California using Cartopy and
    overlays data points from a DataFrame. The map includes major cities, county
    boundaries, and geographic terrain features. Specify the columns in the
    dataframe that map to the longitude (`lon`) and the latitude (`lat`). Then
    specify an optional `hue` column to see changes in this variable by color,
    and/or a `size` column to see changes in this varible by dot size. So two
    variables can be visualized at once.

    A few parameters can be customized, such as the range of the dot sizes
    (`size_range`) if you're using `size`. You can also just use `dot_size` to
    specify a fixed size for all the dots on the map. The `alpha` transparency
    can be adjusted, to make sure you at least have a chance of seeing dots of
    a different color that may be covered up by the top-most layer. You can also
    customize the `color_map` for the `hue` parameter.

    Use this function to visualize geospatial data related to
    California on a clean map.

    Note: This function requires a few libraries to be installed: Cartopy,
    Geopandas, and Matplotlib (pyplot and patheffects). In addition, it uses
    the 2018 Census Bureau's 5-meter county map files, which can be found here:
    https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_5m.zip

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be plotted.
    lon : str, optional
        Column name in `df` representing the longitude coordinates.
        Default is 'Longitude'.
    lat : str, optional
        Column name in `df` representing the latitude coordinates.
        Default is 'Latitude'.
    hue : str, optional
        Column name in `df` for color-coding the points. Default is
        None.
    size : str, optional
        Column name in `df` to scale the size of points. Default is
        None.
    size_range : Tuple[int, int], optional
        Range of sizes if the `size` parameter is used. Default is
        (50, 200).
    title : str, optional
        Title of the plot. Default is 'Geographic Chart'.
    dot_size : int, optional
        Size of all dots if you want them to be uniform. Default is
        None.
    alpha : float, optional
        Transparency of the points. Default is 0.8.
    color_map : colors.Colormap, optional
        Colormap to be used if `hue` is specified. Default is None.
    fig_size : Tuple[int, int], optional
        Size of the figure. Default is (12, 12).

    Returns
    -------
    None

    Examples
    --------
    >>> import pandas as pd
    >>> data = {
    ...     'longitude': [-122.23, -122.22, -122.24, -122.25, -122.25],
    ...     'latitude': [37.88, 37.86, 37.85, 37.85, 37.85],
    ...     'housing_median_age': [41.0, 21.0, 52.0, 52.0, 52.0],
    ...     'total_rooms': [880.0, 7099.0, 1467.0, 1274.0, 1627.0],
    ...     'total_bedrooms': [129.0, 1106.0, 190.0, 235.0, 280.0],
    ...     'population': [322.0, 2401.0, 496.0, 558.0, 565.0],
    ...     'households': [126.0, 1138.0, 177.0, 219.0, 259.0],
    ...     'median_income': [8.3252, 8.3014, 7.2574, 5.6431, 3.8462],
    ...     'median_house_value': [452600.0, 358500.0, 352100.0, 341300.0, 342200.0],
    ...     'ocean_proximity': ['NEAR BAY', 'NEAR BAY', 'NEAR BAY', 'NEAR BAY', 'NEAR BAY']
    ... }
    >>> df = pd.DataFrame(data)
    >>> plot_map_ca(df, lon='longitude', lat='latitude',
    ...             hue='ocean_proximity', size='median_house_value',
    ...             size_range=(100, 500), alpha=0.6,
    ...             title='California Housing Data')
    """
    # Define the locations of major cities
    large_ca_cities = {'Name': ['Fresno', 'Los Angeles', 'Sacramento', 'San Diego', 'San Francisco', 'San Jose'],
                       'Latitude': [36.746842, 34.052233, 38.581572, 32.715328, 37.774931, 37.339386],
                       'Longitude': [-119.772586, -118.243686, -121.494400, -117.157256, -122.419417, -121.894956],
                       'County': ['Fresno', 'Los Angeles', 'Sacramento', 'San Diego', 'San Francisco', 'Santa Clara']}
    df_large_cities = pd.DataFrame(large_ca_cities)

    # Create a figure that utilizes Cartopy
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-125, -114, 32, 42])

    # Add geographic details
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='lightgrey', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)

    # Add county boundaries
    # Determine the path to the data file within the package

    if sys.version_info >= (3, 10):
        # Python 3.10 or higher
        with as_file(files('datawaza.data') / 'cb_2018_us_county_5m.shp') as data_path:
            counties = gpd.read_file(data_path)
    else:
        # Python 3.9 or lower
        data_path = files('datawaza.data') / 'cb_2018_us_county_5m.shp'
        counties = gpd.read_file(str(data_path))

    counties_ca = counties[counties['STATEFP'] == '06']
    counties_ca = counties_ca.to_crs("EPSG:4326")
    for geometry in counties_ca['geometry']:
        ax.add_geometries([geometry], crs=ccrs.PlateCarree(), edgecolor='grey', alpha=0.3, facecolor='none')

    # Draw the scatterplot of data
    if dot_size:
        ax.scatter(df[lon], df[lat], s=dot_size, cmap=color_map, alpha=alpha, transform=ccrs.PlateCarree())
    else:
        sns.scatterplot(data=df, x=lon, y=lat, hue=hue, size=size, alpha=alpha, ax=ax, palette=color_map, sizes=size_range)

    # Add cities
    ax.scatter(df_large_cities['Longitude'], df_large_cities['Latitude'], transform=ccrs.PlateCarree(), edgecolor='black')
    for x, y, label in zip(df_large_cities['Longitude'], df_large_cities['Latitude'], df_large_cities['Name']):
        text = ax.text(x + 0.05, y + 0.05, label, transform=ccrs.PlateCarree(), fontsize=12, ha='left', fontname='Arial')
        text.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])

    # Finish up the chart
    ax.set_title(title, fontsize=18, pad=15)
    ax.set_xlabel('Longitude', fontsize=14, labelpad=15)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.gridlines(draw_labels=True, color='lightgrey', alpha=0.5)
    plt.show()

def print_ascii_image(
        image1: torch.Tensor,
        image2: Optional[torch.Tensor] = None,
        values: str = 'binary',
        channels: int = 1,
        channel_weights: List[float] = [0.2989, 0.5870, 0.1140],
        only_first_channel: bool = False,
        height: int = 28,
        width: int = 28,
        orientation: str = 'vertical',
        add_space: bool = False,
        binary_threshold: float = 0.5,
        merge_channels: bool = False,
        merge_method: str = 'weighted',
        mode: str = 'scale',
        binary_char: str = '█',
        ascii_chars: str = "@%#*+=-:. "
) -> None:
    """
    Print ASCII representation of one or two PyTorch images.

    This function takes one or two PyTorch tensors representing images and prints
    their ASCII representation. It supports various options to customize the
    output, such as the number of channels, orientation, and value representation
    (binary or continuous). The function can also merge multiple channels into a
    single grayscale representation using either the mean or a weighted sum of the
    channels.

    The tensor data should be concatenated into one long vector. For example, for
    a single-channel image of size 28x28, the tensor should have a size of [784].
    For a three-channel image of size 32x32, the tensor should have a size of
    [3072]. Please process the tensors before passing them to this function.

    Use this function when you want to visualize PyTorch images in the console
    using ASCII characters, perhaps for debugging during your modeling workflow.
    For example, you can compare two images side by side (a source image from the
    x dataset, and the corresponding generated image x_hat).

    Parameters
    ----------
    image1 : torch.Tensor
        The first image tensor to be printed as ASCII. It should have 1 dimension.
    image2 : Optional[torch.Tensor], optional
        The second image tensor to be printed as ASCII, by default None. It should
        have 1 dimension.
    values : str, optional
        The value representation mode, either 'binary' or 'continuous',
        by default 'binary'
    channels : int, optional
        The number of channels in the input images, by default 1. Channels will be
        extracted from the input tensor based on this value, they should not be
        a separate dimension.
    channel_weights : List[float], optional
        The weights for each channel when merging channels using the 'weighted'
        method, by default [0.2989, 0.5870, 0.1140] (RGB to grayscale conversion
        weights)
    only_first_channel : bool, optional
        Whether to print only the first channel, by default False
    height : int, optional
        The height of the ASCII representation, by default 32
    width : int, optional
        The width of the ASCII representation, by default 32
    orientation : str, optional
        The orientation of the printed ASCII, either 'vertical' or 'horizontal',
        by default 'vertical'
    add_space : bool, optional
        Whether to add spaces between ASCII characters, by default False
    binary_threshold : float, optional
        The threshold for binarizing pixel values, by default 0.5
    merge_channels : bool, optional
        Whether to merge multiple channels into a single grayscale representation,
        by default False
    merge_method : str, optional
        The method for merging channels, either 'mean' or 'weighted', by default
        'weighted'
    mode : str, optional
        The preprocessing mode for pixel values, either 'binary', 'scale', 'clamp',
        or 'pass', by default 'scale'
    binary_char : str, optional
        The character to use for representing binary values above the threshold,
        by default '█'
    ascii_chars : str, optional
        The ASCII characters to use for the image, by default "@%#*+=-:. ". The
        characters should be ordered from dark to light, with the last character
        representing the lightest value.

    Returns
    -------
    None
        The function does not return anything, but prints the ASCII representation
        of the input images.

    Examples
    --------
    Prepare example image tensors:

    >>> if sys.version_info >= (3, 10):
    ...     with as_file(files('datawaza.data') / 'mnist_28x28_ch1_x0.pt') as path:
    ...         image1 = torch.load(path)
    ...     with as_file(files('datawaza.data') / 'svhn_32x32_ch3_x0.pt') as path:
    ...         image2 = torch.load(path)
    ...     with as_file(files('datawaza.data') / 'svhn_32x32_ch3_xhat0.pt') as path:
    ...         image3 = torch.load(path)
    ... else:
    ...     image1 = torch.load(files('datawaza.data') / 'mnist_28x28_ch1_x0.pt')
    ...     image2 = torch.load(files('datawaza.data') / 'svhn_32x32_ch3_x0.pt')
    ...     image3 = torch.load(files('datawaza.data') / 'svhn_32x32_ch3_xhat0.pt')

    Example 1: Print a single binary image with a space between characters:

    >>> print_ascii_image(image1, add_space=True)  #doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
                                            █
                          █ █             █ █
                          █ █             █ █ █
                          █ █               █
                        █ █ █             █
                      █ █ █             █ █
                        █               █ █
                        █               █
                        █ █           █ █ █
                        █ █           █ █
                    █ █ █ █ █         █ █
                    █ █ █ █ █ █ █   █ █
                      █ █ █ █ █ █ █ █ █     █ █
                      █ █ █ █ █ █ █ █ █ █ █ █ █
                          █ █   █ █         █
                                █ █
                                █ █
                              █ █
                              █ █ █
                            █ █ █
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>

    Example 2: Print a single continuous image with 3 channels:

    >>> print_ascii_image(image2, values='continuous', channels=3, width=32,
    ...                   height=32, orientation='horizontal')
    ++++*#%@@%#*++======+*#%%@@@@%%%   ===++*#%@%#+========+*#%%@@%%%%%   ----=+*###*+=-------==**#%######
    ---=+*%@@#*=----::-==+*%%@@@@@%%   :-:-=*#%%#+=-:::::--=+*%%@@@@%%%   ::::-=*##*+-::....::-=+*#%%#####
    ::-=+*%@%#+=-----::--=+#%@@@@@%%   :::-+*#%%*+-:::-::::-=+#%@@@@%%%   ...:-=*##+=::.::....:-=*##%#####
    :--=*#%%%*=----==-:::-+*%@@@%%%%   ::-=*##%#+=:::-==-:::-+*%@@@%%%#   ..:-=+*#*+-:.::--:...:=+*####***
    :-=+#%%%%+=::-=++=:::-=*%%@%%%##   :--+#%%%#+-:::=++=:::-=*%%@%%%##   .::=+**#*=:..:-==-...:-+*###****
    :-=*%@@%#=-::-+**=:::-=*%%@%%###   :-=*#%%%*=:.:-=**=:.:-=*%@@%%###   .:-+*##*+-:..:=++=...:-+*###****
    --+#%@@%*=:::-+##+::::=*%@@%####   :-+#%@@%*-:.:-+**+::::=*%@@%####   ::=+###*+:...-=**=:...-=*##*****
    :-+#@@@%#=:::=+##+::::=*%%%%####   :-+#%@@%*-:::-+##+::::=*%%%%####   .:=*#%##+:...-+**=:...-=*##**+*+
    :-+#@@@%*=:::=*##+:.:-=*%%%%####   :-+#%@%#*-:.:-+##+:::-=*%%%%####   .:=*#%#*+:..:-+##=...:-+*##**+*+
    :-+#%@@%#=-:-=+##=:::-+*%%%%%###   :-+#%@%#*=:::-+##=:::-+#%%%%%###   .:=*###*+-:.:-=**-...:=+*##*****
    :-*#%@%%#+=--=+#*=:::-+#%@%%%%%#   :-+#%@%#*=-::-+#*=::--+#%@@%%%##   :-=*###*+=:::-=*+-..::=+###*****
    :-*#%@%%#+=-:-=*+=::-=+#%@%%%%%%   :-*#%%%#*+-::-=++=::-=*#%@@%%%%%   :-+*###*+=-:.:-==-..::=*###*#***
    :=*#@@@@%*=-::-=--:--=*#%%%%%%%%   :=*#%@%%#+=-::-=-----=*#%@%%%%%%   :-+*#%##*+-:..:-::..:-=**##*#***
    :-+#%@@@%*=-::-------=*#%%%%%%%%   :-+#%%%%#+=-.:-------=*#%%%%%%%%   .:+*#####+-: .::::::::=+****#**#
    -=++*%%%#+--:-===+=---=*%%%%%%%%   -=++*#%%*=-:::=+++=---+*%%%%%%%#   :-==+*##*=::.:-----:.:-+********
    --==+*##*-:::=+***=-:-=*#%%%%%%%   --==+*#*+-:::=+***=-:-=*#%%%%%##   :---=+**=:...-=++=-:..:=********
    :--==***+-::-+*##*=-::=*#%%%%%%%   :---=+*+=:.:-+*##*=-::=*#%%%%%##   ::---+++=:..:=+**+-:..:=+*******
    :---=+*+=::-=*#%#*=-:-=*#%%%%%%%   ::--=+++-:.:-+#%#*=-:-=*#%%%%%##   .:::-=+=-:.:-+***+-:..:=+*******
    :::-=++=-::-=*#%#*=-:-=*#%%%%%%%   :::--=+=-:::=*###*=-:-=*#%%%%%%#   ...:-==-::.:-+***=-:..-=********
    :::-=++=-::-=*#%#*=-:-=*#%%%%%%%   .::-=++=-:.:=*###+--:-=*#%%%%%%%   ...:-==-:..:-+*#*=::.:-=********
    :::-=++=-:::=*%%#+-::-=*#%%%%%%%   .::-=++-::.:=*###+-::-=*#%%%%%%%   ...:-==-:..:-+*#*=:..:-+********
    :::-=+=-::::=*##*=-:-=+*#%%%%%%%   ..:-=+=-::::=*##*=:::-+*#%%%%%%%   ...:-=-:...:-+**+-:..:=+********
    ::-=++=-:::-=###*=---=+#%%%%%%%%   :::=++=-::::=*##+-::-=+#%%%%%%%%   ..:-==-:...:-***+-:.:-=+*#*****#
    --=+**+-:::-=*##*=:--=*#%@%%%%%%   :-=+**+-:.:-=***+-::-=*#%%%%%%%%   .:-=++=:...:=+**=-..:-+*###**#*#
    -=++*#+-:::-=+**+----=*%%@@%%%%%   -==+*#+-:::-=+*+=-::-=*#%@@%%%%#   :-==++=:...:-=++-::::-+*######**
    ==+*##*=::::-=+=----=+*%%@@%%%%%   -=+*##*=:.::-===-:--=+*%%@%%%%%#   -==+**+-:...:-=-::::-=+*######*#
    -=+*##*=-:::-==-:-===*#%@@%%%%%%   -=+*##*=:...:-=-::-==+#%%%%%%%%%   :-=+**+-:...:--:.::--=*#########
    --=+##*+-::::-=---=++*#%@@%%%%%%   --=*##*=-::.:-----==+*#%%%%%%%##   :-=+**+=:....:-:::--=+*####*####
    :--+*##+=--::-====++*##%%%%%%%%#   :-=+*#*+=-:::----=++*##%%%%##%##   .:-=+*+=-::..:--:-==++*####**##*
    ::-+*%##+==--==++***##%%%%%%%%%%   ::-+*##*+-----=+++**##%%%%%#%%##   .:-=+**+=::::--===+++*#####*####
    --=*#%%%#+++++*#####%%%@@%%%%@%%   --=*#%%#*+++++**#####%%%%%%#%%%%   ::-+*##*+=====++*****######*#%##
    +++*#%%%##****#######%%%%%%#%%%#   +++*#%%%#*****########%%%%######   ===+*###*+++++********####**####
    <BLANKLINE>

    Example 3: Print a single continuous image but merge the channels into one:

    >>> print_ascii_image(image2, values='continuous', channels=3, width=32,
    ...                   height=32, merge_channels=True, add_space=True)
    + + + + * # % @ @ % # * + + = = = = = = + * # % % @ @ @ @ % % %
    - - - = + * % @ @ # * = - - - - : : - = = + * % % @ @ @ @ @ % %
    : : - = + * % @ % # + = - - - - - : : - - = + # % @ @ @ @ @ % %
    : - - = * # % % % * = - - - - = = - : : : - + * % @ @ @ % % % %
    : - = + # % % % % + = : : - = + + = : : : - = * % % @ % % % # #
    : - = * % @ @ % # = - : : - + * * = : : : - = * % % @ % % # # #
    - - + # % @ @ % * = : : : - + # # + : : : : = * % @ @ % # # # #
    : - + # @ @ @ % # = : : : = + # # + : : : : = * % % % % # # # #
    : - + # @ @ @ % * = : : : = * # # + : . : - = * % % % % # # # #
    : - + # % @ @ % # = - : - = + # # = : : : - + * % % % % % # # #
    : - * # % @ % % # + = - - = + # * = : : : - + # % @ % % % % % #
    : - * # % @ % % # + = - : - = * + = : : - = + # % @ % % % % % %
    : = * # @ @ @ @ % * = - : : - = - - : - - = * # % % % % % % % %
    : - + # % @ @ @ % * = - : : - - - - - - - = * # % % % % % % % %
    - = + + * % % % # + - - : - = = = + = - - - = * % % % % % % % %
    - - = = + * # # * - : : : = + * * * = - : - = * # % % % % % % %
    : - - = = * * * + - : : - + * # # * = - : : = * # % % % % % % %
    : - - - = + * + = : : - = * # % # * = - : - = * # % % % % % % %
    : : : - = + + = - : : - = * # % # * = - : - = * # % % % % % % %
    : : : - = + + = - : : - = * # % # * = - : - = * # % % % % % % %
    : : : - = + + = - : : : = * % % # + - : : - = * # % % % % % % %
    : : : - = + = - : : : : = * # # * = - : - = + * # % % % % % % %
    : : - = + + = - : : : - = # # # * = - - - = + # % % % % % % % %
    - - = + * * + - : : : - = * # # * = : - - = * # % @ % % % % % %
    - = + + * # + - : : : - = + * * + - - - - = * % % @ @ % % % % %
    = = + * # # * = : : : : - = + = - - - - = + * % % @ @ % % % % %
    - = + * # # * = - : : : - = = - : - = = = * # % @ @ % % % % % %
    - - = + # # * + - : : : : - = - - - = + + * # % @ @ % % % % % %
    : - - + * # # + = - - : : - = = = = + + * # # % % % % % % % % #
    : : - + * % # # + = = - - = = + + * * * # # % % % % % % % % % %
    - - = * # % % % # + + + + + * # # # # # % % % @ @ % % % % @ % %
    + + + * # % % % # # * * * * # # # # # # # % % % % % % # % % % #
    <BLANKLINE>

    Example 4: Print 2 images side by side with continuous values merged channels:

    >>> print_ascii_image(image2, image3, values='continuous', channels=3,
    ...                   width=32, height=32, merge_channels=True,
    ...                   orientation='vertical', add_space=False)
    ++++*#%@@%#*++======+*#%%@@@@%%%   #%##%##%###%######*###%%###%%%#%
    ---=+*%@@#*=----::-==+*%%@@@@@%%   ***##%%%%#####******##%%%%%%%%##
    ::-=+*%@%#+=-----::--=+#%@@@@@%%   ***###%#%%#**+++++++**##%%%%%%%%
    :--=*#%%%*=----==-:::-+*%@@@%%%%   +****##%#**+=======+++*%%%@@%%%%
    :-=+#%%%%+=::-=++=:::-=*%%@%%%##   +++*######+=+====-===**#%@%@%%%#
    :-=*%@@%#=-::-+**=:::-=*%%@%%###   +++*#####*==-===-=--=++%@@@%@%%%
    --+#%@@%*=:::-+##+::::=*%@@%####   =++*####*+--=-===+=--=*#@@@%%%#%
    :-+#@@@%#=:::=+##+::::=*%%%%####   ==++*#%#++==-==+++=--=*#%@@@%%%%
    :-+#@@@%*=:::=*##+:.:-=*%%%%####   ++++**##*+=-==*+*+=--=*#%@@@@%#%
    :-+#%@@%#=-:-=+##=:::-+*%%%%%###   ++++*#%#*+--=++**+=-=+*#%@@@@%%%
    :-*#%@%%#+=--=+#*=:::-+#%@%%%%%#   ==+*+###+==-=+****=-==#%%@%@@%%%
    :-*#%@%%#+=-:-=*+=::-=+#%@%%%%%%   *++**###*====+****====*#@@@@@@%%
    :=*#@@@@%*=-::-=--:--=*#%%%%%%%%   +=++*##**+===+***+====+#%@@@@%%#
    :-+#%@@@%*=-::-------=*#%%%%%%%%   ++++*###*+====+=*+=-==+#%@@@@%%#
    -=++*%%%#+--:-===+=---=*%%%%%%%%   ==+=*###*+=====+++=--=**%@@@@@%%
    --==+*##*-:::=+***=-:-=*#%%%%%%%   ==++####*++===+++==--=+#%@@@@%%%
    :--==***+-::-+*##*=-::=*#%%%%%%%   ===+*###*++--=+++==--=+#%@@@@%%%
    :---=+*+=::-=*#%#*=-:-=*#%%%%%%%   ==++**#**=+=-=++*+=---+*%%@@@@%#
    :::-=++=-::-=*#%#*=-:-=*#%%%%%%%   =+=+*###*+-=+++*#+==--+*%%@@@%%%
    :::-=++=-::-=*#%#*=-:-=*#%%%%%%%   +++++*##*+=--=+***+---+#%%@@%%%%
    :::-=++=-:::=*%%#+-::-=*#%%%%%%%   ++=++*#**+===+*#**+===+#%@@@@%#%
    :::-=+=-::::=*##*=-:-=+*#%%%%%%%   ++++**##*+=-=++##*+---=*%@@%%%##
    ::-=++=-:::-=###*=---=+#%%%%%%%%   +++++*#**=--=+*##*==--++%@@@%%%#
    --=+**+-:::-=*##*=:--=*#%@%%%%%%   +++++*#**=---=**#*+---+*#%@@@%##
    -=++*#+-:::-=+**+----=*%%@@%%%%%   +==++*#**=---=+***==:=+##@@@%%%#
    ==+*##*=::::-=+=----=+*%%@@%%%%%   +=++**#**+---=+*++==-=*#%@@@%#%#
    -=+*##*=-:::-==-:-===*#%@@%%%%%%   =+++*###*+==--=++==-=+*#%%%@@%#*
    --=+##*+-::::-=---=++*#%@@%%%%%%   ++++#*###++--==+=====**%%@%@%@%#
    :--+*##+=--::-====++*##%%%%%%%%#   ++**###%#*++==+==+=++#%%@@%@@@##
    ::-+*%##+==--==++***##%%%%%%%%%%   ***#*#%%##**++++=++**#%%%@@%%%#%
    --=*#%%%#+++++*#####%%%@@%%%%@%%   *####%%%###**+++**#*##%%%%%%%%%%
    +++*#%%%##****#######%%%%%%#%%%#   ##*###%%%%%#*##**#####%%%@%%%##%
    <BLANKLINE>
    """
    # Validate input parameters
    if not isinstance(image1, torch.Tensor):
        raise ValueError("image1 must be a torch.Tensor")
    if image2 is not None and not isinstance(image2, torch.Tensor):
        raise ValueError("image2 must be a torch.Tensor")
    if values not in ['binary', 'continuous']:
        raise ValueError("values must be 'binary' or 'continuous'")
    if orientation not in ['vertical', 'horizontal']:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    if merge_method not in ['mean', 'weighted']:
        raise ValueError("merge_method must be 'mean' or 'weighted'")
    if mode not in ['binary', 'scale', 'clamp', 'pass']:
        raise ValueError("mode must be 'binary', 'scale', 'clamp', or 'pass'")

    # Helper function to convert a pixel value to an ASCII character
    def pixel_to_ascii(pixel):
        index = int(pixel * (len(ascii_chars) - 1))
        return ascii_chars[index]

    # Helper function to process an input image tensor
    def process_image(image, channels, merge_channels, mode):
        # Apply preprocessing mode
        if mode == 'binary':
            image = torch.where(image > binary_threshold, torch.ones_like(image), torch.zeros_like(image))
        elif mode == 'scale':
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
            else:
                image = torch.zeros_like(image)
        elif mode == 'clamp':
            image = torch.clamp(image, 0, 1)

        # Use only the first channel if specified
        if only_first_channel:
            channels = 1

        channel_data = []
        pixel_per_channel = height * width

        # Merge channels if specified
        if merge_channels:
            if channels != len(channel_weights):
                raise ValueError(f"Number of weights and channels do not match. channels: {channels}, channel_weights: {channel_weights}.")
            if merge_method == 'mean':
                image = image.mean(0)
            elif merge_method == 'weighted':
                weights = torch.tensor(channel_weights).view(channels, 1)
                image = (image * weights).sum(0)
            channels = 1

        # Process each channel
        for channel in range(channels):
            channel_rows = []
            start_index = channel * pixel_per_channel
            for i in range(height):
                row_start = start_index + i * width
                row_end = row_start + width
                row_pixels = image[row_start:row_end]
                if values == 'binary' and add_space:
                    row = ' '.join(' ' if pixel <= binary_threshold else binary_char for pixel in row_pixels)
                elif values == 'binary':
                    row = ''.join(' ' if pixel <= binary_threshold else binary_char for pixel in row_pixels)
                elif values == 'continuous' and add_space:
                    row = ' '.join(pixel_to_ascii(pixel) for pixel in row_pixels)
                elif values == 'continuous':
                    row = ''.join(pixel_to_ascii(pixel) for pixel in row_pixels)
                channel_rows.append(row)
            channel_data.append(channel_rows)

        return channel_data

    # Process input images
    data1 = process_image(image1, channels, merge_channels, mode)
    data2 = process_image(image2, channels, merge_channels, mode) if image2 is not None else None

    # Print the ASCII representation based on the specified orientation
    if orientation == 'vertical':
        for ch in range(len(data1)):
            if data2:
                combined_rows = zip(data1[ch], data2[ch])
                for row1, row2 in combined_rows:
                    print(f"{row1}   {row2}")
            else:
                for row in data1[ch]:
                    print(row)
            print()
    elif orientation == 'horizontal':
        for row in zip(*data1):
            print('   '.join(row))
        print()
        if data2:
            for row in zip(*data2):
                print('   '.join(row))
            print()