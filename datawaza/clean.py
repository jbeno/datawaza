# clean.py – Clean module of Datawaza
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
This module provides tools to clean data in preparation for modeling.
It contains functions to convert data types, convert unites of measurement,
convert time values, reduce multicollinearity, and split out the outliers from
a dataset.

Functions:
    - :func:`~datawaza.clean.convert_data_values` - Convert mixed data values (ex: GB, MB, KB) to a common unit of measurement.
    - :func:`~datawaza.clean.convert_dtypes` - Convert specified columns in a DataFrame to the desired data type.
    - :func:`~datawaza.clean.convert_time_values` - Convert time values in specified columns of a DataFrame to a target format.
    - :func:`~datawaza.clean.reduce_multicollinearity` - Reduce multicollinearity in a DataFrame by removing highly correlated features.
    - :func:`~datawaza.clean.split_outliers` - Split a DataFrame into two based on the presence of outliers.
"""

# Metadata
__author__ = "Jim Beno"
__email__ = "jim@jimbeno.net"
__version__ = "0.1.3"
__license__ = "GNU GPLv3"

# Standard library imports
import re

# Data manipulation and analysis
import numpy as np
import pandas as pd
from pandas import DataFrame

# Typing imports
from typing import Optional, Union, Tuple, List, Dict, Any


# Functions
def convert_data_values(
        df: pd.DataFrame,
        cols: List[str],
        target_unit: str = 'MB',
        show_results: bool = False,
        inplace: bool = False,
        decimal: int = 4,
        conversion_dict: Optional[Dict[str, int]] = None
) -> Optional[pd.DataFrame]:
    """
    Convert mixed data values (ex: GB, MB, KB) to a common unit of measurement.

    This function converts values in the specified columns of the input DataFrame
    to the desired target unit. If `inplace` is set to True, the conversion is done
    in place, modifying the original DataFrame. If `inplace` is False (default), a
    new DataFrame with the converted values is returned. The string suffix is
    dropped and the column is converted to a float. It handles inconsistent suffix
    strings, with or without spaces after the numbers (ex: '10GB', '10 Gb'). A
    variety of spelling options are supported (ex: 'GB', 'Gigabytes'), but you
    can pass a custom dictionary as `conversion_dict` if desired. To display a
    summary of the changes made, set `show_results` to True.

    Use this to clean up messy data that has a variety of units of measurement
    appended as text strings to the numeric values. The result will be columns
    with a common unit of measurement as floats (with no text suffixes).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the columns to be converted.
    cols : List[str]
        List of column names in `df` to apply the conversion.
    target_unit : str, optional
        The target unit for the conversion. Default is 'MB'.
        Possible values are: 'B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB'.
    show_results : bool, optional
        If set to True, prints the before-and-after conversion values for each cell.
        Default is False.
    inplace : bool, optional
        If set to True, the conversion is done in place, modifying the original DataFrame.
        If False (default), a new DataFrame with the converted values is returned.
    decimal : int, optional
        The number of decimal places to show when `show_results` is set to True.
        Default is 4.
    conversion_dict : Dict[str, int], optional
        A custom dictionary mapping units to their corresponding powers of 2.
        If provided, it will override the default `unit_conversion` dictionary.
        Default is None.

    Returns
    -------
    pd.DataFrame or None
        If `inplace` is False (default), returns a new DataFrame with the specified
        columns converted to the target unit. If `inplace` is True, returns None as
        the original DataFrame is modified in place.

    Examples
    --------
    Prepare some sloppy data for the examples:

    >>> df = pd.DataFrame({
    ...     'A': ['5 gb', '500 kb', '2tb'],
    ...     'B': ['3GB', '0.5 MB', '200KB'],
    ...     'C': ['10 Gigabytes', '250 kilo', '1 PB']
    ... })
    >>> cols = ['A', 'B', 'C']

    Example 1: Convert values in specified columns to GB and assign to a new df:

    >>> df_converted = convert_data_values(df, cols, target_unit='GB')
    >>> df_converted
                 A         B             C
    0     5.000000  3.000000  1.000000e+01
    1     0.000477  0.000488  2.384186e-04
    2  2048.000000  0.000191  1.048576e+06

    Example 2: Convert data values to MB in place, modifying the existing df,
    and show a summary of the changes:

    >>> convert_data_values(df, cols, target_unit='MB', inplace=True,
    ... show_results=True, decimal=8)
    Original: 5 gb -> Converted: 5120.00000000 MB
    Original: 500 kb -> Converted: 0.48828125 MB
    Original: 2tb -> Converted: 2097152.00000000 MB
    Original: 3GB -> Converted: 3072.00000000 MB
    Original: 0.5 MB -> Converted: 0.50000000 MB
    Original: 200KB -> Converted: 0.19531250 MB
    Original: 10 Gigabytes -> Converted: 10240.00000000 MB
    Original: 250 kilo -> Converted: 0.24414062 MB
    Original: 1 PB -> Converted: 1073741824.00000000 MB
    >>> df
                  A            B             C
    0  5.120000e+03  3072.000000  1.024000e+04
    1  4.882812e-01     0.500000  2.441406e-01
    2  2.097152e+06     0.195312  1.073742e+09
    """
    # Default conversion factors based on powers of 2
    default_unit_conversion = {
        'B': 0,
        'BYTE': 0,
        'BYTES': 0,
        'KB': 10,
        'KILOBYTE': 10,
        'KILOBYTES': 10,
        'KILO': 10,
        'MB': 20,
        'MEGABYTE': 20,
        'MEGABYTES': 20,
        'MEGA': 20,
        'GB': 30,
        'GIGABYTE': 30,
        'GIGABYTES': 30,
        'GIGA': 30,
        'TB': 40,
        'TERABYTE': 40,
        'TERABYTES': 40,
        'TERA': 40,
        'PB': 50,
        'PETABYTE': 50,
        'PETABYTES': 50,
        'PETA': 50,
        'EB': 60,
        'EXABYTE': 60,
        'EXABYTES': 60,
        'EXA': 60
    }

    # Use the provided conversion_dict if passed as a parameter
    unit_conversion = conversion_dict if conversion_dict is not None else default_unit_conversion

    # Convert target_unit to uppercase for case-insensitive comparison
    target_unit = target_unit.upper()

    # Copy dataframe if not modifying in place
    if not inplace:
        df = df.copy()

    # Iterate through the columns
    for col in cols:
        # Function to convert a data value
        def convert_value(value):
            if pd.isna(value):  # Handle NaNs
                if show_results:
                    print(f"Original: NaN -> Converted: NaN")
                return np.nan

            # RegEx handles both with spaces or without, ex: '10 GB' and '10GB'
            match = re.match(r'(\d+\.?\d*)\s?([A-Za-z]+)', value, re.IGNORECASE)

            if not match:
                raise ValueError(f"Invalid format for value: {value}")

            # Assign the 2 groups from the RegEx matches
            number, unit = match.groups()
            unit = unit.upper()  # Convert the unit to uppercase
            number = float(number)  # Convert the values to floats

            # Convert the number to bytes
            bytes_value = number * (2 ** unit_conversion[unit])

            # Convert the bytes value to the target unit
            converted_value = bytes_value / (2 ** unit_conversion[target_unit])

            # Print the before/after values if show_results is True
            if show_results:
                print(f"Original: {value} -> Converted: {converted_value:.{decimal}f} {target_unit}")

            return converted_value

        # Apply the conversion function to values in each column, convert to float
        df[col] = df[col].apply(convert_value).astype(float)

    # Return a dataframe only if not modifying in place
    if inplace:
        return None
    else:
        return df


def convert_dtypes(
        df: pd.DataFrame,
        cols: List[str],
        target_dtype: Union[type, str],
        show_results: bool = False,
        inplace: bool = True
) -> Optional[pd.DataFrame]:
    """
    Convert specified columns in a DataFrame to the desired data type.

    This function converts the data type of the specified columns in the input
    DataFrame to the desired target data type. It supports both base Python data
    types (e.g., int, float, str) and Pandas-specific data types (e.g., 'int64',
    'float64', 'object', 'bool', 'datetime64', 'timedelta[ns]', 'category'). If
    `inplace` is set to True (default), the conversion is done in place, modifying
    the original DataFrame. If `inplace` is False, a new DataFrame with the
    converted columns is returned. If `show_results` is set to True, it will print
    the results of each successful conversion and any error messages for columns
    that could not be converted.

    Use this function when you need to convert the data types of specific columns
    in a DataFrame to a consistent target data type, especially when dealing with
    multiple columns at once and identifying columns that require further data
    cleaning.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the columns to be converted.
    cols : List[str]
        List of column names in `df` to be converted.
    target_dtype : type or str
        The desired data type for the conversion. Can be a base Python data type
        (e.g., int, float, str) or a string representation of a Pandas data type
        (e.g., 'int64', 'float64', 'object', 'bool', 'datetime64', 'timedelta[ns]',
        'category').
    show_results : bool, optional
        If True, will print the results of each successful conversion and any error
        messages for columns that could not be converted. Default is False.
    inplace : bool, optional
        If True (default), the conversion is done in place, modifying the original
        DataFrame. If False, a new DataFrame with the converted columns is
        returned.

    Returns
    -------
    pd.DataFrame or None
        If `inplace` is False, returns a new DataFrame with the specified columns
        converted to the target data type. If `inplace` is True, returns None as
        the original DataFrame is modified in place.

    Examples
    --------
    Prepare data for examples:

    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': ['0', '23.4 MB', '3.71 GB'],
    ...     'C': ['4.5', '3.7', '12.15'],
    ...     'D': [True, False, True],
    ...     'E': ['Yes', 'No', ''],
    ...     'F': ['Low', 'Medium', 'High']
    ... })
    >>> num_columns = ['A', 'B', 'C']
    >>> cat_columns = ['D', 'E', 'F']
    >>> df.dtypes
    A     int64
    B    object
    C    object
    D      bool
    E    object
    F    object
    dtype: object

    Example 1: Convert columns 'A', 'B', and 'C' to float and show the results:

    >>> convert_dtypes(df, num_columns, 'float')
    Error converting column: B (Current dtype: object). Error message: could not convert string to float: '23.4 MB'
    >>> df.dtypes
    A    float64
    B     object
    C    float64
    D       bool
    E     object
    F     object
    dtype: object

    Example 2: Convert columns 'D', 'E', and 'F' to category, show the results, and
    return a new DataFrame:

    >>> new_df = convert_dtypes(df, cat_columns, 'category', inplace=False,
    ... show_results=True)
    Successfully converted column 'D' from bool to category.
    Successfully converted column 'E' from object to category.
    Successfully converted column 'F' from object to category.
    >>> new_df.dtypes
    A     float64
    B      object
    C     float64
    D    category
    E    category
    F    category
    dtype: object
    """
    if not inplace:
        df = df.copy()

    for col in cols:
        try:
            current_dtype = df[col].dtype
            if isinstance(target_dtype, str):
                df[col] = df[col].astype(target_dtype)
            else:
                df[col] = df[col].astype(target_dtype)
            if show_results:
                print(f"Successfully converted column '{col}' from {current_dtype} to {df[col].dtype}.")
        except (ValueError, TypeError) as e:
            print(f"Error converting column: {col} (Current dtype: {current_dtype}). Error message: {e}")

    if not inplace:
        return df


def convert_time_values(
        df: pd.DataFrame,
        cols: List[str],
        target_format: str = '%Y-%m-%d %H:%M:%S',
        target_dtype: Union[str, type, np.dtype] = 'datetime64[ns]',
        show_results: bool = False,
        inplace: bool = False,
        zero_to_nan: bool = False,
        pattern_list: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Convert time values in columns of a DataFrame to a target format and data type.

    This function converts time values in the specified columns of the input
    DataFrame to the desired target format and data type. If `inplace` is set to
    True, the conversion is done in place, modifying the original DataFrame. If
    `inplace` is False (default), a new DataFrame with the converted values is
    returned.

    The function can handle time values in various formats, including:
    1. Excel serial format (e.g., '45161.23458')
    2. String format (e.g., 'YYYY-MM-DD')
    3. UNIX epoch in milliseconds (e.g., '1640304000000.0')

    If your format is not supported, you can define `pattern_list` as a list of
    custom datetime patterns.

    If `zero_to_nan` is set to True, values of '0', '0.0', '0.00', 0, 0.0, or 0.00
    will be replaced with NaN. Otherwise, zero values will be detected as a
    Unix Epoch format with value 1970-01-01 00:00:00.

    You can use the default `target_format` of '%Y-%m-%d %H:%M:%S', or specify
    a different format. The `target_dtype` parameter allows you to specify the
    desired data type for the converted columns. When working with formatted date
    strings, the applicable data types are 'str', 'object', or 'datetime64[ns]'.

    - If `target_dtype` is set to 'str' or 'object', the converted data will be
      stored as strings in the specified `target_format`. Missing values will be
      represented as 'NaN'.

    - If `target_dtype` is set to 'datetime64[ns]', the converted data will be
      stored as pandas datetime objects. The `target_format` has no effect in
      this scenario. Missing values will be represented as 'NaT'.

    To display a summary of the changes made, set `show_results` to True.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the columns to be converted.
    cols : List[str]
        List of column names in `df` to apply the conversion.
    target_format : str, optional
        The desired datetime format for the conversion. Uses format codes such as:
        %Y: 4-digit year, %m: Month as zero-padded decimal, %d: Day of the month,
        %H: Hour (24-hour clock), %M: Minute, %S: Second.
        Default format is '%Y-%m-%d %H:%M:%S'.
    target_dtype : Union[str, type, np.dtype], optional
        The desired data type for the converted columns. Applicable data types are
        'str', 'object', or 'datetime64[ns]'. Default is 'datetime64[ns]'.
    show_results : bool, optional
        If set to True, prints the before-and-after conversion values for each
        cell. Default is False.
    inplace : bool, optional
        If set to True, the conversion is done in place, modifying the original
        DataFrame. If False (default), a new DataFrame with the converted values
        is returned.
    zero_to_nan : bool, optional
        If set to True, values of '0', '0.0', '0.00', 0, 0.0, or 0.00 will be
        replaced with NaN or NaT, depending on `target_dtype`. Default is False.
    pattern_list : List[str], optional
        A list of custom datetime patterns to override the default patterns.
        If provided, it will be used instead of the default patterns.
        Default is None.

    Returns
    -------
    pd.DataFrame or None
        If `inplace` is False (default), returns a new DataFrame with the specified
        columns converted to the target format and data type. If `inplace` is True,
        returns None as the original DataFrame is modified in place.

    Examples
    --------
    Prepare some sloppy data for the examples:

    >>> df = pd.DataFrame({
    ...     'A': ['45161.23458', '2019-02-09', '1640304000000.0'],
    ...     'B': ['2022-01-01', '45000.5', '1577836800000.0'],
    ...     'C': ['0', '45161.23458', '2019-02-09']
    ... })
    >>> cols = ['A', 'B', 'C']

    Example 1: Convert time values in specified columns to the default format
    and data type:

    >>> df_converted = convert_time_values(df, cols)
    >>> df_converted
                        A                   B                   C
    0 2023-08-25 05:37:47 2022-01-01 00:00:00 1970-01-01 00:00:00
    1 2019-02-09 00:00:00 2023-03-17 12:00:00 2023-08-25 05:37:47
    2 2021-12-24 00:00:00 2020-01-01 00:00:00 2019-02-09 00:00:00
    >>> df_converted.dtypes
    A    datetime64[ns]
    B    datetime64[ns]
    C    datetime64[ns]
    dtype: object

    Example 2: Convert time values in specified columns to the default format
    and data type, showing a summary of changes:

    >>> df_converted = convert_time_values(df, cols, show_results=True)
    Original: 45161.23458 (Excel Serial) -> Converted: 2023-08-25 05:37:47
    Original: 2019-02-09 (Standard Datetime String) -> Converted: 2019-02-09 00:00:00
    Original: 1640304000000.0 (UNIX Epoch in milliseconds) -> Converted: 2021-12-24 00:00:00
    Original: 2022-01-01 (Standard Datetime String) -> Converted: 2022-01-01 00:00:00
    Original: 45000.5 (Excel Serial) -> Converted: 2023-03-17 12:00:00
    Original: 1577836800000.0 (UNIX Epoch in milliseconds) -> Converted: 2020-01-01 00:00:00
    Original: 0 (UNIX Epoch) -> Converted: 1970-01-01 00:00:00
    Original: 45161.23458 (Excel Serial) -> Converted: 2023-08-25 05:37:47
    Original: 2019-02-09 (Standard Datetime String) -> Converted: 2019-02-09 00:00:00
    >>> df_converted
                        A                   B                   C
    0 2023-08-25 05:37:47 2022-01-01 00:00:00 1970-01-01 00:00:00
    1 2019-02-09 00:00:00 2023-03-17 12:00:00 2023-08-25 05:37:47
    2 2021-12-24 00:00:00 2020-01-01 00:00:00 2019-02-09 00:00:00

    Example 3: Convert time values to a custom format and data type in place,
    replacing zeros with NaN, and showing a summary of changes:

    >>> convert_time_values(df, cols, target_format='%d/%m/%Y', target_dtype='str',
    ... inplace=True, show_results=True, zero_to_nan=True)
    Original: 45161.23458 (Excel Serial) -> Converted: 25/08/2023
    Original: 2019-02-09 (Standard Datetime String) -> Converted: 09/02/2019
    Original: 1640304000000.0 (UNIX Epoch in milliseconds) -> Converted: 24/12/2021
    Original: 2022-01-01 (Standard Datetime String) -> Converted: 01/01/2022
    Original: 45000.5 (Excel Serial) -> Converted: 17/03/2023
    Original: 1577836800000.0 (UNIX Epoch in milliseconds) -> Converted: 01/01/2020
    Original: 0 (Zero) -> Converted: NaN
    Original: 45161.23458 (Excel Serial) -> Converted: 25/08/2023
    Original: 2019-02-09 (Standard Datetime String) -> Converted: 09/02/2019
    >>> df
                A           B           C
    0  25/08/2023  01/01/2022         NaN
    1  09/02/2019  17/03/2023  25/08/2023
    2  24/12/2021  01/01/2020  09/02/2019
    >>> df.dtypes
    A    object
    B    object
    C    object
    dtype: object
    """
    # Default datetime patterns
    default_patterns = [
        r"^\d{4}-\d{2}-\d{2}$",                   # YYYY-MM-DD
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", # YYYY-MM-DD HH:MM:SS
        r"^\d{2}/\d{2}/\d{4}$",                   # MM/DD/YYYY
    ]

    # Use the provided pattern_list if passed as a parameter
    patterns = pattern_list if pattern_list is not None else default_patterns

    # Set zero patterns to look for
    zero_patterns = ['0', '0.0', '0.00', 0, 0.0, 0.00]

    # Copy dataframe if not modifying in place
    if not inplace:
        df = df.copy()

    # Iterate through the columns
    for col in cols:
        # Function to convert a time value
        def convert_value(value):
            # If value is NaN or NaT, return the appropriate missing value based on target_dtype
            if pd.isna(value) or value is pd.NaT:
                if target_dtype == 'datetime64[ns]':
                    converted_value = pd.NaT
                else:
                    converted_value = 'NaN'

                if show_results:
                    print(f"Original: {value} -> Converted: {converted_value}")
                return converted_value

            # If zero_to_nan is True and value matches zero patterns, return the appropriate missing value based on target_dtype
            if zero_to_nan and str(value) in zero_patterns:
                if target_dtype == 'datetime64[ns]':
                    converted_value = pd.NaT
                else:
                    converted_value = 'NaN'

                if show_results:
                    print(f"Original: {value} (Zero) -> Converted: {converted_value}")
                return converted_value

            detected_format = None

            try:
                # Convert Excel Serial to datetime
                if isinstance(value, (float, int)) or (isinstance(value, str) and "." in value):
                    float_value = float(value)
                    if 40000 < float_value < 50000:  # Typical range for recent Excel serials
                        datetime_val = pd.Timestamp('1900-01-01') + pd.to_timedelta(float_value, 'D')
                        detected_format = "Excel Serial"
                    else:
                        datetime_val = pd.to_datetime(float(value), unit='ms')
                        detected_format = "UNIX Epoch in milliseconds"
                # If value is '0', convert it to Unix epoch
                elif str(value) in zero_patterns:
                    datetime_val = pd.to_datetime(0, unit='s')
                    detected_format = "UNIX Epoch"
                # Assume it's already in a recognizable format (like 'YYYY-MM-DD')
                elif isinstance(value, str):
                    if any(re.match(pattern, value) for pattern in patterns):
                        datetime_val = pd.to_datetime(value)
                        detected_format = "Standard Datetime String"
                    else:
                        raise ValueError(f"Unrecognized format for value: {value}")

                # Format conversion using the specified target_format
                formatted_datetime = datetime_val.strftime(target_format)

            except Exception as e:
                raise ValueError(f"Error converting value: {value}. Additional info: {e}")

            if show_results:
                print(f"Original: {value} ({detected_format}) -> Converted: {formatted_datetime}")

            return formatted_datetime

        # Apply the conversion function to values in each column
        df[col] = df[col].apply(convert_value)

    # Convert the modified columns to the specified data type
    for col in cols:
        df[col] = df[col].astype(target_dtype)

    # Return a dataframe only if not modifying in place
    if inplace:
        return None
    else:
        return df


def reduce_multicollinearity(
        df: DataFrame,
        target_col: str,
        corr_threshold: float = 0.9,
        consider_nan: bool = False,
        consider_zero: bool = False,
        diff_threshold: float = 0.1,
        decimal: int = 2
) -> DataFrame:
    """
    Reduce multicollinearity in a DataFrame by removing highly correlated features.

    This function iteratively evaluates pairs of features in a DataFrame based on
    their correlation to each other and to a specified target column. If two
    features are highly correlated (above `corr_threshold`), the one with the lower
    correlation to the target column is removed. The number of NaN and/or zero
    values can also be considered (prefering removal of features with more) by
    setting `consider_nan` or `consider_zero` to True. The threshold for
    significant differences (`diff_threshold`) can also be adjusted. Sometimes it
    might appear as if the correlations are the same, but it says one is greater.
    Adjust `decimal` to a larger number to see more precision in the correlation.

    Use this function to remove redundant features, and reduce a large feature set
    to a smaller one that contains the features most correlated with the target.
    This should improve the model's ability to learn from the dataset, improve
    performance, and increase interpretability of results.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to evaluate. It should have features and the target column.
    target_col : str
        The name of the target column against which feature correlations are
        evaluated.
    corr_threshold : float, optional
        The correlation threshold above which a pair of features is considered
        highly correlated. Default is 0.9.
    consider_nan : bool, optional
        If True, considers the number of NaN values in the decision process.
        Default is False.
    consider_zero : bool, optional
        If True, considers the number of zero values in the decision process.
        Default is False.
    diff_threshold : float, optional
        The threshold for considering the difference in NaN/zero counts as
        significant. Default is 0.1.
    decimal : int, optional
        The number of decimals to round to when displaying output. Default is 2.

    Returns
    -------
    DataFrame
        A modified DataFrame with reduced multicollinearity, containing only the
        features that were kept after evaluation.

    Examples
    --------
    Prepare the data for the examples:

    >>> np.random.seed(0)  # For reproducibility
    >>> A = np.random.rand(100)
    >>> B = A + np.random.normal(0, 0.01, 100)  # B is highly correlated with A
    >>> C = np.random.rand(100)
    >>> D = B + np.random.rand(100) # D is highly correlated with B
    >>> Target = A * 0.1 + C * 0.5 + D * 0.8 + np.random.normal(0, 0.1, 100)
    >>> df = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'Target': Target})

    Example 1: Reduce multicollinearity in a DataFrame with correlated features
    and a target column:

    >>> reduced_df = reduce_multicollinearity(df, 'Target', decimal=4)
    Evaluating pair: 'B' and 'A' (1.0) - 5 kept features
     - Correlation with target: 0.6509, 0.6468
     - Keeping 'B' (higher correlation, lower or equal count)
    <BLANKLINE>

    Example 2: Reduce multicollinearity with a lower correlation threshold and
    considering NaN and Zero values:

    >>> reduced_df = reduce_multicollinearity(df, 'Target', corr_threshold=0.6,
    ... consider_zero=True, consider_nan=True, diff_threshold=0.2)
    Evaluating pair: 'D' and 'A' (0.66) - 5 kept features
     - Correlation with target: 0.86, 0.65
     - NaN/0 counts: 0, 0
     - Keeping 'D' (higher correlation, lower or equal count)
    <BLANKLINE>
    Evaluating pair: 'D' and 'B' (0.66) - 4 kept features
     - Correlation with target: 0.86, 0.65
     - NaN/0 counts: 0, 0
     - Keeping 'D' (higher correlation, lower or equal count)
    <BLANKLINE>
    """
    # Initial set up
    original_features = set(df.columns)
    kept_features = original_features.copy()

    # Initialize consider_text
    if consider_nan and consider_zero:
        consider_text = 'NaN/0'
    elif consider_nan:
        consider_text = 'NaN'
    elif consider_zero:
        consider_text = '0'
    else:
        consider_text = 'None'

    while True:
        kept_features_list = list(kept_features)
        corr_matrix = df[kept_features_list].corr()
        target_correlations = corr_matrix[target_col].abs().sort_values(ascending=False)

        changes_made = False
        evaluated_pairs = set()

        for feature in target_correlations.index.drop(target_col):
            if feature not in kept_features:
                continue  # Skip if feature already removed

            feature_removed = False  # Initialize flag to track if 'feature' is removed

            high_corr_features = corr_matrix[feature][corr_matrix[feature].abs() >= corr_threshold].index.drop(feature)
            high_corr_features = high_corr_features.difference(evaluated_pairs)
            high_corr_features = sorted(high_corr_features)

            for other_feature in high_corr_features:
                if other_feature not in kept_features or other_feature == target_col:
                    continue  # Skip if other feature already removed or is target

                evaluated_pairs.add((feature, other_feature))  # Mark this pair as evaluated
                evaluated_pairs.add((other_feature, feature))  # Experimental: does this help?

                feature_pair_corr = corr_matrix.at[feature, other_feature]
                corr_with_target_feature = target_correlations[feature]
                corr_with_target_other = target_correlations[other_feature]
                nan_count_feature = df[feature].isna().sum() if consider_nan else 0
                zero_count_feature = (df[feature] == 0).sum() if consider_zero else 0
                nan_count_other = df[other_feature].isna().sum() if consider_nan else 0
                zero_count_other = (df[other_feature] == 0).sum() if consider_zero else 0
                total_count_feature = nan_count_feature + zero_count_feature
                total_count_other = nan_count_other + zero_count_other

                # Evaluate if significant difference in counts
                max_count = max(total_count_feature, total_count_other)
                if max_count == 0:  # Avoid division by zero
                    sig_diff = False
                else:
                    relative_difference = abs(total_count_feature - total_count_other) / max_count
                    sig_diff = relative_difference > diff_threshold

                # Print decision process
                print(f"Evaluating pair: '{feature}' and '{other_feature}' ({round(feature_pair_corr, 2)}) - {len(kept_features)} kept features")
                print(f" - Correlation with target: {corr_with_target_feature:.{decimal}f}, {corr_with_target_other:.{decimal}f}")
                if consider_nan or consider_zero:
                    print(f" - {consider_text} counts: {total_count_feature}, {total_count_other}")

                # Logic to decide which feature to keep
                if corr_with_target_feature > corr_with_target_other and total_count_feature <= total_count_other:
                    print(f" - Keeping '{feature}' (higher correlation, lower or equal count)\n")
                    if other_feature in kept_features:
                        kept_features.remove(other_feature)
                        changes_made = True
                    else:
                        print(f"{other_feature} already removed\n")
                elif corr_with_target_feature > corr_with_target_other and not sig_diff:
                    print(f" - Keeping '{feature}' (higher correlation, no significant diff: {relative_difference:.{decimal}f} <= {diff_threshold:.{decimal}f})\n")
                    if other_feature in kept_features:
                        kept_features.remove(other_feature)
                        changes_made = True
                    else:
                        print(f"{other_feature} already removed\n")
                elif corr_with_target_feature > corr_with_target_other and sig_diff:
                    print(f" - Keeping '{other_feature}' (higher correlation, significant diff: {relative_difference:.{decimal}f} > {diff_threshold:.{decimal}f})\n")
                    if feature in kept_features:
                        kept_features.remove(feature)
                        changes_made = True
                        feature_removed = True
                    else:
                        print(f"{feature} already removed\n")
                elif corr_with_target_feature == corr_with_target_other and total_count_feature <= total_count_other:
                    print(f" - Keeping '{feature}' (equal correlation, lower or equal count)\n")
                    if other_feature in kept_features:
                        kept_features.remove(other_feature)
                        changes_made = True
                    else:
                        print(f"{other_feature} already removed\n")
                elif corr_with_target_feature == corr_with_target_other and total_count_feature > total_count_other:
                    print(f" - Keeping '{other_feature}' (equal correlation, higher count)\n")
                    if feature in kept_features:
                        kept_features.remove(feature)
                        changes_made = True
                        feature_removed = True
                    else:
                        print(f"{feature} already removed\n")
                elif corr_with_target_feature < corr_with_target_other and total_count_feature == total_count_other:
                    print(f" - Keeping '{other_feature}' (lower correlation, equal count)\n")
                    if feature in kept_features:
                        kept_features.remove(feature)
                        changes_made = True
                        feature_removed = True
                    else:
                        print(f"{feature} already removed\n")
                elif corr_with_target_feature < corr_with_target_other and not sig_diff:
                    print(f" - Keeping '{other_feature}' (lower correlation, no significant diff: {relative_difference:.{decimal}f} <= {diff_threshold:.{decimal}f})\n")
                    if feature in kept_features:
                        kept_features.remove(feature)
                        changes_made = True
                        feature_removed = True
                    else:
                        print(f"{feature} already removed\n")
                else:
                    print(f" - Keeping '{feature}' (otherwise, go with first feature)\n")
                    if other_feature in kept_features:
                        kept_features.remove(other_feature)
                        changes_made = True
                    else:
                        print(f"{other_feature} already removed\n")

                if feature_removed:
                    break  # Break out of the first for loop if the feature was removed

        if not changes_made:
            # No more features to evaluate, end the loop
            break

    # Prepare final dataframe with kept features
    to_drop = original_features - kept_features
    reduced_df = df.drop(columns=to_drop)
    return reduced_df


def split_outliers(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        iqr_multiplier: float = 1.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into two based on the presence of outliers.

    This function identifies outliers in the specified columns of the
    input DataFrame using the Interquartile Range (IQR) method. It then
    splits the DataFrame into two: one containing rows without outliers
    and another containing only the rows with outliers.

    Use this function when you need to separate outliers from the main
    data for further analysis or processing.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be split.
    columns : List[str], optional
        List of column names to consider for outlier detection. If None,
        all columns in the DataFrame will be considered. Default is None.
    iqr_multiplier : float, optional
        The multiplier for the IQR range to determine outliers. Default
        is 1.5.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames: The first DataFrame (df_no_outliers)
        contains rows without outliers. The second DataFrame (df_outliers)
        contains only the rows with outliers.

    Examples
    --------
    Prepare the data for the examples:

    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5, 100],
    ...     'B': [10, 20, 30, 40, 50, 600],
    ...     'C': [-30, 4, 3, 2, 1, 3]
    ... })

    Example 1: Split outliers considering all columns:

    >>> df_no_outliers, df_outliers = split_outliers(df)
    >>> df_no_outliers
       A   B  C
    1  2  20  4
    2  3  30  3
    3  4  40  2
    4  5  50  1
    >>> df_outliers
         A    B   C
    0    1   10 -30
    5  100  600   3

    Example 2: Split outliers considering specific columns:

    >>> df_no_outliers, df_outliers = split_outliers(df, columns=['A', 'B'])
    >>> df_no_outliers
       A   B   C
    0  1  10 -30
    1  2  20   4
    2  3  30   3
    3  4  40   2
    4  5  50   1
    >>> df_outliers
         A    B  C
    5  100  600  3
    """
    # If columns parameter is not provided, use all columns in the DataFrame
    if columns is None:
        columns = df.columns

    # Create an initial mask with all False values (meaning no outliers)
    outlier_mask = pd.Series(False, index=df.index)

    # For each specified column, update the outlier mask to mark outliers
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Update mask for outliers in the current column
        outlier_mask |= (df[col] < (Q1 - iqr_multiplier * IQR)) | \
                        (df[col] > (Q3 + iqr_multiplier * IQR))

    # Use the mask to split the data
    df_no_outliers = df[~outlier_mask]
    df_outliers = df[outlier_mask]

    return df_no_outliers, df_outliers