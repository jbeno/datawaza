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
It contains functions to detect for duplicates in lists.

Functions:
    - :func:`~datawaza.tools.check_for_duplicates` - Check for duplicate items (ex: column names) across multiple lists.
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
import inspect
from scipy.stats import iqr
import re


# Functions
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

