"""
explore.py

This module provides tools and utilities to streamline exploratory data analysis.
It contains functions to find unique values, plot distributions, and more.

Functions:
    - get_unique: Display the unique values in a dataset with counts, percents, plots.

Copyright (C) 2023 Jim Beno

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

# Metadata
__author__ = "Jim Beno"
__email__ = "jim@jimbeno.net"
__version__ = "0.1"
__license__ = "GNU GPLv3"

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Functions
def get_unique(df, n=20, sort='count', show_list=True, count=True, percent=True, plot=False, cont=False, strip=False,
               dropna=False, fig_size=(6, 4), rotation=45):
    """
    Obtains unique values of all variables below a threshold number "n", and can display counts or percents.
    Additionally, it can analyze variables over 'n' as continuous data.

    Parameters:

    - df: DataFrame. The dataframe that contains the variables you want to analyze.
    - n: int, optional (default is 20). The maximum number of unique values to consider. This helps to avoid
        iterating over continuous data.
    - sort: str, optional (default='count'). Determines the sorting of unique values:
        - 'name' - sorts alphabetically/numerically,
        - 'count' - sorts by count of unique values (descending),
        - 'percent' - sorts by percentage of each unique value (descending).
    - show_list: boolean, optional (default=True). If True, shows the list of unique values.
    - count: boolean, optional (default=False). If True, shows counts of each unique value.
    - percent: boolean, optional (default=False). If True, shows the percentage of each unique value.
    - plot: boolean, optional (default=False). If True, shows a basic chart for each variable.
    - cont: boolean, optional (default=False). If True, analyzes variables with unique values greater than 'n'
        as continuous data.
    - strip: boolean, optional (default=False). If True, removes single quotes from the variable names.
    - dropna: boolean, optional (default=False). If True, excludes NaN values from unique value lists.
    - fig_size: tuple, optional (default=(6, 4)). Size of figure if plotting is enabled.
    - rotation: integer, optional (default=45). Rotation angle of X axis ticks if plotting is enabled.

    Returns:
    None. The function prints the analysis directly.

    Example:
    dw.get_unique(df, n=15, sort='count', plot=True, cont=True)

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
            perc = round(number / df.shape[0] * 100, 2)
            # Copy the index to a column
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
                        print("\t" + str(w_pad) + str(y_pad) + str(z) + "%")
                    elif count:
                        print("\t" + str(w_pad) + str(y))
                    elif percent:
                        print("\t" + str(w_pad) + str(z) + "%")
                    else:
                        print("\t" + str(w))

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
