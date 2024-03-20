# datawaza/__init__.py
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
Datawaza: A Python package for streamlined data science workflows.

Datawaza provides a collection of modules designed to simplify data
exploration, cleaning, modeling, and visualization tasks. It leverages
common data science libraries to offer a higher-level API for common
data science operations, making it easier to perform complex analyses.

Modules:
    - :mod:`datawaza.explore`: Tools for exploratory data analysis and visualization.
    - :mod:`datawaza.clean`: Functions for data cleaning and preprocessing.
    - :mod:`datawaza.model`: Utilities for model training, evaluation, and iteration.
    - :mod:`datawaza.tools`: Miscellaneous helper functions and utilities.
"""

# Metadata
__author__ = "Jim Beno"
__email__ = "jim@jimbeno.net"
__version__ = "0.1.2"
__license__ = "GNU GPLv3"

from .explore import (get_outliers,
                      get_corr,
                      get_unique,
                      plot_charts,
                      plot_corr,
                      plot_3d,
                      plot_map_ca)

from .clean import (convert_data_values,
                    convert_dtypes,
                    convert_time_values,
                    reduce_multicollinearity,
                    split_outliers)

from .tools import (calc_pfi,
                    calc_vif,
                    check_for_duplicates,
                    extract_coef,
                    format_df,
                    log_transform,
                    split_dataframe,
                    thousand_dollars,
                    thousands,
                    LogTransformer)

from .model import (create_pipeline,
                    create_results_df,
                    iterate_model,
                    plot_results)
