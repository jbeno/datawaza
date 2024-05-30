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
__version__ = "0.1.3"
__license__ = "GNU GPLv3"

from .explore import (get_outliers,
                      get_corr,
                      get_unique,
                      plot_charts,
                      plot_corr,
                      plot_3d,
                      plot_map_ca,
                      plot_scatt,
                      print_ascii_image)

from .clean import (convert_data_values,
                    convert_dtypes,
                    convert_time_values,
                    reduce_multicollinearity,
                    split_outliers)

from .tools import (calc_pfi,
                    calc_vif,
                    check_for_duplicates,
                    DebugPrinter,
                    extract_coef,
                    format_df,
                    log_transform,
                    model_summary,
                    split_dataframe,
                    dollars,
                    thousands,
                    LogTransformer)

from .model import (compare_models,
                    create_nn_binary,
                    create_nn_multi,
                    create_pipeline,
                    create_results_df,
                    eval_model,
                    iterate_model,
                    plot_acf_residuals,
                    plot_results,
                    plot_train_history)
