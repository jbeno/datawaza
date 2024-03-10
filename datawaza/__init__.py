# datawaza/__init__.py

from .explore import (get_outliers,
                      get_corr,
                      get_unique,
                      plot_charts,
                      plot_corr)

from .clean import (convert_data_values,
                    convert_dtypes,
                    convert_time_values,
                    reduce_multicollinearity)

from .tools import (check_for_duplicates)

# from .model import (iterate_model,
#                     create_pipeline,
#                     results_df,
#                     calc_vif,
#                     calc_fpi)
