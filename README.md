<br />
<img src="https://www.datawaza.com/en/latest/_static/datawaza_logo_name_trans.svg" alt="datawaza_logo_name_trans.svg" width="300"/>

--------------------------------------
[![PyPI Version](https://img.shields.io/pypi/v/datawaza)](https://pypi.org/project/datawaza/)
[![License](https://img.shields.io/github/license/jbeno/datawaza)](https://github.com/jbeno/datawaza/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/jbeno/datawaza)](https://github.com/jbeno/datawaza)
[![Documentation Status](https://readthedocs.org/projects/datawaza/badge/?version=latest)](https://www.datawaza.com/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/jbeno/datawaza/badge.svg?branch=main)](https://coveralls.io/github/jbeno/datawaza?branch=main)
[![Python Version](https://img.shields.io/pypi/pyversions/datawaza)]()

Datawaza streamlines common Data Science tasks. It's a collection of tools for data exploration, visualization, data cleaning, pipeline creation, hyper-parameter searching, model iteration, and evaluation. It builds upon core libraries like [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), and [Scikit-Learn](https://scikit-learn.org/stable/).

<p align="center">
  <a href="https://www.datawaza.com/en/latest/explore.html#datawaza.explore.plot_charts"><img src="https://www.datawaza.com/en/latest/_static/plot_charts.png" width="30%" /></a>
  <a href="https://www.datawaza.com/en/latest/explore.html#datawaza.explore.plot_map_ca"><img src="https://www.datawaza.com/en/latest/_static/plot_map_ca.png" width="30%" style="margin:0 1%;" /></a>
  <a href="https://www.datawaza.com/en/latest/explore.html#datawaza.explore.plot_3d"><img src="https://www.datawaza.com/en/latest/_static/plot_3d.png" width="30%" /></a>
</p>
<p align="center">
  <a href="https://www.datawaza.com/en/latest/model.html#datawaza.model.iterate_model"><img src="https://www.datawaza.com/en/latest/_static/iterate_model_1.png" width="30%" /></a>
  <a href="https://www.datawaza.com/en/latest/model.html#datawaza.model.iterate_model"><img src="https://www.datawaza.com/en/latest/_static/iterate_model_2.png" width="30%" style="margin:0 1%;" /></a>
  <a href="https://www.datawaza.com/en/latest/model.html#datawaza.model.plot_results"><img src="https://www.datawaza.com/en/latest/_static/plot_results.png" width="30%" /></a>
</p>
<p align="center">
  <a href="https://www.datawaza.com/en/latest/explore.html#datawaza.explore.get_corr"><img src="https://www.datawaza.com/en/latest/_static/get_corr.png" width="30%" /></a>
  <a href="https://www.datawaza.com/en/latest/clean.html#datawaza.clean.reduce_multicollinearity"><img src="https://www.datawaza.com/en/latest/_static/reduce_multicollinearity.png" width="30%" style="margin:0 1%;" /></a>
  <a href="https://www.datawaza.com/en/latest/explore.html#datawaza.explore.plot_corr"><img src="https://www.datawaza.com/en/latest/_static/plot_corr.png" width="30%" /></a>
</p>

Installation
------------

The latest release can be found on [PyPI](https://pypi.org/project/datawaza/). See the [Change Log](CHANGELOG.md) for a history of changes. Install Datawaza with pip:

    pip install datawaza

Documentation
-------------

Online documentation is available at [Datawaza.com](https://datawaza.com).

The [User Guide](https://www.datawaza.com/en/latest/userguide.html) is a Jupyter notebook that walks through how to use the Datawaza functions. It's probably the best place to start. There is also an API reference for the major modules: [Clean](https://www.datawaza.com/en/latest/clean.html), [Explore](https://www.datawaza.com/en/latest/explore.html), [Model](https://www.datawaza.com/en/latest/model.html), and [Tools](https://www.datawaza.com/en/latest/tools.html).

Development
-----------

The [Datawaza repo](https://github.com/jbeno/datawaza) is on GitHub.

Please submit bugs that you encounter to the [Issue Tracker](https://github.com/jbeno/datawaza/issues). Contributions and ideas for enhancements are welcome! So far this is a solo effort, but I would love to collaborate.

Dependencies
------------

Datawaza supports Python 3.10. It may support other versions, but these have not been tested yet.

Due to the breadth of use cases, installation requires NumPy, Pandas, Matplotlib, Seaborn, Plotly, Scikit-Learn, SciPy, Cartopy, GeoPandas, StatsModels, and a few other supporting packages. See the [Requirements.txt](https://github.com/jbeno/datawaza/blob/main/requirements.txt).

What is Waza?
-------------

Waza (技) means "technique" in Japanese. In martial arts like Aikido, it is paired with words like "suwari-waza" (sitting techniques) or "kaeshi-waza" (reversal techniques). So we've paired it with "data" to represent Data Science techniques: データ技 "data-waza".

Origin Story
-------------

Most of these functions were created while I was pursuing a [Professional Certificate in Machine Learning & Artificial Intelligence](https://em-executive.berkeley.edu/professional-certificate-machine-learning-artificial-intelligence) from U.C. Berkeley. With every assignment, I tried to simplify repetitive tasks and streamline my workflow. They served me well, and I hope you will find some value in them.

Quick Start
-----------

The [User Guide](https://www.datawaza.com/en/latest/userguide.html) will show you how to use Datawaza's functions in depth. Assuming you already have data loaded, here are some examples of what it can do:

    >>> import datawaza as dw
    
Show the unique values of each variable below the threshold of n = 12:

    >>> dw.get_unique(df, 12, count=True, percent=True)

    CATEGORICAL: Variables with unique values equal to or below: 12
    
    job has 12 unique values:
    
        admin.              10422   25.3%
        blue-collar         9254    22.47%
        technician          6743    16.37%
        services            3969    9.64%
        management          2924    7.1%
        retired             1720    4.18%
        entrepreneur        1456    3.54%
        self-employed       1421    3.45%
        housemaid           1060    2.57%
        unemployed          1014    2.46%
        student             875     2.12%
        unknown             330     0.8%
    
    marital has 4 unique values:
    
        married        24928   60.52%
        single         11568   28.09%
        divorced       4612    11.2%
        unknown        80      0.19%

Plot bar charts of categorical variables, dimensioned by the target variable:

    >>> dw.plot_charts(df, plot_type='cat', cat_cols=cat_columns, hue='y', rotation=90)

![plot_charts output](https://www.datawaza.com/en/latest/_static/plot_charts_output.png)

Get the top positive and negative correlations with the target variable, and save to lists:

    >>> pos_features, neg_features = dw.get_corr(df_enc, n=10, var='subscribed_enc', return_arrays=True)

    Top 10 positive correlations:
    Variable 1      Variable 2  Correlation
    0               duration  subscribed_enc         0.41
    1       poutcome_success  subscribed_enc         0.32
    2   previously_contacted  subscribed_enc         0.32
    3                  pdays  subscribed_enc         0.27
    4               previous  subscribed_enc         0.23
    5              month_mar  subscribed_enc         0.14
    6              month_oct  subscribed_enc         0.14
    7              month_sep  subscribed_enc         0.12
    8           no_default_1  subscribed_enc         0.10
    9            job_student  subscribed_enc         0.09
    
    Top 10 negative correlations:
    Variable 1      Variable 2  Correlation
    0            nr.employed  subscribed_enc        -0.35
    1              euribor3m  subscribed_enc        -0.31
    2           emp.var.rate  subscribed_enc        -0.30
    3   poutcome_nonexistent  subscribed_enc        -0.19
    4      contact_telephone  subscribed_enc        -0.14
    5         cons.price.idx  subscribed_enc        -0.14
    6              month_may  subscribed_enc        -0.11
    7               campaign  subscribed_enc        -0.07
    8        job_blue-collar  subscribed_enc        -0.07
    9     education_basic.9y  subscribed_enc        -0.05

Plot a chart showing the top correlations with the target variable:

    >>> dw.plot_corr(df_enc, 'subscribed_enc', n=16, size=(12,6), rotation=90)

![plot_corr output](https://www.datawaza.com/en/latest/_static/plot_corr_output.png)

Run a model iteration, which dynamically assembles a pipeline and evaluates the model, including
charts of residuals, predicted vs. actual, and coefficients:

    >>> results_df, iteration_6 = dw.iterate_model(X2_train, X2_test, y2_train, y2_test,
    ...     transformers=['ohe', 'log', 'poly3'], model='linreg',
    ...     iteration='6', note='X2. Test size: 0.25, Pipeline: OHE > Log > Poly3 > LinReg',
    ...     plot=True, lowess=True, coef=True, perm=True, vif=True, decimal=2,
    ...     save=True, save_df=results_df, config=my_config)

![iterate_model output 1 of 3](https://www.datawaza.com/en/latest/_static/iterate_model_output_1.png)
![iterate_model output 2 of 3](https://www.datawaza.com/en/latest/_static/iterate_model_output_2.png)
![iterate_model output 3 of 3](https://www.datawaza.com/en/latest/_static/iterate_model_output_3.png)

Compare train/test scores across model iterations, and select the best result:

    >>> dw.plot_results(results_df, metrics=['Train MAE', 'Test MAE'], y_label='Mean Absolute Error',
    ...     select_metric='Test MAE', select_criteria='min', decimal=0)

![plot_results output](https://www.datawaza.com/en/latest/_static/plot_results_output.png)

This was just a sample of some Datawaza tools. Download [userguide.ipynb](https://github.com/jbeno/datawaza/blob/main/docs/userguide.ipynb) and explore the full breadth of the library in your Jupyter environment.
