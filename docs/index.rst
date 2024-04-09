Datawaza
========
Datawaza is a collection of tools for data exploration, visualization, data cleaning, pipeline creation, model iteration, and evaluation. It builds upon core libraries like `pandas <https://pandas.pydata.org/>`_, `matplotlib <https://matplotlib.org/>`_, `seaborn <https://seaborn.pydata.org/>`_, and `scikit-learn <https://scikit-learn.org/stable/>`_.

Modules
.......

.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Explore
        :img-top: _static/explore-blue.svg
        :class-img-top: intro-card-icon
        :class-card: intro-card
        :shadow: md
        :link: explore
        :link-type: doc

        Quickly explore and visualize your data.

    .. grid-item-card::  Clean
        :img-top: _static/clean-blue.svg
        :class-card: intro-card
        :shadow: md
        :link: clean
        :link-type: doc

        Clean your data and engineer features.

    .. grid-item-card::  Model
        :img-top: _static/model-blue.svg
        :class-card: intro-card
        :shadow: md
        :link: model
        :link-type: doc

        Create pipelines, iterate and evaluate models.

    .. grid-item-card::  Tools
        :img-top: _static/tools-blue.svg
        :class-card: intro-card
        :shadow: md
        :link: tools
        :link-type: doc

        Additional utilities and helper functions.

Installation
............

The `latest releases <https://pypi.org/project/datawaza/>`_ can be found on PyPi. Install Datawaza with pip::

   pip install datawaza

See the `Change Log <https://github.com/jbeno/datawaza/blob/main/CHANGELOG.md>`_ for a history of changes.

User Guide
..........

:doc:`userguide` is a Jupyter notebook that walks through how to use the Datawaza functions. It's probably the best place to start, and then you can reference the function specs organized by module above.

Source Code
...........

You can find the `Datawaza repo <https://github.com/jbeno/datawaza/>`_ on Github. Please submit any issues there. It's distributed under the GNU General Public License. Contributions are welcome!

What is Waza?
.............

Waza (技) means "technique" in Japanese. In martial arts like Aikido, it is paired with words like "suwari-waza" (sitting techniques) or "kaeshi-waza" (reversal techniques). So we've paired it with "data" to represent Data Science techniques: データ技 "data-waza".

Origin Story
.............

Most of these functions were created while I was pusuring a `Professional Certificate in Machine Learning & Artificial Intelligence <https://em-executive.berkeley.edu/professional-certificate-machine-learning-artificial-intelligence>` from U.C. Berkeley. With every assignment, I tried to simplify repetitive tasks and streamline my workflow. They served me well, so I'm publishing this library in the hope that it may help others.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   explore
   clean
   model
   tools
   userguide

Reference
.........

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`