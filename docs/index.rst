.. Datawaza documentation master file, created by
   sphinx-quickstart on Sun Jan 21 12:45:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Datawaza documentation
======================
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

User Guide
................

:doc:`userguide` is a Jupyter notebook that walks through how to use the Datawaza functions. It's probably the best place to start, and then you can reference the function specs organized by module above.

Source Code
...........

You can find the `Datawaza repo <https://github.com/jbeno/datawaza/>`_ on Github. Please submit any issues there. It's distributed under the GNU General Public License. Contributions are welcome!

What is Waza?
.............

Waza (技) means "technique" in Japanese. In martial arts like Aikido, it is paired with words like "suwari-waza" (sitting techniques) or "kaeshi-waza" (reversal techniques). So we've paired it with "data" to represent Data Science techniques: データ技 "data-waza".

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