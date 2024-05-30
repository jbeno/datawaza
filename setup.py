from setuptools import setup, find_packages

# Read the contents of the README file
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name = 'datawaza',
    version = '0.1.3',
    python_requires = '>=3.9, <3.13',
    packages = find_packages(),
    package_data = {
        # Map files and sample PyTorch image tensors
        'datawaza': ['data/*.xml', 'data/*.dbf', 'data/*.shp', 'data/*.shx', 'data/*.cpg', 'data/*.prj', 'data/*.pt'],
    },
    include_package_data=True,
    install_requires=[
        'pandas>=2.2.1',
        'numpy>=1.26.2',
        'matplotlib>=3.8.2',
        'seaborn>=0.13.2',
        'pytz>=2023.3',
        'scikit-learn>=1.5.0',
        'joblib>=1.3.2',
        'setuptools>=69.1.1',
        'typing>=3.7.4.3',
        'scipy>=1.11.4',
        'cartopy>=0.22.0',
        'geopandas>=0.14.3',
        'statsmodels>=0.14.1',
        'plotly>=5.19.0',
        'nbformat>=4.2.0',
        'importlib_resources>=6.3.2; python_version<"3.10"',
        'scikeras>=0.13.0',
        'xgboost>=2.0.3',
        'imbalanced-learn>=0.12.3',
        'tensorflow>=2.16.1',
        'keras>=3.2.0',
        'torch>=2.1.0',
    ],
    extras_require = {
        'doc': [
            'pydata_sphinx_theme~=0.15.2',
            'sphinx-design~=0.5.0',
            'Sphinx~=7.2.6',
            'myst-parser~=2.0.0',
            'sphinx-favicon~=1.0.1',
            'nbsphinx~=0.9.3',
        ],
        'test': [
            'pytest>=8.1.1',
            'pytest-cov>=4.1.0',
            'coverage>=6.5.0',
            'coveralls>=3.3.1',
        ]
    },
    author='Jim Beno',
    author_email='jim@jimbeno.net',
    description="Datawaza is a collection of tools for data exploration, visualization, data cleaning, pipeline creation, model iteration, and evaluation.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://datawaza.com',
    project_urls={
        'Documentation': 'https://datawaza.com',
        'Source': 'https://github.com/jbeno/datawaza'
    },
    keywords=['data science', 'visualization', 'machine learning', 'data analysis'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ]
)
