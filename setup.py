from setuptools import setup, find_packages

setup(
    name="datawaza",
    version="0.1.0",
    python_requires='>=3.10',
    packages=find_packages(),
    package_data={
        # Specify files within the datawaza package
        'datawaza': ['data/*.xml', 'data/*.dbf', 'data/*.shp', 'data/*.shx', 'data/*.cpg', 'data/*.prj'],
    },
    include_package_data=True,
    install_requires=[
        'pandas~=1.2.1',
        'numpy~=1.26.2',
        'matplotlib~=3.8.2',
        'seaborn~=0.13.0',
        'pytz~=2024.1',
        'scikit-learn~=1.4.1.post1',
        'category_encoders~=2.6.3',
        'joblib~=1.3.2',
        'setuptools~=69.1.1',
        'typing~=3.7.4.3',
        'scipy~=1.11.4',
        'cartopy~=0.22.0',
        'geopandas~=0.14.3',
        'statsmodels~=0.14.1',
        'plotly~=5.19.0'
    ],
    author="Jim Beno",
    author_email="jim@jimbeno.net",
    description="Datawaza is a collection of tools for data exploration, visualization, data cleaning, pipeline creation, model iteration, and evaluation.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jbeno/datawaza",
    keywords=['data science', 'visualization', 'machine learning', 'data analysis'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ]
)
