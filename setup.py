from setuptools import setup, find_packages

setup(
    name="datawaza",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[],
    author="Jim Beno",
    author_email="jim@jimbeno.net",
    description="Data science tools for exploration, visualization, and model iteration.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jbeno/datawaza",
    keywords=['data science', 'visualization', 'machine learning', 'data analysis'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ]
)

#%%
