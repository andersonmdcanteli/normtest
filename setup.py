from setuptools import setup, find_packages
import pathlib
import sys

# Python supported version checks. Keep right after stdlib imports to ensure we
# get a sensible error for older Python versions
if sys.version_info[:2] < (3, 9) or sys.version_info[:2] > (3, 11):
    raise RuntimeError("Python version 3.10 or 3.11 required.")



exec(open('normtest/version.py').read())

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()



requirements = ["matplotlib==3.7.1", "numpy==1.23.5", "pandas==1.5.3",
                "scipy==11.11.2", 
]

setup(
    name=__name__.lower(),
    python_requires='>=3.10, <3.12',
    version=__version__,
    author=__author__,
    author_email="andersonmdcanteli@gmail.com",
    description="A package to make scientific research easier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andersondcanteli/normtest",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    include_package_data=True,
    keywords="statistics, sample analisys",  # Optional
)
