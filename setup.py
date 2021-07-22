#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Notes:
# * To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev
# * File `MANIFEST.in` is obsolete, see
#   https://www.remarkablyrestrained.com/python-setuptools-manifest-in/
#   Use `setuptools_scm` instead

import io
import os

import setuptools

# Package meta-data.
NAME = "smlb"
VERSION = "0.4.0"
DESCRIPTION = "Scientific Machine Learning Benchmark"
URL = "https://github.com/CitrineInformatics/smlb"
EMAIL = "mrupp@mrupp.io"
AUTHOR = "Matthias Rupp"
LICENSE = "Apache-2.0"  # SPDX short identifier
LICENSE_TROVE = "License :: OSI Approved :: Apache Software License"  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
REQUIRES_PYTHON = ">=3.7.0"

here = os.path.abspath(os.path.dirname(__file__))

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

# Import the README and use it as the long-description.
# todo: verify that MANIFEST.in is indeed not necessary
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    setup_requires=["setuptools_scm"],
    url=URL,
    # project_urls={"Bug Tracker": ..., "Documentation": ..., "Source Code": ...},
    packages=setuptools.find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=[
        "numpy>=1.21.0,<2",
        "scipy>=1.7.0,<2",
        "pandas>=1.2.5,<2",
        "matplotlib>=3.4.2,<4",
        "scikit-learn>=0.24.0,<0.25",
    ],
    extras_require={
        "lolo": [
            "lolopy>=1.2.0,<2",
            "py4j>=0.10.9,<0.11",
        ],
        "optional": [
            "py4j>=0.10.9,<0.11",
            "pymatgen>=2022.0.9",
            "matminer>=0.7.3,<0.8",
        ],
    },
    include_package_data=True,
    license=LICENSE,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        LICENSE_TROVE,
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
