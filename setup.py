#!/usr/bin/env python3
import pathlib

import setuptools
from ruth import VERSION


here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

core_requirements = [
    "click~=8.0.1",
    "rich~=12.0.0",
    "aiohttp~=3.6.3",
    "jsonpickle~=2.1.0",
    "pyyaml~=6.0",
    "torch~=1.12.1",
    "regex",
    "scikit-learn~=1.0.2",
    "matplotlib~=3.5.3",
    "progressbar~=2.5",
    "transformers~=4.20.0",
]


setuptools.setup(
    name='ruth-python',
    description="A Python CLI for Ruth NLP",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="",
    author='Puretalk',
    author_email='info@puretalk.ai',
    version=VERSION,
    install_requires=core_requirements,
    python_requires='>=3.7,<=3.10',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    include_package_data=True,
    package_data={
        "data": ["*.txt"]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3.8',
    ],
    entry_points={"console_scripts": ["ruth = ruth.cli.cli:entrypoint"]},
)