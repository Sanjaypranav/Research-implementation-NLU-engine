#!/usr/bin/env python3
import pathlib

import setuptools
from ruth import VERSION


here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

core_requirements = [
    "click~=8.0.0",
    "randomname~=0.1.3",
    "coloredlogs~=14.0.0",
    "rich~=11.1.0",
    "aiohttp~=3.6.3",
    "numpy~=1.16.1",
    "requests~=2.23.0",
    "pandas~=1.2.5"
]


setuptools.setup(
    name='ruth',
    description="A Python CLI for Ruth NLP",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="",
    author='Prakash R; Kasi Viswanathan I',
    author_email='prakashr7d@gmail.com; vinithkac@gmail.com',
    version=VERSION,
    install_requires=core_requirements,
    python_requires='>=3.8,<3.9',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    include_package_data=True,
    package_data={
        "data": ["*.txt"]
    },
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3.7',
    ],
    entry_points={"console_scripts": ["ruth = ruth.cli.cli:entrypoint"]},
)
