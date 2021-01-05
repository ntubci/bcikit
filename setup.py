# -*- coding: utf-8 -*-
'''
Read doc: 
    https://packaging.python.org/guides/distributing-packages-using-setuptools/
    https://packaging.python.org/tutorials/packaging-projects/

Steps:
    python3 setup.py sdist
    python3 -m twine upload --repository pypi dist/*

'''

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.read().splitlines()

setuptools.setup(
    name="bcikit",
    version="0.0.1.dev1",
    author="NTU Brain-Computing Research",
    author_email="ntuscsebci@gmail.com",
    description="A toolbox for data manipulation and transformation for EEG and machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ntubci/bcikit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.3',
    project_urls={
        'Documentation': 'https://bcikit.readthedocs.io',
        'Source': 'https://github.com/ntubci/bcikit',
        'Tracker': 'https://github.com/ntubci/bcikit/issues',
    },
    install_requires=install_requires
)
