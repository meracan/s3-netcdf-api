import setuptools
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="s3netcdfapi",
    version="0.0.1",
    author="Julien Cousineau",
    author_email="Julien.Cousineau@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meracan/s3-netcdf-api",
     packages=["s3netcdfapi"],
    install_requires=[
        
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)