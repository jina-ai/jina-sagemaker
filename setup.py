import sys
from os import path

from setuptools import find_packages, setup

try:
    pkg_name = "jina-sagemaker"
    libinfo_py = "jina_sagemaker/__init__.py"
    libinfo_content = open(libinfo_py, "r", encoding="utf8").readlines()
    version_line = [l.strip() for l in libinfo_content if l.startswith("__version__")][
        0
    ]
    exec(version_line)  # gives __version__
except FileNotFoundError:
    __version__ = "0.0.0"

try:
    with open("README.md", encoding="utf8") as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ""

setup(
    name=pkg_name,
    packages=find_packages(),
    version=__version__,
    include_package_data=True,
    description="Python library for the Jina endpoints in AWS Sagemaker",
    author="Jina AI",
    author_email="hello@jina.ai",
    license="Apache 2.0",
    url="https://github.com/jina-ai/jina-sagemaker",
    download_url="https://github.com/jina-ai/jina-sagemaker/tags",
    long_description=_long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    setup_requires=["setuptools>=18.0", "wheel"],
    install_requires=["boto3", "sagemaker"],
    extras_require={
        "test": [
            "pytest",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
