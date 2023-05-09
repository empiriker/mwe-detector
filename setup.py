# from setuptools import setup

# import os

# lib_folder = os.path.dirname(os.path.realpath(__file__))
# requirement_path = lib_folder + "/requirements.txt"
# install_requires = []
# if os.path.isfile(requirement_path):
#     with open(requirement_path) as f:
#         install_requires = f.read().splitlines()
import setuptools

setuptools.setup(
    name="mwe_detector",
    version="0.0.1",
    description="A SpaCy MWEDetector pipeline component",
    url="https://github.com/empiriker/mwe_detection.git#egg=mwe_detector&subdirectory=mwe_detector",
    author="Till ÜF",
    author_email="till@ueberfries.de",
    license="All rights reserved",
    packages=["mwe_detector"],
    zip_safe=False,
    include_package_data=True,
    package_data={"mwe_detector": ["data/*.json"]},
)
