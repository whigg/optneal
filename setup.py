import os
from setuptools import setup, find_packages


def read_requirements():
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='genqubo',
    version='0.0.1',
    description='Generate QUBO matrix from constraints',
    long_description='README.md',
    author='Mull Zhang',
    install_requires=['dimod', 'numpy'],
    url='https://github.com/mullzhang/genqubo',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    test_suite='tests'
)