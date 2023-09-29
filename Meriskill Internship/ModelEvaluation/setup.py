from setuptools import setup, find_packages

setup(
    name='ModelEvaluation',
    version='1.0',
    author='Lucas Vital',
    author_email='lucas@123.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.3',
        'pandas>=1.3.4',
        'sklearn',
        'matplotlib'
    ],
)