"""
Flask-ML-client
-------------

A Flask extension for running machine learning code interfacing with flask server
"""
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='flask-ml-client',
    version='0.0.6',
    url='https://github.com/UMass-Rescue',
    license='MIT',
    author='Jagath Jai Kumar',
    author_email='jagath.jaikumar@gmail.com',
    description="A Flask extension for running machine learning code",
    long_description="A Flask extension for running machine learning code",
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    platforms='any',
    install_requires=[
        'numpy',
        'requests',
        'encoder_decoder'
    ],
    classifiers=[
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
