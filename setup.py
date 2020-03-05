"""
Flask-ML
-------------

A Flask extension for running machine learning code
"""
from setuptools import setup


setup(
    name='Flask-ML',
    version='1.0',
    url='https://github.com/UMass-Rescue',
    license='MIT',
    author='Jagath Jai Kumar',
    author_email='jagath.jaikumar@gmail.com',
    description='A Flask extension for running machine learning code',
    long_description=__doc__,
    py_modules=['flask_ml'],
    zip_safe=False,
    include_package_data=True,
    platforms='any',
    install_requires=[
        'Flask',
        'numpy',
        'pillow',
        'jsonpickle',
        'requests',
        'pybase64'

    ],
    python_requires=">= 3.6",
    classifiers=[
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python3.6',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
