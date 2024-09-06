from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='flask-ml',
    version='0.0.1',
    url='https://github.com/UMass-Rescue/Flask-ML',
    license='MIT',
    author='Prasanna Lakkur Subramanyam',
    author_email='psubramanyam@umass.edu',
    description="A Flask extension for running machine learning code",
    long_description="A Flask extension for running machine learning code",
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    platforms='any',
    install_requires=[
        'Flask',
        'requests'
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
