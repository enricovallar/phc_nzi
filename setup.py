from setuptools import setup, find_packages

setup(
    name='phc_nzi',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'schwimmbad',
        'mpi4py',
        'filelock'
    ],
    entry_points={
        'console_scripts': [
            'phc_nzi=phc_nzi.main:main',  # If you have a main script
        ],
    },
)