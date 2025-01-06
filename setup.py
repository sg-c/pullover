from setuptools import setup, find_packages

setup(
    name='pullover',
    version='0.1.0',
    author='sg-c',
    author_email='me@sgche.me',
    description='A deep learning project for computer vision using PyTorch.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'pandas',
    ],
    entry_points={
        'console_scripts': [
            'po=po:main',  # This allows you to run `po` from the command line
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
