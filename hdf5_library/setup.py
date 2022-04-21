from setuptools import setup

setup(
    name='h5_dataset',
    version='0.2',    
    description='This library provides an easy way of using packaging almost any PyTorch dataset into an HDF5 file.',
    python_requires='>=3.8',
    url='',
    author='Ricardo Bigolin Lanfredi',
    author_email='ricardolanfredi@gmail.com',
    license='MIT',
    packages=['h5_dataset'],
    install_requires=[
                          'torch',
                          'numpy>=1.21.2',
                          'dill>=0.3.4',
                          'h5py>=2.10.0',
                          'pillow',
                          'joblib>=1.1.0',
                      ],

    classifiers=[
    ],
)
