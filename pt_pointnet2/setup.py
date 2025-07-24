from setuptools import setup, find_packages

setup(
    name='pt_pointnet2',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A PyTorch implementation of PointNet2 for point cloud processing.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.7.0',
        'numpy',
        'scipy',
        'matplotlib',
        'pytest'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)