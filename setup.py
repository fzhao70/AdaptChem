from setuptools import setup, find_packages

setup(
    name='AdaptChem',  # Name of the library
    version='0.1.0',  # Version
    packages=find_packages(),  # Automatically find all packages in the library
    install_requires=[
        'numpy',  # Dependencies your library needs
        'matplotlib',
        'torch',
        'gymnasium',
        'pandas',
        'stable_baselines3',
    ],
    author='Fanghe Zhao',
    author_email='fzhao97@gmail.com',
    description='Adaptive Atmospheric Chemistry Mechanism Toolkit and Framework with Hybrid Machine Learning',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GPL-3.0 license',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)