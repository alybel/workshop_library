from setuptools import setup

setup(
    name='workshop_library',
    version='1.0',
    author='Alexander Beck',
    packages=['workshop_library'],
    install_requires=['sklearn', 'stockstats', 'numpy',
                      'pandas', 'matplotlib', 'seaborn', 'joblib', 'scipy']
)
