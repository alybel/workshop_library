from setuptools import setup

setup(
    name='workshop_library',
    version='1.0',
    author='Alexander Beck',
    packages=['workshop_library'],
    install_requires=['sklearn', 'stockstats', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'joblib', 'scipy']
)


import os
if not os.path.isfile('workshop_library/settings.py'):
    with open('workshop_library/settings.py', 'w') as f:
        f.write('database="data/financial_data.db"\n')
        f.write('model_path="models/"\n')

    from workshop_library import settings
    for p in [settings.model_path, settings.database]:
        if not os.path.exists(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p))
