import seaborn as sns

sns.set_style('darkgrid')
sns.set_context('talk')
import pylab
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np


ms = pd.DataFrame({'A': np.random.randn(20), 'B': np.random.randn(20), 'C': np.random.randn(20)},
                  index=pd.date_range(start='2100-01-01', periods=20))
