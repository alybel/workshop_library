import seaborn as sns

sns.set_style('darkgrid')
sns.set_context('talk')
import pylab
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

np.seed = 42
ms = pd.DataFrame({'A': np.random.randn(20), 'B': np.random.randn(20), 'C': 100*np.cumprod(1+np.random.randn(20)/100.)},
                  index=pd.date_range(start='2100-01-01', periods=20))
ms.index.name = 'Date'
ms['target'] = ms['C'].pct_change().shift(-1)
