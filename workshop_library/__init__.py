import seaborn as sns

sns.set_style('darkgrid', {"axes.facecolor": ".9"})
sns.set_context('poster')

flatui = ["#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))


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

from .data_manager import get_symbol, get_tsymbol
from .live_forecasts.data_utils import initialize_data_for_symbols, refresh_data_for_symbols