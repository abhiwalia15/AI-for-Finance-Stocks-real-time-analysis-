
import pandas as pd
import numpy as np
#%matplotlib inline
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__) # requires version >= 1.9.0
import cufflinks as cf

# For Notebooks
init_notebook_mode(connected=True)
# For offline use
cf.go_offline()

df = pd.read_csv('535755.csv')
df.iplot(kind='scatter', x='Date', y='Close Price')
