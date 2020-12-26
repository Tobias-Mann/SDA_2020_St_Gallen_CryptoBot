import numpy as np
from datetime import datetime
#For Prediction
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib as plt


# import dataframe
df_raw = pd.read_csv('/Users/tgraf/Google Drive/Uni SG/Master/Smart Data Analytics/00 Group Project/Data/df_raw.csv')

# Set the Subset here
index = df_raw[df_raw['Time'] == '2019-12-01 00:00:00'].index.values[0]

# Take the Subset of the Data
x = index
y = len(df_raw)
df_subset = df_raw[x:y]
print('Number of rows: {}, Number of columns: {}'.format(*df_subset.shape))

# Data Handling
df_subset = df_subset.drop(df_subset.columns.difference(['Time', 'Open', 'Close', 'High', 'Low', 'Volume']), axis = 1)
df_subset['DateTime'] = pd.to_datetime(df_subset['Time'])
df_subset.set_index('DateTime', inplace = True, drop = True)
df_subset.drop(columns = 'Time', inplace = True)
# df_subset.reset_index(inplace = True, drop = True)

"""
# compute the percentage change
df_subset = df_subset.pct_change()
df_subset = df_subset.fillna(0)
"""
# create y
df_subset['prediction'] = df_subset['Close'].shift(-1)
df_subset = df_subset.fillna(0)

# make a prediction with a lasso regression model on the dataset
X, y = df_subset.loc[:, df_subset.columns != 'prediction'], df_subset['prediction']
model = Lasso(alpha=1.0) # alpha represents the lambda
model.fit(X, y)
# make a prediction
yhat = model.predict(X)
print(yhat)

df_subset['yhat'] = yhat
df_subset['Close'].plot()
df_subset['yhat'].plot()

### STEP 2 ---------------------------------------
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = np.arange(0, 1, 0.01)
# define search
search = GridSearchCV(model,
                      grid,
                      scoring = 'neg_mean_absolute_error',
                      cv = cv,
                      n_jobs = -1)
# perform the search
results = search.fit(X, y)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

print(model.coef_)
print(model.intercept_)