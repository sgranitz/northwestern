#Import packages 
import pandas as pd 
import numpy as np 
import statsmodels.api as sm
import statsmodels.formula.api as smf  
import matplotlib.pyplot as plt 
import seaborn as sns
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import feature_selection
from tabulate import tabulate
from statsmodels.iolib.summary2 import summary_col
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold

#Set some display options   
pd.set_option('display.notebook_repr_html', False) 
pd.set_option('display.max_columns', 40) 
pd.set_option('display.max_rows', 10) 
pd.set_option('display.width', 120)
 
# Read in the ames train test datasets
path  = 'C:/Users/sgran/Desktop/northwestern/predict_410/assignment_1/'
train = pd.read_csv(path + 'ames_train.csv')
test  = pd.read_csv(path + 'ames_test.csv')

# convert all variable names to lower case
train.columns = [s.lower() for s in train.columns]
test.columns = [s.lower() for s in test.columns]

# Limit train data to single family homes
train = train[(train.bldgtype == '1Fam')]

cols = [
  'bsmtfinsf1',
  'bsmtfinsf2',
  'garagearea',
  'wooddecksf',
  'openporchsf',
  'enclosedporch',
  'threessnporch',
  'screenporch',
  'poolarea',
  'bsmtfullbath',
  'bsmthalfbath',
  'fullbath',
  'halfbath'
]
train[cols] = train[cols].fillna(0)
test[cols] = test[cols].fillna(0)

lot_ratio = train.lotfrontage.median() / train.lotarea.median()
train.lotfrontage = train.lotfrontage.fillna(train.lotarea * lot_ratio)
test.lotfrontage = test.lotfrontage.fillna(test.lotarea * lot_ratio)

train['qualityindex'] = (train.overallqual * train.overallcond)
train['totalsqftcalc'] = (train.bsmtfinsf1 + train.bsmtfinsf2 + train.grlivarea)
train['outerarea_fin'] = (
  train.garagearea + train.wooddecksf + train.openporchsf +
  train.enclosedporch + train.threessnporch + train.screenporch +
  train.poolarea
)
train['bathrooms'] = (
  train.fullbath + 0.5 * train.halfbath + 
  train.bsmtfullbath + 0.5 * train.bsmthalfbath
)
train['ppsqft'] = (train.saleprice / train.totalsqftcalc)

train[['qualityindex','totalsqftcalc', 'outerarea_fin', 'bathrooms']].hist()


test['qualityindex'] = (test.overallqual * test.overallcond)
test['totalsqftcalc'] = (test.bsmtfinsf1 + test.bsmtfinsf2 + test.grlivarea)
test['outerarea_fin'] = (
  test.garagearea + test.wooddecksf + test.openporchsf +
  test.enclosedporch + test.threessnporch + test.screenporch +
  test.poolarea
)
test['bathrooms'] = (
  test.fullbath + 0.5 * test.halfbath + 
  test.bsmtfullbath + 0.5 * test.bsmthalfbath
)

print("Train neighborhoods: ", len(train.neighborhood.unique()))
print("Test neighborhoods: ", len(test.neighborhood.unique()))

cols = [
  'neighborhood',      
  'saleprice',
  'qualityindex',
  'totalsqftcalc',
  'yearbuilt',
  'lotarea',
  'lotfrontage',
  'outerarea_fin',
  'bathrooms'
]

plt.figure()
train[cols].groupby(['neighborhood']).plot(
  kind='box', 
  subplots=True, 
  layout=(5,9), 
  sharex=False, 
  sharey=False, 
  figsize=(18,14)
)
plt.show()

plt.figure()
train.ppsqft.hist()
plt.show()

train[['neighborhood', 'ppsqft']].groupby(['neighborhood']).describe()

plt.figure()
train[['neighborhood', 'ppsqft']].groupby(['neighborhood']).hist()
plt.show()

plt.figure()
train.boxplot(column='saleprice', by='neighborhood', vert=False)
plt.show()

plt.figure()
train[['neighborhood', 'ppsqft']].groupby(['neighborhood']).agg(np.median).hist()
plt.show()

nbhd_med = pd.DataFrame(
  train[[
    'neighborhood', 
    'ppsqft'
  ]].groupby([
    'neighborhood'
  ]).agg(np.median).reset_index()
)

nbhd_med['type'] = pd.cut(nbhd_med['ppsqft'], bins=5, labels=False)
labels = np.arange(5)
nbhd_med['type'] = labels[nbhd_med['type']]

nbhd_map = pd.Series(
  nbhd_med.type.values, 
  index=nbhd_med.neighborhood
).to_dict()

nbhd_med.plot.scatter(x="type", y="ppsqft")
sns.stripplot("type", "ppsqft", data=nbhd_med, jitter=0.2)
sns.despine()

train['nbhd_type'] = train['neighborhood'].map(nbhd_map)
test['nbhd_type'] = test['neighborhood'].map(nbhd_map)

print(train.describe())
#take a look at some correlations with the saleprice
X = train[[
  'saleprice',
  'qualityindex',
  'totalsqftcalc',
  'yearbuilt',
  'lotarea',
  'lotfrontage',
  'outerarea_fin',
  'bathrooms',
  'nbhd_type'
]].copy()

X1 = train[[
  'qualityindex',
  'totalsqftcalc',
  'yearbuilt',
  'lotarea',
  'lotfrontage',
  'outerarea_fin',
  'bathrooms',
  'nbhd_type'
]].copy()

corr = X[X.columns].corr()
print(corr)

Y = train[['saleprice']].copy()
print(Y.head)

select_top_3 = SelectKBest(score_func=chi2, k = 3)
fit = select_top_3.fit(X1,Y)
features = fit.transform(X1)
features[0:7] 

#Set variable list
y = train['saleprice']
plt.plot(y)

#Code for linear regression with categorical variables c()
model1 = smf.ols(
  formula='y ~ qualityindex+totalsqftcalc+C(lotconfig)+C(housestyle)+\
  yearbuilt+C(roofstyle)+C(heating)', 
  data=train
).fit()
model1.summary()

model2 = smf.ols(
  formula='y ~ qualityindex+totalsqftcalc+yearbuilt', 
  data=train
).fit()
model2.summary()

model3 = smf.ols(
  formula='y ~ qualityindex+totalsqftcalc+yearbuilt+outerarea_fin+\
  bathrooms+C(housestyle)', 
  data=train
).fit()
model3.summary()

model4 = smf.ols(
  formula='y ~ qualityindex+totalsqftcalc+yearbuilt+outerarea_fin+\
  bathrooms+neighborhood', 
  data=train
).fit()
model4.summary()

model5 = smf.ols(
  formula='y ~ qualityindex+totalsqftcalc+yearbuilt+outerarea_fin+\
  bathrooms+nbhd_type+C(heating)', 
  data=train
).fit()
model5.summary()

pred = model4.predict(train)
train['pred'] = pred
train['res'] = train.saleprice - train.pred
cols = ['neighborhood', 'saleprice', 'pred', 'res']

train[cols].sort_values('res', ascending=False)
train[cols].groupby(['neighborhood']).agg(np.median).sort_values('res', ascending=False)
train.res.mean()

plt.figure()
sns.boxplot(x=train.res, y=train.neighborhood, orient='h')
plt.show()

train['pred_ppsqft'] = (train.pred / train.totalsqftcalc)

plt.figure()
sns.regplot(x=train.ppsqft, y=train.pred_ppsqft)
plt.show()

plt.figure()
sns.lmplot(
  x='ppsqft', y='pred_ppsqft', data=train, 
  fit_reg=False, hue='neighborhood', 
  size=6, aspect=2
) 
plt.show()

import math
def log(x):
    if x == 0: return 0
    return math.log(x)

def exp(x):
    return math.exp(x)

y = train['saleprice'].apply(log)
model6 = smf.ols(
  formula='y ~ qualityindex+totalsqftcalc+yearbuilt+outerarea_fin+\
  bathrooms+nbhd_type+C(heating)', 
  data=train
).fit()
model6.summary()

train['log_pred'] = model6.predict(train)
train['log_res'] = y - train.log_pred
cols = ['neighborhood', 'saleprice', 'log_pred', 'log_res']
train[cols].sort_values('log_res', ascending=False)
train[cols].groupby(['neighborhood']).agg(np.median).sort_values('log_res', ascending=False)

print(train.log_res.mean())

plt.figure()
sns.boxplot(x=train.log_res, y=train.neighborhood, orient='h')
plt.show()

train['log_pred_ppsqft'] = (train.log_pred / train.totalsqftcalc)
train['log_ppsqft'] = (y / train.totalsqftcalc)

plt.figure()
sns.regplot(x=train.log_ppsqft, y=train.log_pred_ppsqft)
plt.show()

plt.figure()
sns.lmplot(
  x='log_ppsqft', y='log_pred_ppsqft', data=train, 
  fit_reg=False, hue='neighborhood', 
  size=6, aspect=2
) 
plt.show()

train['log_sqft'] = train['totalsqftcalc'].apply(log)
train['yrs_old'] = 2018 - train['yearbuilt']  

features = [
  'qualityindex',
  'log_sqft',
  'yrs_old',
  'outerarea_fin',
  'bathrooms',
  'nbhd_type',
  'heating'
]
features = "+".join(train[features].columns)
 
model7 = smf.ols(formula='y ~' + features, data=train).fit()
model7.summary()

y, X = dmatrices('y ~' + features, train, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print("vif model 7:", vif.T)

features = [
  'qualityindex',
  'log_sqft',
  'yrs_old',
  'outerarea_fin',
  'bathrooms',
  'garagecars',
  'nbhd_type',
  'exterqual'
]
features = "+".join(train[features].columns)

y = train['saleprice'].apply(log)
model8 = smf.ols(formula='y ~' + features, data=train).fit()
model8.summary()

y, X = dmatrices('y ~' + features, train, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print("vif model 8:", vif.T)

test['log_sqft'] = test['totalsqftcalc'].apply(log)
test['yrs_old'] = 2018 - test['yearbuilt']   
test['garagecars'] = test['garagecars'].fillna(1)
model8.predict(test).apply(exp)

# Try feature selection
y = train['saleprice']
X = train[[
  'qualityindex',
  'log_sqft',
  'yrs_old',
  'outerarea_fin',
  'bathrooms',
  'garagecars',
  'nbhd_type'
]].copy()
X.head()
model = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=4)
results = model.fit(X, y)
print(results.scores_)

# Compare models
out = [model4,
       model5,
       model7,
       model8]

out_df = pd.DataFrame()
out_df['labels'] = ['rsquared', 'rsquared_adj', 'fstatistic', 'aic']
i = 0
for model in out:
    print(i)
    plt.figure()
    if (i == 0 or i == 1):
        train['pred'] = model.predict(train)
        train.plot.scatter(x='saleprice', y='pred', title='model' + str(i+1))
    else:
        train['pred'] = model.predict(train).apply(exp)
        train.plot.scatter(x='saleprice', y='pred', title='model' + str(i+1))
    plt.show()
    out_df['model' + str(i+1)] = [      
      model.rsquared.round(3), 
      model.rsquared_adj.round(3), 
      model.fvalue.round(3),
      model.aic.round(3)
    ]
    i += 1

print(tabulate(out_df, headers=out_df.columns, tablefmt='psql'))
print(summary_col(out, stars=True))

plt.figure()
g = sns.PairGrid(train,
                 x_vars=["bathrooms",  
                         "outerarea_fin",
                         "qualityindex", 
                         "totalsqftcalc", 
                         "nbhd_type"],
                 y_vars=["saleprice"],
                 aspect=.75, size=3.5)
g.map(sns.violinplot, palette="pastel");


# Model 8 kfolds
num_folds = 10

# set up numpy array for storing results
results = np.zeros((num_folds, 1))

kf = KFold(
  n_splits=num_folds, 
  shuffle=False, 
  random_state=85
)

train['log_sp'] = train['saleprice'].apply(log)
train8 = np.array([
  np.array(train['qualityindex']),
  np.array(train['log_sqft']), 
  np.array(train['yrs_old']), 
  np.array(train['outerarea_fin']),
  np.array(train['bathrooms']), 
  np.array(train['garagecars']), 
  np.array(train['nbhd_type']),
  #np.array(train['exterqual']),
  np.array(train['log_sp'])
]).T

def calc_rmse(pred, expect):
    return np.sqrt(((pred - expect) ** 2).mean())

i = 0
for train_index, test_index in kf.split(train8):
    print('\nFold index:', i, '-----')
    X_train = train8[train_index, 0:train8.shape[1]-1]
    y_train = train8[train_index, train8.shape[1]-1]
    
    X_test = train8[test_index, 0:train8.shape[1]-1]
    y_test = train8[test_index, train8.shape[1]-1]
      
    print('\nShape of input data for this fold:\n')
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('\nX_test:', X_test.shape)
    print('y_test:', y_test.shape)

    #model8.fit(X_train, y_train)
    model = sm.OLS(y_train, X_train, probability=True).fit()

    # evaluate on the test set for this fold
    rmse = calc_rmse(model.predict(X_test), y_test)
    results[i, 0] = rmse
    i += 1

print(pd.DataFrame(results))
print("Avg. RMSE:", results.mean())

#Convert the array predictions to a data frame
test_predictions = model8.predict(test).apply(exp)
print(test_predictions)
d = {'p_saleprice': test_predictions}
df1 = test[['index']]
df2 = pd.DataFrame(data=d)
output = pd.concat([df1, df2], axis=1, join_axes=[df1.index])
output.to_csv(
  'C:/Users/sgran/Desktop/northwestern/predict_410/assignment_2/hw02_predictions.csv'
) 
