# Using Linear Regression to predict
# family home sale prices in Ames, Iowa

# Packages
import pandas as pd 
import numpy as np 
import statsmodels.formula.api as smf  
import matplotlib.pyplot as plt 
import seaborn as sns
from tabulate import tabulate
from statsmodels.iolib.summary2 import summary_col

# Set some options for the output
pd.set_option('display.notebook_repr_html', False) 
pd.set_option('display.max_columns', 40) 
pd.set_option('display.max_rows', 10) 
pd.set_option('display.width', 120)

# Read in the data
path  = 'C:/Users/sgran/Desktop/northwestern/predict_410/assignment_1/'
train = pd.read_csv(path + 'ames_train.csv')
test  = pd.read_csv(path + 'ames_test.csv')

# Convert all variable names to lower case
train.columns = [col.lower() for col in train.columns]
test.columns  = [col.lower() for col in test.columns]

# EDA
print('\n----- Summary of Train Data -----\n')
print('Object type: ', type(train))
print('Number of observations & variables: ', train.shape)

# Variable names and information
print(train.info())
print(train.dtypes.value_counts())

# Descriptive statistics
print(train.describe())
print(tabulate(
  train[[
    'saleprice',
    'yrsold',
    'yearbuilt',
    'overallqual',
    'grlivarea',
    'garagecars'
  ]].describe().round(1), 
  headers='keys', 
  tablefmt='psql'
))

# show a portion of the beginning of the DataFrame
print(train.head(10))
print(train.shape)

train.loc[:, train.isnull().any()].isnull().sum().sort_values(ascending=False)
train[train == 0].count().sort_values(ascending=False)

t_null = train.isnull().sum()
t_zero = train[train == 0].count()
t_good = train.shape[0] - (t_null + t_zero)
xx = range(train.shape[1])

plt.figure(figsize=(8,8))
plt.bar(xx, t_good, color='g', width=1,
        bottom=t_null+t_zero)
plt.bar(xx, t_zero, color='y', width=1,
        bottom=t_null)
plt.bar(xx, t_null, color='r', width=1)
plt.show()

print(t_null[t_null > 1000].sort_values(ascending=False))
print(t_zero[t_zero > 1900].sort_values(ascending=False))

drop_cols = (t_null > 1000) | (t_zero > 1900)
train = train.loc[:, -drop_cols]

# Some quick plots of the data
train.hist(figsize=(18,14))
train.plot(
  kind='box', 
  subplots=True, 
  layout=(5,9), 
  sharex=False, 
  sharey=False, 
  figsize=(18,14)
)
train.plot.scatter(x='grlivarea', y='saleprice')
train.boxplot(column='saleprice', by='yrsold')
train.plot.scatter(x='subclass', y='saleprice')
train.boxplot(column='saleprice', by='overallqual')
train.boxplot(column='saleprice', by='overallcond')
train.plot.scatter(x='overallcond', y='saleprice')
train.plot.scatter(x='lotarea', y='saleprice')

# Replace NaN values with medians in train data
train = train.fillna(train.median())
train = train.apply(lambda med:med.fillna(med.value_counts().index[0]))
train.head()

t_null = train.isnull().sum()
t_zero = train[train == 0].count()
t_good = train.shape[0] - (t_null + t_zero)
xx = range(train.shape[1])

plt.figure(figsize=(14,14))
plt.bar(xx, t_good, color='g', width=.8,
        bottom=t_null+t_zero)
plt.bar(xx, t_zero, color='y', width=.8,
        bottom=t_null)
plt.bar(xx, t_null, color='r', width=.8)
plt.show()

train.bldgtype.unique()
train.housestyle.unique()

# Goal is typical family home
# Drop observations too far from typical
iqr = np.percentile(train.saleprice, 75) - np.percentile(train.saleprice, 25)
drop_rows = train.saleprice > iqr * 1.5 + np.percentile(train.saleprice, 75)
train = train.loc[-drop_rows, :]

iqr = np.percentile(train.grlivarea, 75) - np.percentile(train.grlivarea, 25)
drop_rows = train.grlivarea > iqr * 1.5 + np.percentile(train.grlivarea, 75)
train = train.loc[-drop_rows, :]

iqr = np.percentile(train.lotarea, 75) - np.percentile(train.lotarea, 25)
drop_rows = train.lotarea > iqr * 1.5 + np.percentile(train.lotarea, 75)
train = train.loc[-drop_rows, :]

iqr = np.percentile(train.totalbsmtsf, 75) - np.percentile(train.totalbsmtsf, 25)
drop_rows = train.totalbsmtsf > iqr * 1.5 + np.percentile(train.totalbsmtsf, 75)
train = train.loc[-drop_rows, :]

# Replace 0 values with median to living area in train data
m = np.median(train.grlivarea[train.grlivarea > 0])
train = train.replace({'grlivarea': {0: m}}) 

# Discrete variables
plt.figure()
g = sns.PairGrid(train,
                 x_vars=["bldgtype",  
                         "exterqual",
                         "centralair", 
                         "kitchenqual", 
                         "salecondition"],
                 y_vars=["saleprice"],
                 aspect=.75, size=3.5)
g.map(sns.violinplot, palette="pastel");

# Print correlations
corr_matrix = train.corr()
print(corr_matrix["saleprice"].sort_values(ascending=False).head(10))
print(corr_matrix["saleprice"].sort_values(ascending=True).head(10))

## Pick 10 variable to focus on
pick_10 = [
  'saleprice',
  'grlivarea', 
  'overallqual', 
  'garagecars', 
  'yearbuilt', 
  'totalbsmtsf',
  'salecondition', 
  'bldgtype', 
  'kitchenqual', 
  'exterqual', 
  'centralair'
]

corr = train[pick_10].corr()
blank = np.zeros_like(corr, dtype=np.bool)
blank[np.triu_indices_from(blank)] = True
fig, ax = plt.subplots(figsize=(10, 10))
corr_map = sns.diverging_palette(255, 133, l=60, n=7, 
                                 center="dark", as_cmap=True)
sns.heatmap(corr, mask=blank, cmap=corr_map, square=True, 
vmax=.3, linewidths=0.25, cbar_kws={"shrink": .5})

# Quick plots
for variable in pick_10[1:]:
    if train[variable].dtype.name == 'object':
        plt.figure()
        sns.stripplot(y="saleprice", x=variable, data=train, jitter=True)  
        plt.show()
        
        plt.figure()
        sns.factorplot(y="saleprice", x=variable, data=train, kind="box") 
        plt.show()
    else:
        fig, ax = plt.subplots()
        ax.set_ylabel('Sale Price')
        ax.set_xlabel(variable)
        scatter_plot = ax.scatter(
          y=train['saleprice'], 
          x=train[variable],
          facecolors = 'none', 
          edgecolors = 'blue'
        ) 
        plt.show() 

plt.figure()
sns.factorplot(x="bldgtype", y="saleprice", col="exterqual", row="kitchenqual", 
  hue="overallqual", data=train, kind="swarm") 

plt.figure()
sns.countplot(y="overallqual", hue="exterqual", data=train, palette="Greens_d")

# Run simple models
model1 = smf.ols(formula='saleprice ~ grlivarea', data=train).fit() 
model2 = smf.ols(formula='saleprice ~ grlivarea + overallqual', data=train).fit()  
model3 = smf.ols(formula='saleprice ~ grlivarea + overallqual + garagecars' , data=train).fit()  
model4 = smf.ols(formula='saleprice ~ grlivarea + overallqual + garagecars + yearbuilt' , data=train).fit()  
model5 = smf.ols(formula='saleprice ~ grlivarea + overallqual + garagecars + yearbuilt + totalbsmtsf + kitchenqual + exterqual + centralair', data=train).fit()  
  
print('\n\nmodel 1----------\n', model1.summary())
print('\n\nmodel 2----------\n', model2.summary())
print('\n\nmodel 3----------\n', model3.summary())
print('\n\nmodel 4----------\n', model4.summary())
print('\n\nmodel 5----------\n', model5.summary())

out = [model1,
       model2,
       model3,
       model4,
       model5]
out_df = pd.DataFrame()
out_df['labels'] = ['rsquared', 'rsquared_adj', 'fstatistic', 'aic']

i = 0
for model in out:
    train['pred'] = model.fittedvalues
    plt.figure()
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


train['predictions'] = model5.fittedvalues
print(train['predictions'])

# Clean test data
test.info()
test[3:] = test[3:].fillna(test[3:].median())
test["kitchenqual"] = test["kitchenqual"].fillna(test["kitchenqual"].value_counts().index[0])
test["exterqual"] = test["exterqual"].fillna(test["exterqual"].value_counts().index[0])
m = np.median(test.grlivarea[test.grlivarea > 0])
test = test.replace({'grlivarea': {0: m}}) 

print(test)

# Convert the array predictions to a data frame then merge with the index for the test data
test_predictions = model5.predict(test)
test_predictions[test_predictions < 0] = train['saleprice'].min()
print(test_predictions)

dat = {'p_saleprice': test_predictions}
df1 = test[['index']]
df2 = pd.DataFrame(data=dat)

submission = pd.concat([df1,df2], axis = 1, join_axes=[df1.index])
print(submission)
