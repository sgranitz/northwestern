# Boston Housing Study (Python)
# using data from the Boston Housing Study case
# as described in "Marketing Data Science: Modeling Techniques
# for Predictive Analytics with R and Python" (Miller 2015)

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1
SET_FIT_INTERCEPT = True

# import base packages into the namespace for this program
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

# modeling routines from Scikit Learn packages
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn import metrics, cross_validation
from sklearn.model_selection import KFold

# read data for the Boston Housing Study
path = 'C:/Users/sgran/Desktop/northwestern/predict_422/assignment_3/jump-start-boston-housing-v001/'
boston_input = pd.read_csv(path + 'boston.csv')

# check the DataFrame object boston_input
print('\nboston DataFrame (first and last five rows):')
print(boston_input.head())
print(boston_input.tail())

print('\nGeneral description of the boston_input DataFrame:')
print(boston_input.info())

# drop neighborhood from the data being considered
boston = boston_input.drop('neighborhood', 1)
print('\nGeneral description of the boston DataFrame:')
print(boston.info())

print('\nDescriptive statistics of the boston DataFrame:')
print(boston.describe())

boston.hist(figsize=(12,10))
boston.plot(
  kind='box', 
  subplots=True, 
  layout=(4,4), 
  sharex=False, 
  sharey=False, 
  figsize=(12,9)
)

# set up preliminary data for data for fitting the models 
# the first column is the median housing value response
# the remaining columns are the explanatory variables

cols = [
  'mv',
  'crim',
  'zn',
  'indus',
  'chas',
  'nox',
  'rooms',
  'age',
  'dis',
  'rad',
  'tax',
  'ptratio',
  'lstat'
]

for col in cols[1:]:
    fig, ax = plt.subplots()
    ax.set_ylabel('Median Housing Value')
    ax.set_xlabel(col)
    scatter_plot = ax.scatter(
      data=boston,
      y=cols[0], 
      x=col,
      facecolors='none', 
      edgecolors='green'
    ) 
    plt.show() 

prelim_model_data = np.array(boston[cols])

# dimensions of the polynomial model X input and y response
# preliminary data before standardization
print('\nData dimensions:', prelim_model_data.shape)

# standard scores for the columns along axis 0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(prelim_model_data))

# show standardization constants being employed
print(scaler.mean_)
print(scaler.scale_)

# the model data will be standardized form of preliminary model data
model_data = scaler.fit_transform(prelim_model_data)

for i in np.arange(1, model_data.shape[1]):
    fig, ax = plt.subplots()
    ax.set_ylabel('Median Housing Value')
    ax.set_xlabel(cols[i])
    scatter_plot = ax.scatter(
      y=model_data[0], 
      x=model_data[i],
      facecolors = 'green', 
      edgecolors = 'green'
    ) 
    plt.show() 

# dimensions of the polynomial model X input and y response
# all in standardized units of measure
print('\nDimensions for model_data:', model_data.shape)

names = [
  'Linear_Regression', 
  'Ridge_Regression', 
  'Lasso_Regression', 
  'ElasticNet_Regression'
] 

regressors = [
  LinearRegression(fit_intercept=SET_FIT_INTERCEPT), 
  Ridge(alpha=1, solver='cholesky', fit_intercept=SET_FIT_INTERCEPT, 
        normalize=False, random_state=RANDOM_SEED),
  Lasso(alpha=0.1, max_iter=10000, tol=0.01, 
        fit_intercept=SET_FIT_INTERCEPT, random_state=RANDOM_SEED),
  ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=0.01, 
             fit_intercept=SET_FIT_INTERCEPT, normalize=False, 
             random_state=RANDOM_SEED)
]

N_FOLDS = 10
# set up numpy array for storing results
results = np.zeros((N_FOLDS, len(names)))

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state=RANDOM_SEED)
# check the splitting process by looking at fold observation counts

# fold count initialized 
i = 0  
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', i,
          '------------------------------------------')
   
    X_train = model_data[train_index, 1:model_data.shape[1]]
    y_train = model_data[train_index, 0]
    
    X_test = model_data[test_index, 1:model_data.shape[1]]
    y_test = model_data[test_index, 0]   
    
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)
    
    print('X_test:  ', X_test.shape)
    print('y_test:  ', y_test.shape)

    j = 0
    for name, model in zip(names, regressors):
        print('\nRegression model evaluation for:', name)
        print('  Scikit Learn method:', model)
        model.fit(X_train, y_train)  # fit on the train set for this fold
        print('Fitted regression intercept:', model.intercept_)
        print('Fitted regression coefficients:', model.coef_)
 
        # evaluate on the test set for this fold
        y_test_predict = model.predict(X_test)
        print('Coefficient of determination (R-squared):',
              r2_score(y_test, y_test_predict))
        fold_method_result = sqrt(mean_squared_error(y_test, y_test_predict))
        print(model.get_params(deep=True))
        print('Root mean-squared error:', fold_method_result)
        results[i, j] = fold_method_result
        
        plt.figure()
        plt.scatter(x=y_test, y=y_test_predict)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
                  ls="--", c=".3")
        plt.title(name + " " + str(i+1))
        plt.xlabel("Actual")
        plt.ylabel("Estimate")
        plt.show()
        
        j += 1
    i += 1

results_df = pd.DataFrame(results)
results_df.columns = names

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      'in standardized units (mean 0, standard deviation 1)\n',
      '\nMethod               Root mean-squared error', sep = '')     
print(results_df.mean())   

plt.figure(figsize=(8,6))
results_df.boxplot(rot=45)
