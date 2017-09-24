# Background Materials
## The MSPA Software Survey was collected in December 2016. 
## Data from the survey were used to inform data science curriculum planning. 
## These data are provided in the comma-delimited text file <mspa-survey-data.csv>.

## The survey was conducted online using Survey Monkey. 
## A printed form of the survey items is provided in the file <mspa_software_survey.pdf>.

# Libraries
import pandas as pd                # data frame operations  
import numpy as np                 # arrays and math functions
import matplotlib.pyplot as plt    # static plotting
import seaborn as sns              # pretty plotting
import plotly.offline              # scatterplot matrix
import plotly.figure_factory as ff

# Options
np.set_printoptions(precision=3)
sns.set(style="whitegrid")

# Functions
def gather(df, key, value, cols):
    # Combine multiple columns into key/value columns
    id_vars = [col for col in df.columns if col not in cols]
    return pd.melt(df, id_vars, cols, key, value)

# Read data
valid_survey_input = pd.read_csv('mspa-survey-data.csv')

# Set Index to RespondentID
valid_survey_input.set_index(
  'RespondentID', 
  drop=True, 
  inplace=True
)

# Examine the structure of the DF
# Get understanding of size, columns, and data types for
# the survey data file
print('\nContents of initial survey data ---------------')
print(valid_survey_input.info())

# Print the first few rows to understand what the data looks like
print(valid_survey_input.head(5)) 

# Clean names for easier analysis
survey_df = valid_survey_input.rename(
  index=str, 
  columns={
    'Personal_JavaScalaSpark': 'java_per',
    'Personal_JavaScriptHTMLCSS': 'js_per',
    'Personal_Python': 'python_per',
    'Personal_R': 'r_per',
    'Personal_SAS': 'sas_per',
    'Professional_JavaScalaSpark': 'java_pro',
    'Professional_JavaScriptHTMLCSS': 'js_pro',
    'Professional_Python': 'python_pro',
    'Professional_R': 'r_pro',
    'Professional_SAS': 'sas_pro',
    'Industry_JavaScalaSpark': 'java_ind',
    'Industry_JavaScriptHTMLCSS': 'js_ind',
    'Industry_Python': 'python_ind',
    'Industry_R': 'r_ind',
    'Industry_SAS': 'sas_ind'
  }
)

# Convert course columns to 1/0 to check total courses
course_cols = [col for col in list(survey_df) if 
               col.startswith('PREDICT') or col.startswith('Other')]
survey_df[course_cols] = survey_df[course_cols].fillna(0)
for col in course_cols:
    survey_df[col] = (
      pd.to_numeric(survey_df[col], errors='coerce').fillna(1)
    )
    survey_df.loc[survey_df[col] > 1, col] = 1

# Find total courses taken by respondent if survey value is NA
survey_df['total_courses'] = survey_df[course_cols].sum(axis=1)
survey_df['Courses_Completed'] = survey_df['Courses_Completed'].fillna(
        survey_df['total_courses'])

print(survey_df.info())
print('\nDescriptive statistics for survey data ---------------')
print(survey_df.describe())

# Define subset DF for software preferences 
software_df = survey_df.loc[:, 'java_per':'sas_ind']
software_df['total_courses'] = survey_df['Courses_Completed']
software_df = pd.DataFrame(software_df.to_records())

# Make a PairGrid plot for overview of survey responses
g = sns.PairGrid(software_df.sort_values("total_courses", ascending=True),
                 x_vars=software_df.columns[1:-1], y_vars=["total_courses"],
                 size=10, aspect=.25)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h",
      palette="GnBu_d", edgecolor="black")
g.set(xlim=(0, 100), xlabel="Preference", ylabel="")

# Titles for the columns
titles = ['java_per', 'js_per', 'python_per', 'r_per', 'sas_per',
          'java_pro', 'js_pro', 'python_pro', 'r_pro', 'sas_pro',
          'java_ind', 'js_ind', 'python_ind', 'r_ind', 'sas_ind']

for ax, title in zip(g.axes.flat, titles):
    # Set a different title for each axes
    # Make the grid horizontal instead of vertical
    ax.set(title=title)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)
                  
# Scatter plot of two most popular languages (Python/R)
fig, ax = plt.subplots()
ax.set_xlabel('Personal Preference for R')
ax.set_ylabel('Personal Preference for Python')
plt.title('R and Python Perferences')
scatter_plot = ax.scatter(survey_df['r_per'], 
    survey_df['python_per'],
    facecolors = 'none', 
    edgecolors = 'blue') 
plt.show() 

# Gather preferences
pref_df = gather(software_df.copy(), 'software', 'pref',  
       ['java_per', 'js_per', 'python_per', 'r_per', 'sas_per',
        'java_pro', 'js_pro', 'python_pro', 'r_pro', 'sas_pro',
        'java_ind', 'js_ind', 'python_ind', 'r_ind', 'sas_ind'])
pref_df[['software','use']] = pref_df['software'].str.split('_', expand=True)

# Quick overview with scatter matrix    
fig = ff.create_scatterplotmatrix(
        pref_df.iloc[:, 1:], diag='histogram', index='software',
        height=800, width=800
)
plotly.offline.plot(fig, filename='scatter_matrix.html')

# Compare software preferences
sns.stripplot(x="pref", y="software", data=pref_df, jitter=True)  

# Software preference as factor of number of courses taken
sns.factorplot(x="total_courses", y="pref", col="software", row="use", 
               hue="total_courses", data=pref_df, kind="swarm")    

# Boxplot of software preferences    
sns.factorplot(x="software", y="pref", col="use", 
               data=pref_df, kind="box")  

# Bar plot of software preferences  
sns.factorplot(x="software", y="pref", col="use", 
               data=pref_df, kind="bar")  

# Cluster map of software preferences
sns.clustermap(software_df.iloc[:, 1:-1], 
               cmap="viridis", robust=True)    

pref_stats = pref_df.iloc[:, 2:].groupby(['software','use'], as_index=False)

# Descriptive statistics for software preference variables
print('\nDescriptive statistics for survey data ---------------')
print(pref_stats.aggregate([np.median, np.std, np.mean, np.max, np.sum]))

# Descriptive statistics for total courses
print('\nDescriptive statistics for courses completed ---------------')
print(survey_df['Courses_Completed'].describe().transpose())

# Interest in new courses
int_cols = [col for col in valid_survey_input.columns if 'Interest' in col]
interest_df = valid_survey_input[int_cols].copy()
interest_df = interest_df.rename(
  index=str, 
  columns={
    'Python_Course_Interest': 'python', 
    'Foundations_DE_Course_Interest': 'fnd_de',
    'Analytics_App_Course_Interest': 'analytics_app', 
    'Systems_Analysis_Course_Interest': 'sys_analysis'
  }
)
interest_df = pd.DataFrame(interest_df.to_records())

print('\nDescriptive statistics for interest in new courses ---------------')
print(interest_df.describe().transpose())

# Examine intercorrelations among course interests
plt.figure()
corr = interest_df.corr()
blank = np.zeros_like(corr, dtype=np.bool)
blank[np.triu_indices_from(blank)] = True
fig, ax = plt.subplots(figsize=(10, 10))
corr_map = sns.diverging_palette(255, 133, l=60, n=7, 
                                 center="dark", as_cmap=True)
sns.heatmap(corr, mask=blank, cmap=corr_map, square=True, 
            vmax=.3, linewidths=0.25, cbar_kws={"shrink": .5})

# Cluster map for course interest
plt.figure()
sns.clustermap(interest_df.iloc[:, 1:].fillna(0), 
               cmap="viridis", robust=True)

interest_df = gather(interest_df.copy(), 'class', 'pref',  
       ['python', 'fnd_de', 'analytics_app', 'sys_analysis'])

interest_stats = interest_df.iloc[:, 1:].groupby(['class'], as_index=False)
print('\nDescriptive statistics for interest in new courses ---------------')
print(interest_stats.aggregate([np.median, np.std, np.mean, np.max, np.sum]))

plt.figure()
# View course interest patterns
sns.stripplot(x="pref", y="class", data=interest_df, jitter=True)  
sns.factorplot(x="pref", y="class", data=interest_df, kind="box")   

# Examine intercorrelations among software preference variables
# Correlation matrix/heat map
plt.figure()
corr = software_df.iloc[:, :-1].corr()
blank = np.zeros_like(corr, dtype=np.bool)
blank[np.triu_indices_from(blank)] = True
fig, ax = plt.subplots(figsize=(10, 10))
corr_map = sns.diverging_palette(255, 133, l=60, n=7, 
                                 center="dark", as_cmap=True)
sns.heatmap(corr, mask=blank, cmap=corr_map, square=True, 
            vmax=.3, linewidths=0.25, cbar_kws={"shrink": .5})
