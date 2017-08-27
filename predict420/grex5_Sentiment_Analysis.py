# Stephan Granitz [ GrEx 5 ]

import pandas as pd
import numpy as np
import json
import os
import shelve
import matplotlib
import matplotlib.pyplot as plt
from afinn import Afinn

# Read what's in the hotel file 100506.json
folder = 'C:/Users/sgran/Desktop/GrEx3/'
os.chdir(folder)
io1 = folder + '100506.json'
data = json.load(open(io1))

reviews = data['Reviews']

# Get required values from JSON and create DF
reviews_df = pd.io.json.json_normalize(reviews)

# afinn sentiment score for the reviewer's written comment
afinn = Afinn()
score = []
for index, row in reviews_df.iterrows():
    score.append(afinn.score(row['Content']))
reviews_df['afinn_score'] = score

# Create and save in a shelve database a DF
path = folder + 'afinn_scores.csv'
reviews_df.to_csv(path, header=True)

afinn_scores_shelf = shelve.open('afinn_scores_shelf.dbm')
afinn_scores_shelf['afinn'] = reviews_df
afinn_scores_shelf.sync()
afinn_scores_shelf.close()

# Calculate descriptive statistics for the sentiment score
reviews_df.afinn_score.describe()

reviews_df.head(4).transpose()

avg_rating = []
rtg_cols = [col for col in list(reviews_df) if col.startswith('Ratings')]
for index, row in reviews_df.iterrows():
    rate = []
    for col in reviews_df[rtg_cols]:
        rate.append(pd.to_numeric(row[col], errors='coerce'))
    avg_rating.append(np.nanmean(rate))
    
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = [9, 4]

x = avg_rating
y = reviews_df['afinn_score']
plt.scatter(x, y, color = 'g', marker = '.')
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.title('Afinn Score as a function of Average Rating');
