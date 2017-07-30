# Stephan Granitz [ GrEx3 ]
# Import libraries
import pandas as pd
import numpy as np
import json
import glob
import os
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Starter: Get a Look At Some of The Hotel Data
# Read what's in the hotel file 100506.json
folder = 'C:/Users/sgran/Desktop/GrEx3/'
os.chdir(folder)
io1 = folder + '100506.json'
data = json.load(open(io1))
print(type(data), ' - ', data.keys())

reviews = data['Reviews']
info = data['HotelInfo']
print(type(reviews), ' - ', type(info))
print(info.keys())
print(info['HotelID'], ' - ', len(reviews), ' - ', type(reviews[0]))
print(reviews[0].keys())
print(reviews[0]['Ratings'])

pd.io.json.json_normalize(reviews).head(3).transpose().head(8)

# Part 1: Numeric Peceptual Mapping Data
def check_json(file):
    try:
        json.load(file)
    except ValueError:
        return False
    return True

def check_key(dictionary, key):
    if key in dictionary.keys():
        return dictionary[key]
    return np.nan
    
json_list = glob.glob('*.json')
# (a) Build it as a pandas DataFrame. 
# DataFrame should have a row for each hotel review
hotel_rev = pd.DataFrame()
for json_file in json_list:
    with open(json_file) as json_data:
        if check_json(json_data) == False:
            json_list.remove(json_file)
            next
            
    path = folder + json_file
    data = json.load(open(path))
    info = data['HotelInfo']
    hotel_name = check_key(info, 'Name')
    hotel_id = check_key(info, 'HotelID')
    
    if pd.isnull(hotel_name):
        hotel_name = re.compile(r'(?<=\-)([A-Z].*?)(?=\-)').search(
            info['HotelURL']).group().replace('_', ' ')
    
    reviews = data['Reviews']
    revs_df = pd.io.json.json_normalize(reviews)
    revs_df['HotelName'] = hotel_name
    revs_df['HotelID'] = hotel_id
        
    hotel_rev = hotel_rev.append(revs_df)
hotel_rev.replace('-1', np.nan, inplace=True)

print(hotel_rev.info())
print(hotel_rev['HotelName'].unique())
hotel_rev.head(3).transpose().head(6)

cols = ['HotelID', 'HotelName', 'Date', 'ReviewID', 'Author']
rtg_cols = [col for col in list(hotel_rev) if col.startswith('Ratings')]
cols += rtg_cols

# (b) Report the number of reviews for each hotel in the DataFrame.
reviews_df = hotel_rev[cols]
print(reviews_df['HotelName'].value_counts())

# (b) Calculate and report statistics describing 
# the distribution of the overall rating received by the hotels.
rating_stats = pd.DataFrame()
for col in reviews_df[rtg_cols]:
    rating_stats[col] = pd.to_numeric(reviews_df[col], errors='coerce')
rating_stats.describe().transpose()

# (c) Save your DataFrame by pickling it, 
# and verify that your DataFrame was saved correctly.
reviews_out = open('hotel_reviews', 'wb')
pickle.dump(reviews_df, reviews_out)
reviews_out.close()

test_pickle = open('hotel_reviews', 'rb')
valid = pickle.load(test_pickle)

# Part 2: Text Data for Perceptual Mapping
stop_words = set(stopwords.words('english'))

# 1 Create one string of the contents of comments about the hotel
# 2 Clean the string of all html tags, punctuation, etc.
# 3 Convert the string to a list of words.
# 4 Remove all stop words from the list
# 5 You might want to do word stemming
# 6 Create a dict fom the list in which the keys 
# are the "content" words, and their values are 
# the number of times each word occurs.

hotel_dict = {}
for hotel in hotel_rev['HotelID'].unique():
    temp_df = hotel_rev.loc[hotel_rev['HotelID'] == hotel]
    words = temp_df['Content'].str.cat(sep=' ')
    words_nohtml = re.compile('<.*?>').sub('', words)
    words_az = word_tokenize(
        re.compile('[^a-zA-Z]').sub(' ', words_nohtml).lower())
    
    words_filtered = []
    ps = PorterStemmer()
    for word in words_az:
        if word not in stop_words:
            words_filtered.append(ps.stem(word))
    
    content_dict = {}
    for word in words_filtered:
        if word in content_dict:
            content_dict[word] += 1
        else:
            content_dict[word] = 1
    hotel_dict[hotel] = content_dict
    
# Create for each hotel a dict of comment content words and 
# their frequencies, the counts of their occurrences.
# Add each of the hotel content word dicts to a dict with their 
# hotel IDs as the keys.
print(hotel_dict.keys())

print(type(hotel_dict), ' - ', type(hotel_dict['100506']),
      ' - ', len(hotel_dict['100506']))
      
# Write this dict to a json file, 
# and verify that it is written correctly.
with open('hotel.json', 'w') as hd:
    json.dump(hotel_dict, hd)
with open('hotel.json') as json_data:
        if check_json(json_data) == True:
            print('It worked.')

# Report the number of unique content words 
# in each of the hotel's dicts
num_words = {}
for hotel in hotel_dict:
    num_words[hotel] = len(hotel_dict[hotel])
print(num_words)
