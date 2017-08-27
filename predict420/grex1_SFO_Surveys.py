# Stephan Granitz [ GrEx1 ]

# Import libraries------------------------------------------------------------
import pandas as pd
import numpy as np
import pickle

# Grab files------------------------------------------------------------------
folder = "C:/Users/sgran/Desktop/GrEx1/"
io14 = folder + "sfo2014_data_file.xlsx"
io15 = folder + "sfo2015_data_file.csv"
io16 = folder + "sfo2016_data_file.xlsx"
sr_all = folder + "select_resps.csv"

sfo14 = pd.read_excel(io14, sheetname=1)
sfo15 = pd.read_csv(io15)
sfo16 = pd.read_excel(io16)
select_resps = pd.read_csv(sr_all)

# Part 1----------------------------------------------------------------------
# Create a single summary data set of ratings data
cols = list([    # Selected columns for ratings data
  'RESPNUM',     # Respondent Number (Automatically generated upon data entry) 
  'Q16LIVE',     # Live in... 
  'Q7ART',       # Artwork and exhibitions 
  'Q7FOOD',      # Restaurants 
  'Q7STORE',     # Retail shops and concessions 
  'Q7SIGN',      # Signs and directions inside SFO 
  'Q7WALKWAYS',  # Escalators/Elevators/Moving walkways
  'Q7SCREENS',   # Information on screens/monitors 
  'Q7INFODOWN',  # Information booths (lower level - near baggage claim) 
  'Q7INFOUP',    # Information booths (upper level - departure area) 
  'Q7WIFI',      # Accessing and using free WiFi at SFO 
  'Q7ROADS',     # Signs and directions on SFO airport roadways
  'Q7PARK',      # Airport parking facilities
  'Q7AIRTRAIN',  # AirTrain 
  'Q7LTPARKING', # Long term parking lot shuttle (bus ride) 
  'Q7RENTAL',    # Airport Rental Car Center
  'Q7ALL',       # SFO Airport as a whole 
  'Q9BOARDING',  # Boarding areas
  'Q9AIRTRAIN',  # Airtrain 
  'Q9RENTAL',    # Airport Rental Car Center 
  'Q9FOOD',      # Airport restaurants 
  'Q9RESTROOM',  # Restrooms 
  'Q9ALL',       # Overall cleanliness 
  'Q10SAFE',     # How safe do you feel at SFO? 
  'Q12PRECHECKRATE', # TSA Pre-check vs regular security line rating 
  'Q13GETRATE',  # How would you rate your experience getting to the airport? 
  'Q14FIND',     # Finding your way around airport 
  'Q14PASSTHRU'  # Passing through security and screening 
])
# Special cases:
# RESPNUM : named '*RESPNUM' in 2016 dataset
# Q12PRECHECKRATE : named 'Q12PRECHEKCRATE in 2015 dataset

sfo14_pt1 = sfo14.ix[:, cols]
sfo14_pt1.insert(loc=1, column='YEAR', value=2014, allow_duplicates=True)

cols[cols.index('Q12PRECHECKRATE')] = 'Q12PRECHEKCRATE'
sfo15_pt1 = sfo15.ix[:, cols]
sfo15_pt1.insert(loc=1, column='YEAR', value=2015, allow_duplicates=True)
sfo15_pt1.rename(columns={'Q12PRECHEKCRATE': 'Q12PRECHECKRATE'}, inplace=True)

cols[cols.index('RESPNUM')] = '*RESPNUM'
cols[cols.index('Q12PRECHEKCRATE')] = 'Q12PRECHECKRATE'
sfo16_pt1 = sfo16.ix[:, cols]
sfo16_pt1.insert(loc=1, column='YEAR', value=2016, allow_duplicates=True)
sfo16_pt1.rename(columns={'*RESPNUM': 'RESPNUM'}, inplace=True)

sfo_dfs = [sfo14_pt1, sfo15_pt1, sfo16_pt1]
df_pt1 = pd.concat(sfo_dfs)
df_pt1.replace('Blank', 0, inplace=True)

df_pt1.info()
df_pt1.groupby('YEAR').agg(['count', 'mean', 'std'])

path = folder + 'df_pt1.csv'
df_pt1.to_csv(path, header=True)

# Part 2----------------------------------------------------------------------
# Identify the top three (3) comments made in survey years 2015 and 2016
cols15 = list(['Q8COM1', 'Q8COM2', 'Q8COM3'])
cols16 = list(['Q8COM', 'Q8COM2', 'Q8COM3', 'Q8COM4', 'Q8COM5'])
sfo15_pt2 = sfo15.ix[:, cols15]
sfo16_pt2 = sfo16.ix[:, cols16]
sfo16_pt2.rename(columns={'Q8COM': 'Q8COM1'}, inplace=True)

df_pt2 = sfo15_pt2.append(sfo16_pt2)
num_people = df_pt2.shape[0]
df_pt2_sgl = pd.DataFrame(df_pt2.values.reshape(-1, 1), columns=['Q8COM'])
rel_freq = df_pt2_sgl.apply(pd.value_counts) / num_people
rel_freq.head()

## Top 3 comments (0 is nonresponse, this can be skipped)
# 999: Good experience/keep up the good work/other positive comment 
# 202: Need more places to eat/drink/more variety in types of restaurants
# 103: Going through security takes too long/add more checkpoints 

# Part 3----------------------------------------------------------------------
# Using the data you created in 1, summarize the distribution of the 
# SFO Airport "as a whole" ratings by respondent residence location category 
df_pt1.groupby('Q16LIVE').Q7ALL.value_counts().unstack().fillna(0)

df_pt3 = df_pt1.groupby('Q16LIVE').Q7ALL.agg([
    'count', 'mean', 'std']).drop(df_pt1.index[[4]])
print(df_pt3)

# Q7ALL: Ratings
# 5: Outstanding, 1: Unacceptable, 0: Blank
# 6: Have never used or visited / Not applicable
# Q16LIVE: Locations (Note: 4 is likely an input error)
# 1: 9 County Bay Area,  2: Northern California outside the Bay Area  
# 3: In another region,  0: Blank/Multiple responses

# Part 4----------------------------------------------------------------------
# Profile respondents for follow-up participation in qualitative research by 
# creating a data set that describes them.
cols2 = list([
  'RESPNUM',      # Respondent ID
  'INTDATE',      # Date and time interviewed
  'DESTGEO',      # Destination geographic area
  'DESTMARK',     # Size of destination market
  'Q2PURP1',      # Purpose(s) of travel 
  'Q2PURP2',      # Purpose(s) of travel 
  'Q2PURP3',      # Purpose(s) of travel 
  'Q3PARK',       # Used parking?
  'Q4BAGS',       # Checked baggage?
  'Q4STORE',      # Purchased from a store?
  'Q4FOOD',       # Purchased in a restaurant?
  'Q4WIFI',       # Used free WiFi?
  'Q5TIMESFLOWN', # Times Flown in last 12 mo. 
  'Q5FIRSTTIME',  # First time flying out of SFO?
  'Q6LONGUSE',    # How long using SFO?
  'Q16LIVE',      # Residence Location?
  'Q18AGE',       # Age   
  'Q19GENDER',    # Gender  
  'Q20INCOME',    # Income
  'Q21FLY',       # Fly 100K miles or more per year?
  'LANG',         # Language version of questionnaire
  'Q22SJC',       # Have used the San Jose airport
  'Q22OAK'        # Have used the Oakland airport
])
# Special cases: In sfo16, Age is Q19 and those following are all +1

sfo15_pt4 = sfo15.ix[:, cols2]
sfo15_pt4.insert(loc=1, column='YEAR', value=2015, allow_duplicates=True)

cols2[cols2.index('RESPNUM')] = '*RESPNUM'
a = cols2.index('Q18AGE')
cols2[a::] = ['Q19AGE', 'Q20GENDER', 'Q21INCME', 'Q22FLY', 
              'LANG', 'Q23SJC', 'Q23OAK']  
sfo16_pt4 = sfo16.ix[:, cols2]
sfo16_pt4.insert(loc=1, column='YEAR', value=2016, allow_duplicates=True)
sfo16_pt4.rename(columns={
    '*RESPNUM': 'RESPNUM', 'Q19AGE': 'Q18AGE', 'Q20GENDER': 'Q19GENDER',
    'Q21INCME': 'Q20INCOME', 'Q22FLY': 'Q21FLY', 'Q23SJC': 'Q22SJC', 
    'Q23OAK': 'Q22OAK'}, inplace=True)
sfo16_pt4['INTDATE'] = pd.to_datetime(
    sfo16_pt4.INTDATE, format='%d').dt.day
    
# Adjust age answers to given scale
sfo16_pt4.loc[sfo16_pt4['Q18AGE'].str.contains(
  'Know', na=False), 'Q18AGE'] = 0
sfo16_pt4.loc[sfo16_pt4['Q18AGE'].str.contains(
  'Under', na=False), 'Q18AGE'] = 0
sfo16_pt4.replace({
  '65-Over': 7, '55-64': 6, '45-54': 5, '35-44': 4, 
  '25-34': 3, '18-24': 2}, inplace=True)
# Adjust gender answers to given scale
sfo16_pt4.loc[sfo16_pt4['Q19GENDER'].str.contains(
  'Female', na=False), 'Q19GENDER'] = 2
sfo16_pt4.loc[sfo16_pt4['Q19GENDER'].str.contains(
  'Male', na=False), 'Q19GENDER'] = 1

sfo_dfs = [sfo15_pt4, sfo16_pt4]
df_pt4 = pd.concat(sfo_dfs)
df_pt4.replace('Blank', 0, inplace=True)
select_resps.rename(columns={'year': 'YEAR'}, inplace=True)
df_pt4_flt = df_pt4.merge(select_resps)
path = folder + 'df_pt4.csv'
df_pt4_flt.to_csv(path, header=True)

df_pt4_flt.describe()

df_pt4_flt.groupby([
    'Q19GENDER', 'Q18AGE']).agg(['count', 'mean', 'std']).head()

print(
  df_pt4_flt.groupby('Q3PARK').Q3PARK.count(), '\n',
  df_pt4_flt.groupby('Q5TIMESFLOWN').Q5TIMESFLOWN.count(), '\n',
  df_pt4_flt.groupby('Q6LONGUSE').Q6LONGUSE.count()
)

# Part 5----------------------------------------------------------------------
# Save each of the data sets you have created above by "pickling."
part1_out = open('part_1', 'wb')
pickle.dump(df_pt1, part1_out)
part1_out.close()

part2_out = open('part_2', 'wb')
pickle.dump(df_pt2, part2_out)
part2_out.close()

part3_out = open('part_3', 'wb')
pickle.dump(df_pt3, part3_out)
part3_out.close()

part4_out = open('part_4', 'wb')
pickle.dump(df_pt4_flt, part4_out)
part4_out.close()

# Validate pickle worked
test_pickle = open('part_1', 'rb')
valid = pickle.load(test_pickle)
