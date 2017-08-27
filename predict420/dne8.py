import pandas as pd
import numpy as np
import json
import os
import timeit

folder = 'C:/Users/sgran/Desktop/GrEx3/'
os.chdir(folder)
io1 = folder + '100506.json'
data = json.load(open(io1))

def BL2DF(reps):
    y = []
    i = 0
    while i < reps:
        x = pd.io.json.json_normalize(
                data,  
                record_path = ['Reviews'],
                meta = ['HotelInfo']
            )
        y.append(x)
        i += 1
    df = pd.DataFrame()
    for x in range(len(y)): 
        df = df.append(y[x]).reset_index(drop = True)
    return df
    
timeit.timeit('BL2DF(100)', setup="from __main__ import BL2DF", number = 2000)
# 433.140783980086

def DF2DF(reps):
    i = 0
    df = pd.DataFrame()
    while i < reps:
        x = pd.io.json.json_normalize(
                data,  
                record_path = ['Reviews'],
                meta = ['HotelInfo']
            )
        df = df.append(x).reset_index(drop = True)
        i += 1
    return df

timeit.timeit('DF2DF(100)', setup="from __main__ import DF2DF", number = 10)
# 422.84023664324195

reps = 100
y = []
i = 0
while i < reps:
    x = pd.io.json.json_normalize(
            data,  
            record_path = ['Reviews'],
            meta = ['HotelInfo']
        )
    y.append(x)
    i += 1
df = pd.DataFrame()
for x in range(len(y)): 
    df = df.append(y[x]).reset_index(drop = True)
df.info()

i = 0
df = pd.DataFrame()
while i < reps:
    x = pd.io.json.json_normalize(
            data,  
            record_path = ['Reviews'],
            meta = ['HotelInfo']
        )
    df = df.concat(x).reset_index(drop = True)
    i += 1
df.info()

df = DF2DF(100)
df.info()

i = 0
df = pd.DataFrame(columns=[])
while i < 100:
    x = pd.io.json.json_normalize(
            data,  
            record_path = ['Reviews'],
            meta = ['HotelInfo']
        )
    df = pd.concat([x, df], axis = 0)
    i += 1
df

