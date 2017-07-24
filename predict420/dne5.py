# 1. Read the names of the json files from the directory;
# 2. Print out the json file names.

# Import packages:
import os
import glob
import magic
import json

# Set the working directory to the new GrEx. Print the working directory to confirm:
path = 'C:\\Users\Stephan\Desktop\GrEx3'
os.chdir(path)
print(os.getcwd())
# C:\Users\Stephan\Desktop\GrEx3

# Read list of files in the folder using the os package:
print(os.listdir())

# Various ways to pull out only the json files.
# 1. Use os package and create a dictionary by file type 
# (this method won't work if any file has a '.' in the file name):
files = os.listdir()
files_dict = dict(file.split('.') for file in files)
json_dict = {k:v for (k,v) in files_dict.items() if v == 'json'}
print(json_dict)

# 2. Use glob to get a list:
print(glob.glob('*.json'))

# 3. If the folder is huge and memory usage or speed is a concern, you can iterate over the files 
# in the folder without storing them to memory using iglob:

for json_file in glob.iglob('*.json'):
    print (json_file)

# Validate they are .json
# 1. Use mime / magic
mime = magic.Magic(mime=True)
print(mime.from_file('100506.json'))

# 2. Create a function to find the fake json:

def check_json(file):
    try:
        json.load(file)
    except ValueError:
        return False
    return True

for json_file in json_list:
    with open(json_file) as json_data:
        if check_json(json_data) == False:
            print('File named ' + json_file + ' is not really a json file.')
            
# File named fakeJSON.json is not really a json file.
