# Stephan Granitz [ GrEx4 ]
# Import libraries

import pandas as pd
import numpy as np
import pprint
import pickle
import re
import operator
import matplotlib
from elasticsearch import Elasticsearch, helpers 
import matplotlib.pyplot as plt

# Client to connect to Enron index on SSCC
client = Elasticsearch('http://enron:spsdata@129.105.88.91:9200')

# Query all messages on Enron index on SSCC 
query = {"query" : {"match_all" : {}}} 

# Scan Enron index and pull all email messages
scan_results = helpers.scan(
    client = client, query = query, 
    scroll = "10m", timeout = "10m",
    index = "enron", doc_type = "email"
)
meta = [resp['_source'] for resp in scan_results]

# Dump the messages into a pandas DF
enron_mail = pd.DataFrame()
for hit in meta:
    mail = pd.io.json.json_normalize(hit)
    enron_mail = enron_mail.append(mail)
enron_mail = enron_mail.reset_index(drop = True)

# Save DF in case anything is lost
enron_mail.to_csv('enron.csv')
enron_df = open('enron', 'wb')
pickle.dump(enron_mail, enron_df)
enron_df.close()

# Clean DF
enron_mail.drop(
    ['size',
     'aggregations.top-senders.terms.field',
     'aggregations.top-senders.terms.order._term'],
    axis = 1, inplace = True
)
enron_mail.replace(r'\s+', np.nan, regex = True).replace('', np.nan)
enron_mail.info()

{col:len(enron_mail[col].unique()) for col in enron_mail}

# Try various searches to find Ken Lay email addresses
enron_mail['mailbox'].unique()
enron_mail[enron_mail['headers.X-From'].str.contains('ken lay', case=False) == True]['headers.From'].unique()
enron_mail[enron_mail['headers.X-To'].str.contains(
  'ken lay', case=False) == True]['headers.To'].unique()
add_list = ['lay', 'ken', 'chairman']
pd.DataFrame([a for a in enron_mail['headers.From'] if any(
  name in str(a) for name in add_list)])[0].unique()

en_to = enron_mail['headers.To']
en_to_adds = []
for word in en_to:
    en_to_adds.append(word)
all_to = [str(add_single).split(',', 1)[0] for add_single in en_to_adds]
all_en_to = [x for x in all_to if any(name in x for name in add_list)]
pd.DataFrame(all_en_to)[0].unique()

# Compile emails that look to be Ken Lay's
# no.address@enron.com has emails from Ken signed by him,
# but this may be multiple accounts 
kl_email = [
  'ssskenneth.lay@enron.com', 'kenneth_lay@enron.com', 'kennethlay@enron.com', 
  'klay@enron.com', 'ken_lay@enron.net', 'kenlay@enron.com', 
  'kenneth.l.lay@enron.com', 'chairman.ken@enron.com', 'kenneth.enron@enron.com', 
  'ken.lay-.chairman.of.the.board@enron.com', 'ken.lay-@enron.com', 
  'no.address@enron.com', 'chairman.ken@enron.com', 'ken.skilling@enron.com',
  'ken.board@enron.com', 'ken.lay@enron.com', 'k_lay@enron.com', 'ken.lay-@enron.com', 
  'kenneth.lay@enron.com', 'ken.communications@enron.com', 'enron.chairman@enron.com', 
  'lay.kenneth@enron.com', 'k.lay@enron.com', 'k.l.lay@enron.com'
]

# 1. Extract the messages from the enron index that include 
#    a Ken Lay email address in them in a To: or From: message header. 
#    How many email messages are these? 

# All emails sent by Ken
from_ken = enron_mail[enron_mail['headers.From'].str.contains(
    '|'.join(kl_email), na=False)]

# All emails sent to Ken
to_ken = enron_mail[enron_mail['headers.To'].str.contains(
    '|'.join(kl_email), na=False)]

print("Emails to or from Ken: ", len(from_ken) + len(to_ken))

# 2. How many different Ken Lay email addresses are there in these messages? 
#    Provide a count of how many times each one occurs in the messages.

to_from = {}
for i in range(len(kl_email)):
    num = len(enron_mail[enron_mail['headers.To'].str.contains(kl_email[i], na=False)])
    num += len(enron_mail[enron_mail['headers.From'].str.contains(kl_email[i], na=False)])
    to_from[kl_email[i]] = num
print(to_from)

# 3. Provide counts of how many of the messages are to Ken Lay, 
#    and are from Ken Lay.

print("From Ken: ", len(from_ken), " and To Ken: ", len(to_ken))

# 4. a) Who did he receive the most from? 
#    How many did he receive from this sender?

to_ken.groupby('headers.From')['body'].count().reset_index(
    name='count'
).sort_values(['count'], ascending=False).head(5)

# 4. b) Who did Lay send the most emails to? 
#    How many did he send to this recipient?

people = from_ken['headers.To'].dropna().tolist()
new_list = []
for email in people:
    list_people = re.sub('[\s+]', '', email).split(',')
    new_list += list_people

dict_of_ppl = {x:new_list.count(x) for x in set(new_list)}
pd.DataFrame(sorted(dict_of_ppl.items(), key=operator.itemgetter(1), reverse=True)[0:5])

# 5. Did the volume of emails sent by Lay increase or decrease after Enron filed for bankruptcy? 
#    How many did he send before the filing? How many, after?

from_ken['headers.Date'] = pd.to_datetime(from_ken['headers.Date']).sort_values()
from_ken['Day'] = from_ken['headers.Date'].apply(lambda x: "%d/%d/%d" % (x.year, x.month, x.day))
n_emails = from_ken.groupby('Day')['body'].count()

vals = pd.DataFrame(n_emails).reset_index()
vals['Day'] = pd.to_datetime(vals['Day']).sort_values()
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = [9, 4]

x = vals['Day']
y = vals['body']
plt.plot(x, y)
plt.axvline(x='2001-12-02', color='red', linestyle='--')
plt.xticks([])
plt.title('Email Frequency');

a = sum(vals[vals['Day'] < '2001-12-02']['body'])
b = sum(vals[vals['Day'] >= '2001-12-02']['body'])
print("Before bankruptcy: ", a, " and After bankruptcy: ", b)

# 6. How many of the email messages in 4., above, 
#    mention Arthur Andersen, Enron's accounting firm?

aa = from_ken[from_ken['headers.To'].str.contains('all.worldwide@enron.com', na=False)]
aa = aa.append(to_ken[to_ken['headers.From'].str.contains('leonardo.pacheco@enron.com', na=False)])
print(
  'Arthur count: ', len(aa[aa['body'].str.contains('Arthur', na=False)]),
  '\nAndersen count: ', len(aa[aa['body'].str.contains('Andersen', na=False)])
)

# Below are any emails to or from Ken that contain 'Arthur Andersen'
art_and = to_ken[to_ken['body'].str.contains('Arthur Andersen', na=False)]
cols = ['body', 'headers.From', 'headers.To']
art_and[cols]

