# Stephan Granitz [ GrEx2 ]

# Import libraries
import pandas as pd
import numpy as np
import shelve
import sqlite3

# 1 Import each of the csv files you downloaded from the SSCC into a pandas DF
# Grab files
folder = "C:/Users/sgran/Desktop/GrEx2/"
io1 = folder + "seg6770cust.csv"
io2 = folder + "seg6770item.csv"
io3 = folder + "seg6770mail.csv"

# 1 a)
# Fill blanks with NaN for easier handling
customer = pd.read_csv(io1, low_memory=False).fillna(np.nan)
item = pd.read_csv(io2).fillna(np.nan)
mail = pd.read_csv(io3).fillna(np.nan)

# 1 b) print columns in item DF and first 4 records
list(item.columns.values)

item.head(4)

# 1 c) describe data types of cols in DFs
item.info(verbose=False)
mail.info(verbose=False)
customer.info(verbose=False)

# 2 Write each of you pandas DataFrames to a local SQLite DB named xyz.db. 
# Include only data for active buyers in these tables
active_customer = customer[customer.buyer_status == 'ACTIVE']

# Filter 'item' and 'mail' tables to only include active buyers
active_item = item[item['acctno'].isin(active_customer['acctno'])]
active_mail = mail[mail['acctno'].isin(active_customer['acctno'])]

# Connect to xyz.db
db = sqlite3.connect('xyz.db')

# Put DFs into the DB
active_customer.to_sql(
  'customer',
  db,
  if_exists='replace',
  index=False
)

active_item.to_sql(
  'item',
  db,
  if_exists='replace',
  index=False
)

active_mail.to_sql(
  'mail',
  db,
  if_exists='replace',
  index=False
)

# Commit the DB write
db.commit()

# Verify that you have written the tables to your SQLite DB correctly
cursor = db.cursor()
query = 'select * from customer limit 1'
res = cursor.execute(query)
res.fetchall()[0][0:10]
cursor.executescript('drop table if exists custSum;')
db.commit()

# 3 Using the same data from 2 above, create a new table called custSum
cursor.execute('''
  CREATE TABLE custSum(
    acctno TEXT PRIMARY KEY, zip INTEGER, zip4 INTEGER, heavy_buyer TEXT, 
    has_amex TEXT, has_disc TEXT, has_visa TEXT, has_mc TEXT, 
    est_income INTEGER, adult1_g TEXT, adult2_g TEXT
  )
''')
db.commit()

# Filter to the columns needed
cols = [
  'acctno', 'zip', 'zip4', 'ytd_sales_2009', 'amex_prem', 'amex_reg', 
  'disc_prem', 'disc_reg', 'visa_prem', 'visa_reg', 'mc_prem', 'mc_reg',
  'inc_scs_amt_v4', 'adult1_g', 'adult2_g'
]

custSum = active_customer[cols]
# Validate
custSum.head(3).transpose().head(6)

# 3 a) indicator of whether the customer is a 'heavy buyer,' where the definition 
# of a 'heavy buyer' is a customer whose YTD purchasing in 2009 is greater than 
# 90% of the 2009 YTD purchasing of all customers who are active buyers
heavy = custSum.ytd_sales_2009.dropna().quantile([0.9])[0.9]

custSum['heavy_buyer'] = 'N'
custSum.loc[custSum.ytd_sales_2009 > heavy, 'heavy_buyer'] = 'Y'

# 3 b) Add whether the customer has the following credit cards 
# (AMEX, DISC, VISA, MC)
custSum['has_amex'] = 'N'
custSum.loc[custSum.amex_prem == 'Y', 'has_amex'] = 'Y'
custSum.loc[custSum.amex_reg == 'Y', 'has_amex'] = 'Y'

custSum['has_disc'] = 'N'
custSum.loc[custSum.disc_prem == 'Y', 'has_disc'] = 'Y'
custSum.loc[custSum.disc_reg == 'Y', 'has_disc'] = 'Y'

custSum['has_visa'] = 'N'
custSum.loc[custSum.visa_prem == 'Y', 'has_visa'] = 'Y'
custSum.loc[custSum.visa_reg == 'Y', 'has_visa'] = 'Y'

custSum['has_mc'] = 'N'
custSum.loc[custSum.mc_prem == 'Y', 'has_mc'] = 'Y'
custSum.loc[custSum.mc_reg == 'Y', 'has_mc'] = 'Y'

# Drop columns no longer needed
custSum.drop(
  ['ytd_sales_2009', 'amex_prem', 'amex_reg', 'disc_prem', 'disc_reg', 
   'visa_prem', 'visa_reg', 'mc_prem', 'mc_reg'], inplace=True, axis=1
)

# 3 c,d,e) Est income, zip, acctno
custSum.rename(columns={'inc_scs_amt_v4': 'est_income'}, inplace=True)
custSum.est_income = custSum.est_income.astype(float)
custSum = custSum[[
  'acctno', 'zip', 'zip4', 'heavy_buyer', 'has_amex', 'has_disc',
  'has_visa', 'has_mc', 'est_income', 'adult1_g', 'adult2_g'
]]

# Fill the table in the DB
query = ''' 
  insert or replace into custSum 
  (acctno, zip, zip4, heavy_buyer, has_amex, has_disc, 
  has_visa, has_mc, est_income, adult1_g, adult2_g) 
  values (?,?,?,?,?,?,?,?,?,?,?) 
'''

# 3 f) count of the number of records in each table
query = 'select count(*) from '

res1 = cursor.execute(query + 'custSum')
print('Rows in custSum', res1.fetchall())

res2 = cursor.execute(query + 'customer')
print('Rows in customer', res2.fetchall())

res3 = cursor.execute(query + 'item')
print('Rows in item', res3.fetchall())

res4 = cursor.execute(query + 'mail')
print('Rows in mail', res4.fetchall())

# 3 g) Verify table written to SQLite DB correctly
query = 'select * from custSum limit 5'
res = cursor.execute(query)
res.fetchall()

# Close the db connection
db.close()

# 4 a) Target maketing with active buyers or lapsed buyers
marketing = customer[
    (customer.buyer_status == 'ACTIVE') | 
    (customer.buyer_status == 'LAPSED')
]

# 4 b) Find which categories each customer made purchases in
purchase = item.groupby(['acctno','deptdescr'], as_index=False)
purchase = purchase.aggregate(np.sum)

# 4 b) Indicator variable (1/0) for each product category customer made 
# at least one purchase
purchase_cats = purchase.pivot(
  index='acctno', columns='deptdescr', values='totamt'
)
# NaN means they didn't make any purchases
purchase_cats = pd.DataFrame(purchase_cats.to_records()).fillna(0)

def findSales (cat_list):
    for cat in cat_list:
        purchase_cats[cat] = purchase_cats[cat].apply(
            lambda x: 1 if (x > 0) else 0)
        
findSales(list(purchase_cats.columns.values)[1::])

# 4 c)  Include buyer status & total dollar amount of purchases
cols = ['acctno', 'buyer_status', 'ytd_sales_2009']
sales_info = marketing[cols].merge(purchase_cats)
sales_info.head(3).transpose()

# 4 d) Write your DataFrame to a csv file & store in a shelve database
path = folder + 'sales.csv'
sales_info.to_csv(path, header=True)

sales_shelf = shelve.open('sales_shelf.dbm')
sales_shelf['sales'] = sales_info
sales_shelf.sync()
sales_shelf.close()

# 4 e) Verify the shelve worked
sales_shelf = shelve.open('sales_shelf.dbm')
sales_shelf['sales'].head(3).transpose()

sales_shelf.close()

# 5 Report 6 most frequently purchased product cats by the gender of adult 1
# Add column to count number of adults in each category
purchase['count_adults'] = 1
cols = ['acctno', 'adult1_g']
purchase_gender = purchase.merge(marketing[cols]).groupby(
    ['adult1_g','deptdescr'], as_index=False)
purchase_gender = purchase_gender.aggregate(np.sum)

purchase_gender.drop('price', axis=1, inplace=True)
# List gender types
purchase_gender.adult1_g.unique()

# Print top 6 most purchased by gender
purchase_gender[purchase_gender['adult1_g'] == 'B'].sort(
    ['qty'], ascending=False).head(6)
    
purchase_gender[purchase_gender['adult1_g'] == 'F'].sort(
    ['qty'], ascending=False).head(6)
    
purchase_gender[purchase_gender['adult1_g'] == 'M'].sort(
    ['qty'], ascending=False).head(6)
    
purchase_gender[purchase_gender['adult1_g'] == 'U'].sort(
    ['qty'], ascending=False).head(6)
