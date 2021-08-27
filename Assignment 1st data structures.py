#!/usr/bin/env python
# coding: utf-8

# # Prac 1: Data Structure & Modelling and Compilation

# In[2]:


my_dict={'Name':["a","b","c","d","e","f","g"],'Age':[20,27,35,45,55,43,35],'Designation':["VP","CEO","CFO","VP","VP","CEO","MD"]}
import pandas as pd
import numpy as np
df=pd.DataFrame(my_dict)
df


# In[3]:


df.to_csv('Csv_example')
df


# In[4]:


df.to_csv('Csv_example')
df


# In[5]:


## df_csv=pd.read_csv('Csv_example')
# df_csv
df.to_csv('Csv_Ex',index=False)
df_csv=pd.read_csv('Csv_Ex')
df_csv


# In[6]:


# Load data from csv file and display data without headers
import pandas as pd
Location= "C:/python/student-mat.csv"
df= pd.read_csv(Location,header=None)
df.head()


# In[7]:


import pandas as pd
Location= "C:/python/student-mat.csv"
df= pd.read_csv(Location)
df.head()


# In[8]:


import pandas as pd
Location= "C:/python/student-mat.csv"
# To add headers as we load the data
df= pd.read_csv(Location, names=['RollNo','Names','Grades'])
# To add headers to a dataframe
df.columns = ['RollNo','Names','Grades']
df.head()


# In[9]:


import pandas as pd
name = ['Mansi','Vinit','Pratiksha','Maya','Aaksh']
grade = [58,78,65,85,44]
bsc = [1,1,0,0,1]
msc = [1,2,0,0,0]
phd = [0,1,0,0,0]
Degrees = zip(name,grade,bsc,msc,phd)
columns = ['Names','Grades','BSC','MSC','PHD']
df = pd.DataFrame(data = Degrees, columns=columns)
df


# In[10]:


# Loading Data from Excel File and changing column names
import pandas as pd
Location= "C:/python/gradedata.xlsx"
df = pd.read_excel(Location)


# In[11]:


#Changing column names
df.columns = ['First','Last','Sex','Age','Exer','Hrs','Grd','Address']
df.head()


# In[12]:


import pandas as pd
name = ['Mansi','Vinit','Pratiksha','Maya','Aaksh']
grade = [58,78,65,85,44]
GradeList = zip(name,grade)
df = pd.DataFrame(data = GradeList,columns=['Names','Grades'])

writer = pd.ExcelWriter('dataframe.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()


# In[14]:


# Load Data from sqlite
import sqlite3
con = sqlite3.connect("C:/python/portal_mammals.sqlite")
cur = con.cursor()

for row in cur.execute('SELECT * FROM species;'):
    print(row)
con.close()


# In[15]:


import sqlite3
con = sqlite3.connect("C:/python/portal_mammals.sqlite")
cur = con.cursor()
cur.execute('SELECT plot_id FROM plots WHERE plot_type="Control"')
print(cur.fetchall())
cur.execute('SELECT species FROM species WHERE taxa="Bird"')
print(cur.fetchone())
con.close()


# In[16]:


import pandas as pd
import sqlite3

con = sqlite3.connect("C:/python/portal_mammals.sqlite")
df = pd.read_sql_query("SELECT * from surveys", con)

print(df.head())
con.close()


# In[17]:


# Saving Data to SQL
from pandas import DataFrame
Cars = {'Brand':['honda','Toyota','Audi '],
        'Price':[22000,25000,35000]} 
df = DataFrame(Cars,columns=['Brand','Price'])
print(df)


# In[18]:


import sqlite3
conn= sqlite3.connect('TestDB1.db')
c=conn.cursor()


# In[19]:


c.execute('CREATE TABLE CARS(Brand text, Price number)')
conn.commit()


# In[20]:


df.to_sql('CARS',conn,if_exists='replace',index=False)
df


# In[21]:


c.execute('''
SELECT Brand,max(Price) from CARS
''')


# In[22]:


df= DataFrame(c.fetchall(),columns=['Brand','Price'])
df


# In[23]:


# Example 
import os
import sqlite3 as lite
from sqlalchemy import create_engine


# In[24]:


Student_id=[105,106,102,107]
SName=["Mansi","Vinit","Pratiksha","Utpal"]
LName=["Shah","Deewan","Jain","Mishra"]
Department=["DSAI","BScIT","BMS","BCom"]
Email=["mansi@gmail.com","vinit@gmail.com","pratiksha@gmail.com","utpla@gmail.com"]


# In[26]:


Studata = zip(Student_id,SName,LName,Department,Email)
df = pd.DataFrame(data = Studata, columns=['Student_id','SName','LName','Department','Email'])
df


# In[28]:


df1 = df.to_csv('Studata.csv',index=False,header=True)
df1
df2 = df.to_excel('Studata.xlsx',index=False,header=True)
df2


# In[29]:


db_filename = r'Studata.db'
con = lite.connect(db_filename)
df.to_sql('student',
con,
schema=None,
if_exists='replace',
index=True,
index_label=None,
chunksize=None,
dtype=None
)
con.close()

db_file = r'Studata.db'
engine = create_engine(r"sqlite:///{}" .format(db_file))
sql = 'SELECT * from Student'

studf = pd.read_sql(sql, engine)
studf


# In[2]:


# Data Preprocessing
import pandas as pd
import numpy as np
state=pd.read_csv("C:/python/US_violent_crime.csv")
state.head()


# In[3]:


def some_func(x):
 return x*2
state.apply(some_func)
state.apply(lambda n: n*2)


# In[4]:


state.transform(func = lambda x: x*10)


# In[5]:


mean_purchase = state.groupby('State')["Murder"].mean().rename("User_mean").reset_index()
print(mean_purchase)


# In[6]:


#checking missing values
print(state.isnull().sum())


# In[5]:


# Example 2
cols=['col0', 'col1', 'col2', 'col3', 'col4']
rows=['row0', 'row1', 'row2', 'row3', 'row4']
data=np.random.randint(0, 100, size=(5,5))
df=pd.DataFrame(data, columns=cols, index=rows)
df.head()


# In[9]:


df.iloc[3,2]


# In[6]:


import pandas as pd
import numpy as np
#dealing with 0 and NAN (not a number) comman way of representing missing value
df=pd.DataFrame(data, columns=cols, index=rows)
df.iloc[3,3]=0
df.iloc[1,2]=np.nan
df.iloc[4,0]=np.nan
df['col5']=0
df['col6']=np.nan
df.head()


# In[7]:


df.loc[:,df.all()]


# In[10]:


df.loc[:,df.any()]


# In[11]:


df.loc[:,df.isnull().any()]


# In[12]:


df.loc[:,df.notnull().any()]


# In[13]:


df.dropna(how="all",axis=1)


# In[14]:


df.dropna(how="all",axis=0)


# In[15]:


df.fillna(df.mean())


# In[16]:


df.fillna(df.max())


# In[17]:


df.fillna(df.sum())


# In[19]:


def fn(n):
    return n*2
df.apply(fn)


# In[20]:


def fn(n):
    return n + 50
df.apply(fn)


# In[21]:


df['new_col']=df['col3'].apply(lambda n:n*2)
df['new_col']


# In[22]:


df


# In[23]:


df=pd.DataFrame(np.array([[1,2,3],[4,5,6],[7,8,9]]),columns=['a','b','c'])
df


# In[24]:


df.apply(lambda n:n*10)


# In[28]:


#tranform function using pandas in python
import random
data=pd.DataFrame({
    'C':[random.choice(('a','b','c'))for i in range(1000000)],
    'A':[random.randint(1,10)for i in range(1000000)],
    'B':[random.randint(1,10)for i in range(1000000)]
})
data


# In[30]:


m=data.groupby('C')["A"].mean
m


# In[31]:


mean=data.groupby('C')["A"].mean().rename("D").reset_index()
mean


# In[32]:


df_1=data.merge(mean)
df_1


# In[42]:


import pandas as pd
import numpy as np
airline=pd.read_csv("‪C:/python/airline_stats.csv")
airline.head()


# In[1]:


import pandas as pd
import numpy as np
airline=pd.read_csv("C:/python/airline_stats.csv")
airline.head()


# In[6]:


print(airline.isnull().sum())


# In[5]:


df=airline.fillna(airline.mean(),inplace=True)
print(df)


# In[7]:


print(airline.isnull().sum())


# In[8]:


z=airline.groupby('pct_atc_delay')["pct_weather_delay"].mean().rename("user").reset_index()
print(z)


# In[9]:


airline.merge(z)


# In[10]:


def fn(x):
    return x*2
airline.apply(fn)

