
import pandas as pd
import re
import io
import os
import requests
from requests.exceptions import HTTPError
import numpy as np
import datetime

#set end date to 1 month ago
end_date=int((datetime.date.today()-datetime.timedelta(days=30)).strftime('%Y%m%d'))
start_date=int((datetime.date.today()-datetime.timedelta(days=31)).strftime('%Y%m%d'))

#go back one month because for some reason knmi website does not provide the current month data
#since this is just to try github action capabilities, and not make real time predcitions that will be used, it s ok
#better than causing data leakage
#it will simulate how system would work one month ago

data = {
    'start': start_date,
    'end': end_date,
    'vars': 'ALL',
    'stns': '919',
}

URLS = ["https://daggegevens.knmi.nl/klimatologie/monv/reeksen"]

for url in URLS:
    try:
        response = requests.post('https://www.daggegevens.knmi.nl/klimatologie/monv/reeksen', data=data)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred : {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    else:
        print("Success with status code",response.status_code)

print("preparing downloaded data")
##Parse response as df

s=str(response.content)# response as string
s=s[s.rindex('#')+1:] #find last # and delete the first part
s=s.replace('\\n','\n') #replace //n with /n
s=s.replace(' ','') #delete space
s=s.replace("'",'') #delete '
df=pd.read_csv(io.StringIO(s), sep=",")
type(df)


#Convert data to the expected format by LGBM model
df.columns = df.columns.str.strip()
df=df.rename(columns={'STN': "station", 'YYYYMMDD': "date",'RD':'rain_amount_in_tenth_mm','SX':'snow_code'})
df['rain_amount_mm']=df['rain_amount_in_tenth_mm']/10
df['year']=np.floor((df['date']/10000)).astype('int')
df['month']=(np.floor(df['date']/100)-np.floor((df['date']/10000))*100).astype('int')

#previous day rain in mm
df['previous_day_rain_mm']=df['rain_amount_mm'].shift(periods=1).fillna(0)

#seasons
#map one column to another
conditions = [(df['month'].isin ([12,1,2])),
              (df['month'].isin ([3,4,5])),
              (df['month'].isin ([6,7,8])),
            (df['month'].isin ([9,10,11]))
              ]
choices = ['Winter', 'Spring', 'Summer','Fall']

df['season'] = np.select(conditions, choices,default='na')

# Define features and target
df.columns
features = ['year', 'month', 'previous_day_rain_mm', 'season']
target = "rain_amount_mm"

# Change string and object type columns to category for LGBM
df[features].dtypes
for col in df.columns:
    col_type = df[col].dtype
    if col_type == 'object' or col_type.name == 'string':
        df[col] = df[col].astype('category')
print(df.dtypes)

print("loading rainfall models")
import pickle
with open('rainfall_models.pickle', 'rb') as handle:
    clf_model,reg_model = pickle.load(handle)

print("Making predcitions")
rain_occurrence_prediction=clf_model.predict(df[features])
rain_amount_mm_prediction=reg_model.predict(df[features])
chance_of_rain_prediction=clf_model.predict_proba(df[features])[:,1]

df_test = pd.DataFrame({'chance_of_rain_prediction':chance_of_rain_prediction,
                        'rain_occurrence_prediction':rain_occurrence_prediction,
                        'rain_amount_mm_prediction':rain_amount_mm_prediction
                        })
df=pd.concat([df,df_test],axis=1)
path='files/daily_prediction.csv'
print('saving to path:',path)
df.to_csv(path,index=False)
