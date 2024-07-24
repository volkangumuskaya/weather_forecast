print('Importing libs..')
#Import libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import pandas as pd
import os
import calendar
import re
from sklearn import datasets
import random
# import seaborn as sns
# import plotly.express as px
# import plotly.io as pio
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
# import datetime
# matplotlib.use('Qt5Agg')
# pio.renderers='svg'

# #Common pandas options for viewing
# pd.options.display.width = 100
# pd.set_option('display.max_columns', 40)
# pd.set_option('display.max_rows', 30)
# pd.set_option('max_colwidth', 250)
# pd.set_option('display.float_format', lambda x: '%.3f' % x)


# THESE DATA CAN BE USED FREELY PROVIDED THAT THE FOLLOWING SOURCE IS ACKNOWLEDGED:
# ROYAL NETHERLANDS METEOROLOGICAL INSTITUTE
# RD: 24-hour sum of precipitation in tenths of a millimeter from 08:00 UTC previous day to 08:00 UTC current day.
# SX: Snow cover code number at 08:00 UTC.

# #read dataset
# dir_path=os.getcwd()

print('Reading daily_rain_data.csv')
# filename='ehv_10_years.txt'
filename='daily_rain_data.csv'
df=pd.read_csv(filename)

print('Data preparation in process...')
df.columns = df.columns.str.strip()
df=df.rename(columns={'STN': "station", 'YYYYMMDD': "date",'RD':'rain_amount_in_tenth_mm','SX':'snow_code'})
df['rain_amount_mm']=df['rain_amount_in_tenth_mm']/10
df['year']=np.floor((df['date']/10000)).astype('int')
df['month']=(np.floor(df['date']/100)-np.floor((df['date']/10000))*100).astype('int')
df['day']=(df['date']-(np.floor(df['date']/100)*100)).astype('int')
df['daymonthyear']=df['date']
df['yearmonth']=df['year']*100+df['month']
df['date']=df['date'].astype('string')

#previous day rain in mm
df['previous_day_rain_mm']=df['rain_amount_mm'].shift(periods=1).fillna(0)

#some desc stats
df.describe()

#aggragate rains
df.groupby('month')['rain_amount_mm'].sum()
df.groupby('month')['rain_amount_mm'].mean()
df.groupby('year')['rain_amount_mm'].sum()
df.groupby(['year','month'])['rain_amount_mm'].sum()
df.groupby('yearmonth')['rain_amount_mm'].sum()

df['monthly_rain_mm']=df.groupby(['year','month'])[['rain_amount_mm']].transform('sum')

#seasons
#map one column to another
conditions = [(df['month'].isin ([12,1,2])),
              (df['month'].isin ([3,4,5])),
              (df['month'].isin ([6,7,8])),
            (df['month'].isin ([9,10,11]))
              ]
choices = ['Winter', 'Spring', 'Summer','Fall']

df['season'] = np.select(conditions, choices,default='na')

#month names
#map one column to another

df['month_name'] =df['month'].apply(lambda x: calendar.month_name[x])

print('Creating train and test set..')
#####form train-test set
lgbm_params = {
    'n_estimators': 10,  # 100, opt 1000
    'max_depth': 6,  # 6, opt 14
    'learning_rate': 0.5,  # 0.3
    'reg_alpha': 0.5,  # none
    'reg_lambda': 0,  # none,
    # 'monotone_constraints': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #This is used when we want monotonic constraints for example for regression wrt a feature
}

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

# get col types
# features = [x for x in df.columns if x!='target']
# cols = list(df.select_dtypes(include=['bool','object','category','string','float']).columns)
# cols.remove('target')


# Create X and y
df.index=np.arange(len(df))

X = df[features].copy()  # Features table
X.index=df.index.copy()
y = df[target]
y.index=df.index.copy()


# Split X and y randomly
X_train=X[X.index<=len(X)*0.8].copy()
X_test=X[X.index>len(X)*0.8].copy()

Y_train=y[y.index<=len(y)*0.8].copy()
Y_test=y[y.index>len(y)*0.8].copy()

Y_train_clf=pd.Series(np.where((Y_train<=0.1),0,1),index=Y_train.index,name='rain_occurrence')
Y_test_clf=pd.Series(np.where((Y_test<=0.1),0,1),index=Y_test.index,name='rain_occurrence')

# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.20,random_state=42)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
Y_train_clf.shape
Y_test_clf.shape

print('Training classification and regression models..')
# Fit model using predetermined parameters
# lgbr = lgb.LGBMRegressor(**lgbm_params)  # lgbr.get_params()
lgbr = lgb.LGBMRegressor()  # lgbr.get_params()
lgbr.fit(X_train, Y_train, eval_set=(X_test, Y_test), feature_name='auto', categorical_feature='auto')

lgbr_clf = lgb.LGBMClassifier()  # lgbr.get_params()
lgbr_clf.fit(X_train, Y_train_clf, eval_set=(X_test, Y_test_clf), feature_name='auto', categorical_feature='auto')
lgbr_clf.best_score_

# #Plot importance
# print('feature importance by gain')
# lgb.plot_importance(lgbr,importance_type='gain',figsize=(6,20),max_num_features=55)
# lgbr.feature_importances_
# plt.show()
#
# print('feature importance by split')
# lgb.plot_importance(lgbr,importance_type='split',figsize=(6,20),max_num_features=55)
# lgbr.feature_importances_
# plt.show()

print('Making predcitions..')
# make predictions
pred_test = lgbr.predict(X_test)
pred_train = lgbr.predict(X_train)

pred_test_clf = lgbr_clf.predict(X_test)
pred_train_clf = lgbr_clf.predict(X_train)

pred_test_probs=lgbr_clf.predict_proba(X_test)[:,1]
pred_train_probs=lgbr_clf.predict_proba(X_train)[:,1]

print('Creating comprehensive DataFrames..')
# predictions as df using index of X_test
pred_test_df = pd.DataFrame({'pred_rainfall':pred_test,
                             'pred_rain_occurrence':pred_test_clf,
                             'rain_probability':pred_test_probs
                             },index=X_test.index)
pred_train_df = pd.DataFrame({'pred_rainfall':pred_train,
                              'pred_rain_occurrence':pred_train_clf,
                              'rain_probability':pred_train_probs
                              },index=X_train.index)

#Accuracy on training and test set
test_df=pd.concat([X_test,Y_test,Y_test_clf, pred_test_df], axis=1)
train_df=pd.concat([X_train,Y_train,Y_train_clf, pred_train_df], axis=1)


test_df[['rain_occurrence','pred_rain_occurrence']].value_counts(normalize=True)
train_df[['rain_occurrence','pred_rain_occurrence']].value_counts(normalize=True)

#error
test_df['error']=test_df['pred_rainfall']-test_df['rain_amount_mm']
train_df['error']=train_df['pred_rainfall']-train_df['rain_amount_mm']

test_df['error'].mean()
test_df['pred_rainfall'].mean()

test_df['sample_type']='test'
train_df['sample_type']='train'


df_all=pd.concat([train_df,test_df])
df_all=pd.merge(df['date'],df_all,how='left',left_index=True,right_index=True)

df_all.to_csv('daily_rainfall_comprehensive.csv',index=False)
