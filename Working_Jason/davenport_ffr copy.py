#Importing required packages.
from asyncio.windows_events import NULL
from datetime import date
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import datetime

# Input data sets
DATA_PATH = "C:/Users/JCTew/Documents/RNL_FFR_Data/"
df = pd.read_csv(DATA_PATH + "DavenportLeadData.csv")

#Data Set Information
df.head()
df.info()
df.columns

df.fillna()

#Drop#
df = df.drop(columns=['CAMPUS_LOCATION',
       'UTM_SOURCE', 'UTM_MEDIUM', 'UTM_CAMPAIGN', 'UTM_CONTENT', 'UTM_TERM',
       'GOOGLE_ANALYTICS_ID', 'GCLID', 'FBCLID', 'SESSION_ID', 'TIMESTAMP',
       'VARIANT', 'IP_ADDRESS', 'PAGE_UUID', 'PAGE_URL', 'PAGE_NAME', 'CITY',
       'REGION', 'ZIP_CODE', 'COUNTRY', 'GENDER', 'RACE', 'HISPANIC',
       'CITIZENSHIP_PRIMARY','AGE'], axis=1)

# print dataframe
print("Dataframe")
display(df)
  
# shift column 'Name' to first position
first_column = df.pop('CRM_REFERENCE_ID')
  
# insert column using insert(position,column_name,
# first_column) function
df.insert(0, 'CRM_REFERENCE_ID', first_column)

fourth_column = df.pop('DATE_INQ')
  
df.insert(4, 'DATE_INQ', fourth_column)

df = df.rename(columns={"STARTED_CLASSES": "CLASS_STARTS"})

#FLAG_INQUIRY#
df['flag_inquiry'] = np.where(df.DATE_INQ.notnull(), 1, 0)

#FLAG_APP_CREATED#
df['flag_app_create'] = np.where(df.APPLICATION_CREATE_DATE.notnull(), 1, 0 &
                        np.where(df.DATE_INQ.notnull(), 1, 0))


#FLAG_APP_SUBMITTED#
df['flag_app_submit'] = np.where(df.APPLICATION_SUBMIT_DATE.notnull(), 1, 0 &
                            np.where(df.APPLICATION_CREATE_DATE.notnull(), 1, 0 &
                            np.where(df.DATE_INQ.notnull(), 1, 0)))

#FLAG_APP_DECISION#
df['flag_app_decided'] = np.where(df.APPLICATION_DECISION_DATE.notnull(), 1, 0 &
                            np.where(df.APPLICATION_SUBMIT_DATE.notnull(), 1, 0 &
                            np.where(df.APPLICATION_CREATE_DATE.notnull(), 1, 0 &
                            np.where(df.DATE_INQ.notnull(), 1, 0 ))))

#FLAG_APP_DECISION#
df['flag_enrolled'] = np.where(df.ENROLLED_DATE.notnull(), 1, 0 &
                            np.where(df.APPLICATION_DECISION_DATE.notnull(), 1, 0 &
                            np.where(df.APPLICATION_SUBMIT_DATE.notnull(), 1, 0 &
                            np.where(df.APPLICATION_CREATE_DATE.notnull(), 1, 0 &
                            np.where(df.DATE_INQ.notnull(), 1, 0 )))))
                            
df.to_csv(DATA_PATH + 'DavenportFFRFlags.csv')
                            