#!/usr/bin/env python
# coding: utf-8

# In[1]:

import datetime 

print(datetime.datetime.now())

#READ EXCEL AND CSV DATA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#READ all source csc/xlsx files into dataframes 

#from kaggle
flights_df_path = "C:\\_ML Projects\\CapStone Data\\flight-delays\\flights.csv"
airports_df_path = "C:\\_ML Projects\\CapStone Data\\flight-delays\\airports.csv"
airlines_df_path = "C:\\_ML Projects\\CapStone Data\\flight-delays\\airlines.csv"

#my personal list of info 
ap_df_path = "C:\\_ML Projects\\CapStone Data\\Airport Operations Counts\\airport info.xlsx"

#custom reports using FAA-available websites
ap_ops_df_path = "C:\\_ML Projects\\CapStone Data\\Airport Operations Counts\\2015 airport ops.xlsx"
reg_ops_df_path = "C:\\_ML Projects\\CapStone Data\\Airport Operations Counts\\2015 region ops.xlsx"


flights_df = pd.read_csv(flights_df_path, dtype={"ORIGIN_AIRPORT": np.str, "DESTINATION_AIRPORT": np.str}) #, index_col = 0)
airports_df = pd.read_csv(airports_df_path)
airlines_df = pd.read_csv(airlines_df_path)

ap_df = pd.read_excel(ap_df_path, sheet_name='Airports', dtype={"AirportID":np.str, "StateID": np.str, "Region" : np.str})
ap_df = ap_df[(ap_df['AirportID'] != 'nan') & (ap_df['AirportID'] != '0')]
ap_df = ap_df[['AirportID', 'StateID', 'Region']]

ap_ops_df = pd.read_excel(ap_ops_df_path, sheet_name='Airport Ops')
reg_ops_df = pd.read_excel(reg_ops_df_path, sheet_name='Region Ops')

#look at first rows
#flights_df.head(10)
#print(ap_df.columns)
#print(ap_df.shape)
#print(ap_df.head(10))


# In[2]:


#GLOBALLY SHARED METHODS

import math
import numpy as np

def CheckIfNullOrEmpty(myValue):
    
    retValue = 0
    
    if type(myValue) == float:
        if math.isnan(myValue):
            retValue = 1
    elif type(myValue) != str:
        if myValue == None:
            retValue = 1
    elif myValue == None:
        retValue = 1
    elif len(myValue.strip()) == 0:
        retValue = 1
    elif pd.isnull(myValue):
        retValue = 1
    elif type(myValue) == str and myValue.strip() == 'nan':
        retValue = 1
    else:
        retValue = 0
        
    return retValue 


def CheckIfMissing(mySeries):
    
    result = mySeries.apply(CheckIfNullOrEmpty).sum()
    
    return result



def CheckRow(mySeries):
    
    
    sumValue = mySeries.apply(CheckIfNullOrEmpty).sum()
    
    if sumValue >= 1:
        retValue = False
    else:
        retValue = True
        
    return retValue 


def IsDataPresent(myValue):
    
    retValue = True
    
    if type(myValue) == float:
        if math.isnan(myValue):
            retValue = False
    elif type(myValue) != str:
        if myValue == None:
            retValue = False
    elif myValue == None:
        retValue = False
    elif len(myValue.strip()) == 0:
        retValue = False
    elif pd.isnull(myValue):
        retValue = False
    elif type(myValue) == str and myValue.strip() == 'nan':
        retValue = False
    else:
        retValue = True
        
    return retValue 





def APCategory(value):

    if value < mid:
        result = 'MID'
    else:
         result = 'SMALL'
       
    return result 



def PredictThreshold(model, Xtestlr, threshold):
    
    prob = model.predict_proba(Xtestlr)[:,1]
    
    predictions = np.where(prob >= threshold, 1, 0)
    
    return predictions


# In[3]:


#replace some column null values in flight data
#FILTER OUT cancelled or diverted flights

flights_df['CANCELLATION_REASON'] = flights_df['CANCELLATION_REASON'].fillna('Not Provided')
flights_df.loc[:, ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']] = flights_df.loc[:, ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']].fillna(0) 

#don't include canceled or flights diverted to another airport
active_flights_df = flights_df[flights_df.CANCELLED  != 1]
active_flights_df = active_flights_df[active_flights_df.DIVERTED != 1]


# In[4]:


#REMOVE DATA COLUMNS THAT WONT INCLUDE IN THE ANALYSIS

#flightdata_df = active_flights_df.drop(columns={
#    'FLIGHT_NUMBER', 'TAIL_NUMBER', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
#'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN'}, inplace=False)

#keeping only first 1000 rows while in test mode. remove that restraint later
flightdata_df = active_flights_df[['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT','DESTINATION_AIRPORT',
                               'DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'DISTANCE']]    
#[0:1000]

airports_df = airports_df[['IATA_CODE', 'CITY','STATE']]
ap_df = airports_df.rename(columns={'IATA_CODE':'AirportID', 'CITY':'CityID','STATE':'StateID'}, inplace=False)

airlines_df = airlines_df[['IATA_CODE']]
airln_df = airlines_df.rename(columns={'IATA_CODE':'AirlineID'}, inplace=False)

ap_ops_df = ap_ops_df[['Facility', 'TotalOperations']]
ap_oper_df = ap_ops_df.rename(columns={'Facility':'AirportID', 'TotalOperations':'TotalOps'}, inplace=False)


# In[ ]:


#remove missing flight data
#dataframe.isnull().sum()

emptylist_df = flightdata_df.apply(CheckIfMissing, axis=0)
print('check if any missing data in result_df\n', emptylist_df)
print(flightdata_df.shape)

flights_df_nonempty = flightdata_df[flightdata_df.apply(CheckRow, axis=1)] 
emptylist_df = flights_df_nonempty.apply(CheckIfMissing, axis=0)
print('check if any missing data in result_df\n', emptylist_df)
print(flights_df_nonempty.shape)


# In[6]:


#REMOVE BOGUS AIRPORT INFO FLIGHT DATA ROWS
#get rid of numbered airports (conveniently they all start with the number 1 instead of a letter)
print(flights_df_nonempty.shape)
flights2_df = flights_df_nonempty[flights_df_nonempty['ORIGIN_AIRPORT'].astype(str).str[0] != '1']
print(flights2_df.shape)
flights2_df = flights2_df[flights2_df['ORIGIN_AIRPORT'].astype(str).str[0] != '1']
print(flights2_df.shape)


# In[7]:


##remove missing airport data
#print(airports_df.iloc[0:10, 0:16])
#print(airports_df.iloc[0:10, 16:-1])


emptylist_df = ap_df.apply(CheckIfMissing, axis=0)
print('check if any missing data in result_df\n', emptylist_df)
print(ap_df.shape)

airports_df_nonempty = ap_df[ap_df.apply(CheckRow, axis=1)] 
emptylist_df = airports_df_nonempty.apply(CheckIfMissing, axis=0)
print('check if any missing data in result_df\n', emptylist_df)
print(airports_df_nonempty.shape)


# In[8]:


#REMOVE AIRLINES MISSING DATA

emptylist_df = airln_df.apply(CheckIfMissing, axis=0)
print('check if any missing data in result_df\n', emptylist_df)
print(airln_df.shape)

airlines_df_nonempty = airln_df[airln_df.apply(CheckRow, axis=1)] 
emptylist_df = airlines_df_nonempty.apply(CheckIfMissing, axis=0)
print('check if any missing data in result_df\n', emptylist_df)
print(airlines_df_nonempty.shape)


# In[9]:


#REMOVE AIRPORT OPS MISSING DATA

emptylist_df = ap_oper_df.apply(CheckIfMissing, axis=0)
print('check if any missing data in result_df\n', emptylist_df)
print(ap_oper_df.shape)

ap_ops_df_nonempty = ap_oper_df[ap_oper_df.apply(CheckRow, axis=1)] 
emptylist_df = ap_ops_df_nonempty.apply(CheckIfMissing, axis=0)
print('check if any missing data in result_df\n', emptylist_df)
print(ap_ops_df_nonempty.shape)


# In[10]:


#JOIN data sets together

flights2_df
ap_ops_df_nonempty


#result_df = pd.merge(flights2_df,
#                 airports_df_nonempty[['AirportID', 'StateID']],
#                 left_on = 'ORIGIN_AIRPORT', right_on='AirportID', 
#                 how='right').rename(columns={'StateID':'Origin_State'})
                 
#result_df = pd.merge(result_df,
#                 airports_df_nonempty[['AirportID', 'StateID']],
#                 left_on = 'DESTINATION_AIRPORT', right_on='AirportID', 
#                 how='right').rename(columns={'StateID':'Destination_State'})


result_df = pd.merge(flights2_df,
                 ap_ops_df_nonempty[['AirportID', 'TotalOps']],
                 left_on = 'ORIGIN_AIRPORT', right_on='AirportID', 
                 how='inner').rename(columns={'TotalOps':'TotalOps_OriginAirport'})

result_df.drop(columns=['AirportID','ORIGIN_AIRPORT'], inplace=True)

result_df = pd.merge(result_df,
                 ap_ops_df_nonempty[['AirportID', 'TotalOps']],
                 left_on = 'DESTINATION_AIRPORT', right_on='AirportID', 
                 how='inner').rename(columns={'TotalOps':'TotalOps_DestinationAirport'})

result_df.drop(columns=['AirportID','DESTINATION_AIRPORT'], inplace=True)


#result_df['late'] = result_df['ARRIVAL_DELAY'] > 0 

result_df.loc[ result_df['ARRIVAL_DELAY'] > 0, 'late'] = 1
result_df.loc[ result_df['ARRIVAL_DELAY'] <= 0, 'late'] = 0

#flights_df.loc[ flights_df['DEPARTURE_DELAY'] == 0, 'late'] = 'On Time'
#flights_df.loc[ flights_df['DEPARTURE_DELAY'] > 0, 'late'] = 'Late'

#result_df = pd.merge(result_df,
#                 ap_ops_df_nonempty[['Facility', 'TotalOperations']],
#                 left_on = 'DESTINATION_AIRPORT', right_on='Facility', 
#                 how='right').rename(columns={'TotalOperations':'TotalOperations_DestinationAirport'})

#result_df['TotalOperations_OriginAirport'] = result_df['TotalOperations_OriginAirport'].astype(int)


# In[11]:


#note - try the library that Jeff recommended instead of this

result_df2 = pd.get_dummies(result_df['AIRLINE']) #,prefix=['AL_'])

result_df3 = pd.concat([result_df, result_df2], axis = 1)


# In[12]:


result_df3.head(3)


# In[13]:


#result_df.dtypes


#result_df = result_df[result_df['ORIGIN_AIRPORT'].astype(str).str[0] != '1']
result_df3[['late']] = result_df3[['late']].astype(int)
result_df3.dtypes
result_df3.head(3)
#result_df3


#result_df.shape #(1320, 11)
#flights2_df.shape #(1000,9)
#airports_df_nonempty.shape #(322,3)



#emptylist_df = flights2_df.apply(CheckIfMissing, axis=0)
#print('check if any missing data in result_df\n', emptylist_df)
#print(flights2_df.shape)

#test_df = flights2_df[flights2_df.apply(CheckRow, axis=1)] 
#emptylist_df = test_df.apply(CheckIfMissing, axis=0)
#print('check if any missing data in result_df\n', emptylist_df)
#print(test_df.shape)


# In[14]:


#result_df3.where(result_df3[['late']] == 1)
#result_df4 = result_df3.loc[result_df3['late'] == 1]
#result_df3

#result_df4.groupby('AIRLINE')['AIRLINE'].count()


result_df4 = result_df3.loc[result_df3['late'] == 1].groupby('AIRLINE')['AIRLINE'].count()
result_df4
#If you wanted to add frequency back to the original dataframe use transform to return an aligned index:
#df['freq'] = df.groupby('a')['a'].transform('count')


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[16]:


#split data input with the intent of predicting 'late'
#random_state - the seed used by the random number generator
# test_size is auto set to 0.25 since it wasn't defined

#data_df = result_df.drop(columns={'AIRLINE'}, inplace=True)

# Split the data into a training and test set.
#Xlr, Xtestlr, ylr, ytestlr = train_test_split(result_df[['TotalOps_OriginAirport','TotalOps_DestinationAirport']], 
#                                              result_df.late, random_state=5)
Xlr, Xtestlr, ylr, ytestlr = train_test_split(result_df3[['TotalOps_OriginAirport','TotalOps_DestinationAirport','AA','DL','EV','OO','UA','WN']], 
                                              result_df3.late, random_state=5)


# In[17]:


Xlr.shape
ylr.shape


# In[18]:


clf = LogisticRegression()
# Fit the model on the training data.
clf.fit(Xlr, ylr)


# In[19]:


# Print the accuracy from the testing data.
print(accuracy_score(clf.predict(Xtestlr), ytestlr))


# In[20]:


pd.Series(clf.predict_proba(Xtestlr)[:,1]).hist() #shows our predictions


# In[21]:


#y_pred = clf.predict(Xtestlr)
#y_pred
#Xtestlr

#y_pred = clf.predict([[379376, 314616]])

#y_pred
#ytestlr
#Xtestlr

df = Xtestlr
df = clf.predict(Xtestlr)
df


# In[22]:


predictions = PredictThreshold(clf, Xtestlr, 0.4)

print(accuracy_score(predictions, ytestlr))

predictions

#precision in recall 


# In[23]:


print(classification_report(predictions, ytestlr))

#can sample


# In[24]:


#RANDOM FORESTS

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import numpy as np


# In[25]:


#number of trees in the forest
n_estimators = [10, 50, 100]
#the function used to measure the quality of a split.
criterion = ['gini', 'entropy']

print(n_estimators)
print(np.ravel(n_estimators))
print(np.reshape(n_estimators, (-1,)))
print(np.reshape(n_estimators, (1, -1)))

n_estimators = np.reshape(n_estimators, (-1,))
n_estimators = np.ravel(n_estimators)

print(criterion)

param_grid = {'n_estimators':n_estimators, 'criterion':criterion}

rfc = RandomForestClassifier()

GSCV_rfc = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 5)


# In[31]:


from datetime import date

X = result_df3.drop(['late'], axis = 1)
X = X.drop(['AIRLINE', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY'], axis = 1)
Y = result_df3[['late']]




#print(X.head())
#print(Y.head())

print(date.today())
#GSCV_rfc.fit(X,Y) #a column vector was passed when a 1d array was expected. please shange shape of y to (n_samples,) using ravel()
GSCV_rfc.fit(X,np.ravel(Y)) #NOTE that this takes hours to run
print(GSCV_rfc.best_params_)
#print(date.today())


#store model in pickle file
import pickle
filename = 'C:\\temp\\FlightDelayModel.pik'
outfile = open(filename, 'wb')

pickle.dump(GSCV_rfc, outfile)
outfile.close()

#end pickle logic


#split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 100, test_size = 0.3, train_size = 0.7)


# In[33]:


from sklearn.metrics import accuracy_score

y_predict = GSCV_rfc.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_predict))

print("Report:", classification_report(y_test, y_predict))


# In[37]:


#peek inside the random forest model to see what had the greatest influences

importances = GSCV_rfc.best_estimator_.feature_importances_
#print(importances)

forest = GSCV_rfc.best_estimator_.estimators_
#print(forest)

print('tree count', len(GSCV_rfc.best_estimator_.estimators_))
print('feature importances', len(importances))
print('features', len(X.columns))


# In[38]:


std = np.std([tree.feature_importances_ for tree in forest], axis=0)
indices = np.argsort(importances)[::-1]
print("feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# In[39]:


#plot the feature importances of the forest
plt.figure()
plt.title("feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[40]:


#X.columns


print(datetime.datetime.now())