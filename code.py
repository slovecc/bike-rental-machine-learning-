'''
This Python code explores two machine learning algorithms available from the Python [scikit-learn] library.

The idea is to predict how many bikes will be rented each hour of a day, based on data including weather, time, temperature, etc.

The script can be run by typing "python code.py" on the command line. It is necessary to have in the
same folder the file data.csv with the data of input
'''
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas 
import csv
import time
import seaborn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression



def main():

    numvars, numdata = 15,10886
###### read from the file data.csv#####

    df=pandas.read_csv('data.csv', sep=',')
    date_heure=df.values[:,0]
 

############################################
#split the date_heure in data and heure
    new_items = []

    for item in date_heure:
        new_items.extend(item.split())
    data = new_items[0::2]
    heure = new_items[1::2]
#split the data into year, mount and day
#and split the heure into hour
    year=[]
    month=[]
    day=[]
    hour=[]
    for line in data:
        types = line.split("-")
        year.append(int(types[0]))
        month.append(int(types[1]))
        day.append(int(types[2]))

    for lines in heure:
        hours = lines.split(":")
        hour.append(int(hours[0]))
############################################
    season=df.values[:,1]
    holiday=df.values[:,2]
    workingday=df.values[:,3]
    weather=df.values[:,4]
    temp=df.values[:,5]
    atemp=df.values[:,6]
    humidity=df.values[:,7]
    wind=df.values[:,8]
    casual=df.values[:,9]
    registered=df.values[:,10]
    count=df.values[:,11]

#hist(hour)
    datas=np.column_stack((year,month,day,hour,season,holiday,workingday,weather,temp,atemp,humidity,wind,casual,registered,count))
## split randomly the data into train (the 80% of total dataset) and test (the remaining 20%) 
    np.random.shuffle(datas)
    sample=datas[:int(numdata*0.8)]
    test=datas[int(numdata*0.8):]
    X_train=sample[:,0:11]
    y_train=sample[:,14]
    X_test=test[:,0:11]
    y_test=test[:,14]


    print '------------------------------------\n'

    choose_model = raw_input('CHOOSE MODEL OF MACHINE LEARNING.\nYou can select any of the following MODELS:\n\n   SUPPORT VECTOR METHOD (SVC) \n   RANDOM FOREST (RFC) \n   \n\nENTER MODEL (type SVC or RFC): ')
  
    if choose_model == 'SVC':
       from sklearn import svm
       print '\n RUNNING SUPPORT VECTOR METHOD ---------------\n'


       choose_features = raw_input('CHOOSE MODEL OF SVC.\nYou can select any of the following features:\n\n   linear \n   rbf \n   \n\nENTER FEATURES: ')
       if choose_features == 'linear':
          start_time = time.time()
          print '\nRunning SVM Classifier with linear kernel \n'
          clf=svm.SVC(kernel='linear')
          clf.fit(X_train, y_train)
          y_pred=clf.predict(X_test)
       elif choose_features == 'rbf':
          print '\nRunning SVM Classifier with rbf kernel \n'
          start_time = time.time()
          clf=svm.SVC(kernel='rbf')
          clf.fit(X_train, y_train)
          y_pred=clf.predict(X_test)

######### compute the rmsle
       from sklearn.metrics import mean_squared_error
       from math import sqrt
       n=len(y_pred)

       summation_arg = (np.log(y_test.astype('float64')+1.) - np.log(y_pred.astype('float64')+1.))**2.
       rsme=sqrt(np.sum(summation_arg)/n)
       print ("--- %s rmsle ---" % rsme)
       print("--- %s time needed: seconds ---" % (time.time() - start_time))
    #    summation_arg = (np.log(count+1.) - np.log(true_count+1.))**2.
    #    rmsle = np.sqrt(np.sum(summation_arg)/n)


    if choose_model == 'RFC':
       print '\n RUNNING RANDOM FOREST METHOD ---------------\n'
       from sklearn.ensemble import RandomForestClassifier
       from sklearn import svm
       start_time = time.time()
       clf = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)

       y_pred=clf.predict(X_test)

       from sklearn.metrics import mean_squared_error
       from math import sqrt
       n=len(y_pred)

       summation_arg = (np.log(y_test.astype('float64')+1.) - np.log(y_pred.astype('float64')+1.))**2.
       rsme=sqrt(np.sum(summation_arg)/n)

       print ("--- %s rmsle ---" % rsme)
       print("--- %s time needed: seconds ---" % (time.time() - start_time))

       print '\n PRINT THE IMPORTANCE OF FEATURES ---------------\n'
   
       imp=clf.feature_importances_

       names=['year','month','day','hour','season','holiday','workingday','weather','temp','atemp','humidity','wind']
       imp,names = zip(*sorted(zip(imp,names)))
       plt.barh(range(len(names)),imp,align='center')
       plt.yticks(range(len(names)),names)
       plt.xlabel('importance of feautures')
       plt.ylabel('features')
       plt.show()
    


main()



