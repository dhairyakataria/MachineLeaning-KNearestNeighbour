'''$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            K-Nearest Neighbors
                        Auther:-Dhairy Kataria
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'''
#Importing the Libraries
import numpy as np
import pandas as pd
import seaborn as sns

#Load Data From CSV File
df = pd.read_csv('teleCust1000t.csv')
df.info()                #Seeing if there is any null value present

sns.countplot(df['custcat'])            #VLet’s see how many of each class is in our data set


sns.countplot(x='custcat',hue='retire',data=df)

df.drop(['retire', 'gender'], axis=1)
#setting the independent and dependent variable
X = df.iloc[ : , 0:9].values
y = df['custcat'].values


#Data Standardization give data zero mean and unit variance, it is good practice, 
#especially for algorithms such as KNN which is based on distance of cases:
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

#Splitting data into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1)

#finding the best value of k for which the accuracy is maximum 
from sklearn.metrics import accuracy_score
ma = 0

for k in range (1,15):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier( algorithm='auto', p=2, n_neighbors=k).fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if(accuracy > ma):
        ma = accuracy

print(ma)