import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
labelencoder = LabelEncoder()
import pickle

df = pd.read_excel('fake1.xlsx')
df.to_csv('csvfile4.csv', encoding='utf-8',index=False)
df4 = pd.read_csv('csvfile4.csv')

df = pd.read_excel('real1.xlsx')
df.to_csv('csvfile3.csv', encoding='utf-8',index=False)
df3 = pd.read_csv('csvfile3.csv')

df = pd.read_excel('real2.xlsx')
df.to_csv('csvfile2.csv', encoding='utf-8',index=False)
df2 = pd.read_csv('csvfile2.csv')

df = pd.read_excel('real3.xlsx')
df.to_csv('csvfile1.csv', encoding='utf-8',index=False)
df1 = pd.read_csv('csvfile1.csv')

df1 = df1.append(df2,ignore_index=True)
df1 = df1.append(df3,ignore_index=True)
df1 = df1.append(df4,ignore_index=True)

category_col =['caste1', 'degree', 'employed1', 'income1', 'mother1', 'occupation',
               'religion1','account']  

  
mapping_dict ={} 
for col in category_col: 
    df1[col] = labelencoder.fit_transform(df1[col]) 
  
    le_name_mapping = dict(zip(labelencoder.classes_, 
                        labelencoder.transform(labelencoder.classes_))) 
  
    mapping_dict[col]= le_name_mapping 

X = df1.drop(['account'],axis=1)
y = df1['account']
X_train ,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=101)
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
prediction1 = rfc.predict(X_test)
pickle.dump(rfc, open('model.pkl','wb'))





